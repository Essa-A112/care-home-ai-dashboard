# -*- coding: utf-8 -*-
"""
Care Home Investment Decision Dashboard

Author: Essa Abikar
MSc Robotics – King's College London

This Streamlit application presents the outputs of a machine learning
pipeline designed to identify UK locations with strong potential for
care home investment.

The dashboard integrates:
- Model predictions and investment scores
- Regional ROI projections
- SHAP-based feature explanations
- Interactive geographic visualisation of Local Authority Districts
- Natural language explanations generated from model outputs

The goal is to support transparent, data-driven decision making
for care home real estate investment.
"""

from __future__ import annotations

import os
import json
import time
import re
from typing import Dict, List, Tuple, Iterable, Optional

import pandas as pd
import streamlit as st
import folium

from difflib import get_close_matches
from streamlit_folium import st_folium


# ==========================================================
# APPLICATION CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="UK Care Home Investment Analysis",
    page_icon="🏡",
    layout="wide"
)

DATA_PATH = "final_model_data_with_grade.csv"
GEOJSON_PATH = "LAD_MAY_2025_Simplified_5.geojson"

SHAP_FOLDER = "shap_visuals"
GPT_FOLDER = "gpt_explanation"
ROI_FOLDER = "roi_gpt"

ROI_TABLE_PATH = "roi_by_district.csv"

# Optional directories used when additional analysis files exist
PER_LAD_SHAP_DIR = "SHAP/per_LAD"
FEATURE_GUIDE_JSON = "feature_guide.json"
FEATURE_GUIDE_TXT = "feature_guide.txt"
GLOBAL_LAD_SHAP_TOP = "lad_shap_top_drivers.csv"

ZONING_DIR_CANDIDATES = [
    "zoning_planning_summary",
    "zoning_reports",
    "Outputs/gpt_summaries",
]

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

OPENAI_MODEL_PRIMARY = "gpt-5.2"
OPENAI_MODEL_FALLBACK = "gpt-5-nano"

TEMPERATURE = 0.3
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2.0

DEFAULT_MAP_CENTER = [54.5, -3.0]
DEFAULT_MAP_ZOOM = 5


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def normalise(name: str) -> str:
    """Standardise names for consistent matching across datasets."""
    return (
        str(name)
        .strip()
        .lower()
        .replace("&", "and")
        .replace("'", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def safe_number(val, digits=2, default="N/A") -> str:
    """Safely format numerical values for display."""
    try:
        if pd.isna(val):
            return default
        return f"{float(val):.{digits}f}"
    except Exception:
        return default


def shap_available(key: str) -> bool:
    """Check whether a SHAP visualisation exists for a LAD."""
    return os.path.exists(os.path.join(SHAP_FOLDER, f"{key}.png"))


def best_match_norm(key_like: str, known: Iterable[str], cutoff: float = 0.84) -> Optional[str]:
    """
    Attempt to match a LAD name against known normalised keys.
    Handles suffix variations and approximate matches.
    """
    k = normalise(key_like)

    if k in known:
        return k

    for suf in ["_enriched", "_lad", "_district"]:
        if k.endswith(suf):
            base = k[:-len(suf)]

            if base in known:
                return base

            close = get_close_matches(base, list(known), n=1, cutoff=cutoff)
            if close:
                return close[0]

    close = get_close_matches(k, list(known), n=1, cutoff=cutoff)
    return close[0] if close else None


# ==========================================================
# DATA LOADING
# ==========================================================

@st.cache_data(show_spinner=False)
def load_base_data() -> pd.DataFrame:
    """
    Load the merged dataset used by the investment model.
    """
    if not os.path.exists(DATA_PATH):
        st.error(f"Missing data file: {DATA_PATH}")
        st.stop()

    df = pd.read_csv(DATA_PATH)

    if "Local_Authority" not in df.columns:
        st.error("Column 'Local_Authority' is missing from the model data.")
        st.stop()

    df = df.copy()
    df["norm_lad"] = df["Local_Authority"].apply(normalise)

    numeric_columns = [
        "Investment_Potential_Score",
        "Percent_65plus",
        "GDHI_per_head_2022",
        "House_Price_Growth_%",
        "Care_Homes_Count",
        "Care_Homes_per_10k",
        "%_CQC_Good",
        "%_CQC_RequiresImprovement",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_geojson() -> dict:
    """Load geographic boundary data for Local Authority Districts."""
    if not os.path.exists(GEOJSON_PATH):
        st.error(f"Missing GeoJSON: {GEOJSON_PATH}")
        st.stop()

    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_roi_table() -> pd.DataFrame:
    """Load ROI simulation outputs if available."""
    if not os.path.exists(ROI_TABLE_PATH):
        return pd.DataFrame(columns=["norm_lad", "ROI (%)"])

    roi = pd.read_csv(ROI_TABLE_PATH)

    if "District" in roi.columns:
        roi["norm_lad"] = roi["District"].apply(normalise)
    elif "LAD" in roi.columns:
        roi["norm_lad"] = roi["LAD"].apply(normalise)
    else:
        name_col = next((c for c in roi.columns if roi[c].dtype == object), None)

        if name_col:
            roi["norm_lad"] = roi[name_col].apply(normalise)
        else:
            roi["norm_lad"] = ""

    return roi


@st.cache_data(show_spinner=False)
def load_text_blob(folder: str) -> Dict[str, str]:
    """Load text summaries stored for individual LADs."""
    data = {}

    if not os.path.isdir(folder):
        return data

    for f in os.listdir(folder):

        if f.lower().endswith(".txt"):

            key = normalise(os.path.splitext(f)[0])

            try:
                with open(os.path.join(folder, f), "r", encoding="utf-8") as fh:
                    data[key] = fh.read()
            except Exception:
                pass

    return data


# ==========================================================
# OPENAI CLIENT
# ==========================================================

def call_openai(prompt: str, model: str) -> str:
    """
    Send a prompt to the OpenAI API and return the response.
    If the API fails, an error string is returned.
    """

    if not OPENAI_API_KEY:
        return "LLM error: no API key configured."

    try:

        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a UK care-home investment analyst. "
                        "Provide concise, decision-focused answers "
                        "based strictly on the supplied context."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"LLM error: {e}"


def ask_gpt_with_retry(prompt: str) -> str:
    """Retry the OpenAI request if the first attempt fails."""

    for attempt in range(MAX_RETRIES):

        output = call_openai(prompt, OPENAI_MODEL_PRIMARY)

        if not output.startswith("LLM error:"):
            return output

        time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

    return call_openai(prompt, OPENAI_MODEL_FALLBACK)


# ==========================================================
# STREAMLIT USER INTERFACE
# ==========================================================

st.title("🏡 Care Home Investment Dashboard (UK)")
st.caption("MSc Project – AI for Care Home Investment Support")

df = load_base_data()
geojson = load_geojson()
roi_table = load_roi_table()

roi_blobs = load_text_blob(ROI_FOLDER)
gpt_blobs = load_text_blob(GPT_FOLDER)


# ==========================================================
# SIDEBAR FILTERS
# ==========================================================

st.sidebar.header("Filters")

grades = sorted(df["Investment_Grade"].dropna().unique().tolist())

chosen_grades = st.sidebar.multiselect(
    "Investment grade",
    options=grades,
    default=grades
)

score_min, score_max = st.sidebar.slider(
    "Score range",
    0.0,
    100.0,
    (0.0, 100.0),
    step=1.0
)

filtered = df[
    df["Investment_Grade"].isin(chosen_grades)
    & df["Investment_Potential_Score"].between(score_min, score_max)
].copy()


# ==========================================================
# TABS
# ==========================================================

tab_overview, tab_map, tab_assistant = st.tabs(
    ["Overview", "Map & Details", "LLM Assistant"]
)


# ==========================================================
# OVERVIEW TAB
# ==========================================================

with tab_overview:

    st.subheader("National snapshot")

    c1, c2, c3 = st.columns(3)

    c1.metric("LADs in dataset", len(df))
    c2.metric("Median score", f"{df['Investment_Potential_Score'].median():.1f}")
    c3.metric("Top grade share", f"{(df['Investment_Grade'].eq('Good').mean()*100):.1f}%")

    st.markdown("#### Top 15 LADs by score")

    top = (
        filtered[
            [
                "Local_Authority",
                "Investment_Potential_Score",
                "Investment_Grade",
                "Percent_65plus",
                "%_CQC_Good",
                "House_Price_Growth_%"
            ]
        ]
        .sort_values("Investment_Potential_Score", ascending=False)
        .head(15)
    )

    st.dataframe(top, use_container_width=True)


# ==========================================================
# MAP TAB
# ==========================================================

with tab_map:

    st.subheader("Interactive Local Authority Map")

    df_map = filtered.copy()

    m = folium.Map(
        location=DEFAULT_MAP_CENTER,
        zoom_start=DEFAULT_MAP_ZOOM,
        tiles="cartodbpositron"
    )

    score_map = {
        r["Local_Authority"]: float(r["Investment_Potential_Score"])
        for _, r in df_map.iterrows()
    }

    folium.Choropleth(
        geo_data=geojson,
        data=pd.DataFrame({"LAD": list(score_map.keys()), "Score": list(score_map.values())}),
        columns=["LAD", "Score"],
        key_on="feature.properties.LAD25NM",
        fill_color="YlGnBu",
        nan_fill_color="#efefef",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Investment Potential Score",
    ).add_to(m)

    folium.GeoJson(
        geojson,
        style_function=lambda x: {"fillOpacity": 0, "weight": 0.3, "color": "#333"},
        highlight_function=lambda x: {"weight": 2, "color": "#111"},
        tooltip=folium.GeoJsonTooltip(fields=["LAD25NM"], aliases=["LAD:"]),
    ).add_to(m)

    map_output = st_folium(m, height=540, returned_objects=["last_active_drawing"])

    st.markdown("---")

    lad_names = df_map["Local_Authority"].sort_values().unique().tolist()

    if not lad_names:
        st.info("No LADs match the current filters.")
        st.stop()

    selected_lad = st.selectbox("Select Local Authority District:", lad_names)

    sel = df_map.loc[df_map["Local_Authority"] == selected_lad]

    if not sel.empty:

        r = sel.iloc[0]
        key = r["norm_lad"]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Score", f"{safe_number(r.get('Investment_Potential_Score'),2)} / 100")
        c2.metric("Grade", str(r.get("Investment_Grade")))
        c3.metric("% aged 65+", safe_number(r.get("Percent_65plus"),2))
        c4.metric("House price growth %", safe_number(r.get("House_Price_Growth_%"),2))

        shap_path = os.path.join(SHAP_FOLDER, f"{key}.png")

        st.markdown("##### SHAP explanation")

        if os.path.exists(shap_path):
            st.image(shap_path, width=800)
        else:
            st.info("No SHAP visualisation available.")


# ==========================================================
# LLM ASSISTANT
# ==========================================================

with tab_assistant:

    st.subheader("Investment Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, content in st.session_state.chat_history:
        st.chat_message(role).markdown(content)

    user_query = st.chat_input("Ask about any location or comparison")

    if user_query:

        st.chat_message("user").markdown(user_query)

        st.session_state.chat_history.append(("user", user_query))

        with st.spinner("Generating response..."):

            prompt = f"""
User question:
{user_query}

Available data relates to UK Local Authority District care home investment analysis.
Provide a concise and evidence-based response.
"""

            answer = ask_gpt_with_retry(prompt)

        st.chat_message("assistant").markdown(answer)

        st.session_state.chat_history.append(("assistant", answer))


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Created for MSc research on AI-assisted decision support for care home investment. "
    "Data includes demographic indicators, economic metrics, care quality ratings, "
    "and model-derived investment scores."
)
