# -*- coding: utf-8 -*-
"""
Care Home Investment Dashboard (GPT‑5 enhanced)
------------------------------------------------
Key improvements over the original:
• Clear module layout (config, data, ui, LLM) and stricter error handling.
• Faster loading via caching, single normalisation pass, and defensive file checks.
• Richer UX: sidebar filters, tabs, choropleth by score, comparison table.
• Safer, more deterministic LLM behaviour with a compact, grounded prompt.
• GPT‑5-ready client with retry logic, token‑aware context builder, and streaming fallback.
• Cleaner SHAP/ROI discovery with consistent name normalisation.
"""

from __future__ import annotations
import os
import json
import time
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# ----------------------------- #
#           CONFIG              #
# ----------------------------- #

st.set_page_config(
    page_title="Care Home Investment – UK",
    page_icon="🏡",
    layout="wide",
)

# File system
DATA_PATH = "final_model_data_with_grade.csv"
GEOJSON_PATH = "LAD_MAY_2025_Simplified_5.geojson"
SHAP_FOLDER = "shap_visuals"
GPT_FOLDER = "gpt_explanation"
ROI_FOLDER = "roi_gpt"
ROI_TABLE_PATH = "roi_by_district.csv"

# OpenAI
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
OPENAI_MODEL_PRIMARY = "gpt-5"
OPENAI_MODEL_FALLBACK = "gpt-4o-mini"  # used only if the primary call fails
TEMPERATURE = 0.2
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2.0

# UI constants
DEFAULT_MAP_CENTER = [54.5, -3.0]
DEFAULT_MAP_ZOOM = 5
SCORE_MIN, SCORE_MAX = 0.0, 100.0


# ----------------------------- #
#        UTILITIES & I/O        #
# ----------------------------- #

def normalise(name: str) -> str:
    """Normalise LAD-like names for file and key matching."""
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

@st.cache_data(show_spinner=False)
def load_base_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        st.stop()
    df = pd.read_csv(DATA_PATH)
    # Expected columns (defensive formatting)
    # Local_Authority, Investment_Potential_Score, Investment_Grade, Percent_65plus,
    # GDHI_per_head_2022, House_Price_Growth_%, Care_Homes_Count, Care_Homes_per_10k,
    # %_CQC_Good, %_CQC_RequiresImprovement
    if "Local_Authority" not in df.columns:
        st.error("Local_Authority column is missing from the model data.")
        st.stop()
    df = df.copy()
    df["norm_lad"] = df["Local_Authority"].apply(normalise)
    # Coerce numeric columns where applicable
    for col in [
        "Investment_Potential_Score",
        "Percent_65plus",
        "GDHI_per_head_2022",
        "House_Price_Growth_%",
        "Care_Homes_Count",
        "Care_Homes_per_10k",
        "%_CQC_Good",
        "%_CQC_RequiresImprovement",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_geojson() -> dict:
    if not os.path.exists(GEOJSON_PATH):
        st.error("GeoJSON file not found.")
        st.stop()
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_roi_table() -> pd.DataFrame:
    if not os.path.exists(ROI_TABLE_PATH):
        return pd.DataFrame(columns=["norm_lad", "ROI (%)"])
    roi = pd.read_csv(ROI_TABLE_PATH)

    # Always normalise from 'District'
    if "District" in roi.columns:
        roi["norm_lad"] = roi["District"].apply(normalise)
    else:
        roi["norm_lad"] = ""

    roi.columns = [c.strip() for c in roi.columns]
    return roi

@st.cache_data(show_spinner=False)
def load_text_blob(folder: str) -> Dict[str, str]:
    """Load .txt files from a folder into {norm_name: text}."""
    data: Dict[str, str] = {}
    if not os.path.isdir(folder):
        return data
    for f in os.listdir(folder):
        if f.lower().endswith(".txt"):
            key = normalise(os.path.splitext(f)[0])
            try:
                with open(os.path.join(folder, f), "r", encoding="utf-8") as fh:
                    data[key] = fh.read()
            except Exception:
                # Skip unreadable files
                continue
    return data

def shap_available(key: str) -> bool:
    if not os.path.isdir(SHAP_FOLDER):
        return False
    return os.path.exists(os.path.join(SHAP_FOLDER, f"{key}.png"))

def safe_number(val, digits=2, default="N/A") -> str:
    try:
        if pd.isna(val):
            return default
        return f"{float(val):.{digits}f}"
    except Exception:
        return default


# ----------------------------- #
#           OPENAI LLM          #
# ----------------------------- #

client = OpenAI(api_key=OPENAI_API_KEY)

def call_openai(prompt: str, model: str) -> str:
    """
    OpenAI Chat Completions caller using the new >=1.0.0 API.
    """
    if not OPENAI_API_KEY:
        return "No API key was found. Add OPENAI_API_KEY to Streamlit secrets or environment."

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a domain expert in UK care home investment analytics. "
                        "Write in a concise, board-ready style. Use only the supplied context. "
                        "If a fact is not present in the context, state that it is unavailable."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        return f"LLM error: {e}"


def ask_gpt5_with_retry(prompt: str) -> str:
    """
    Retry wrapper that prefers GPT-5 with a single fallback model.
    """
    for attempt in range(MAX_RETRIES):
        out = call_openai(prompt, OPENAI_MODEL_PRIMARY)
        if not out.startswith("LLM error:"):
            return out
        time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

    # Fallback model (one shot)
    return call_openai(prompt, OPENAI_MODEL_FALLBACK)

# ----------------------------- #
#      CONTEXT CONSTRUCTION     #
# ----------------------------- #

def build_lad_snapshot(df: pd.DataFrame, roi: pd.DataFrame, lad_key: str) -> str:
    row = df.loc[df["norm_lad"] == lad_key]
    if row.empty:
        return ""
    r = row.iloc[0]
    roi_row = roi.loc[roi["norm_lad"] == lad_key] if "norm_lad" in roi.columns else pd.DataFrame()
    roi_val = None if roi_row.empty or "ROI (%)" not in roi_row.columns else roi_row.iloc[0]["ROI (%)"]

    lines = [
        f"[{r['Local_Authority']}]",
        f"Investment_Score={safe_number(r.get('Investment_Potential_Score'), 2)}",
        f"Grade={str(r.get('Investment_Grade'))}",
        f"Pct_65plus={safe_number(r.get('Percent_65plus'), 2)}",
        f"GDHI_per_head_2022={safe_number(r.get('GDHI_per_head_2022'), 0)}",
        f"House_Price_Growth_pct={safe_number(r.get('House_Price_Growth_%'), 2)}",
        f"Care_Homes_Count={safe_number(r.get('Care_Homes_Count'), 0)}",
        f"Care_Homes_per_10k={safe_number(r.get('Care_Homes_per_10k'), 2)}",
        f"CQC_Good_pct={safe_number(r.get('%_CQC_Good'), 2)}",
        f"CQC_RI_pct={safe_number(r.get('%_CQC_RequiresImprovement'), 2)}",
    ]
    if roi_val is not None and pd.notna(roi_val):
        lines.append(f"ROI_pct={safe_number(roi_val, 2)}")
    lines.append(f"SHAP_available={'yes' if shap_available(lad_key) else 'no'}")
    return "\n".join(lines)


def build_context_from_query(
    df: pd.DataFrame,
    roi: pd.DataFrame,
    roi_blobs: Dict[str, str],
    gpt_blobs: Dict[str, str],
    user_query: str,
) -> Tuple[str, List[str]]:
    """
    Creates a compact, token‑friendly context from the query.
    Returns context and the list of matched LAD keys (norm form).
    """
    q = user_query.lower()
    lad_keys = df["norm_lad"].unique().tolist()

    matched: List[str] = [key for key in lad_keys if key in q]
    # If nothing matched, attempt a fuzzy contains over display names
    if not matched:
        for _, r in df.iterrows():
            disp = str(r["Local_Authority"]).lower()
            if disp in q:
                matched.append(r["norm_lad"])

    matched = list(dict.fromkeys(matched))[:6]  # cap to avoid token bloat

    context_parts: List[str] = []
    if len(matched) >= 2:
        context_parts.append("[Comparison]")
        for key in matched:
            context_parts.append(build_lad_snapshot(df, roi, key))

    for key in matched or []:
        context_parts.append(build_lad_snapshot(df, roi, key))
        if key in gpt_blobs:
            # Keep GPT summary short; truncate if extremely long
            summary = gpt_blobs[key].strip()
            if len(summary) > 1500:
                summary = summary[:1500].rsplit("\n", 1)[0] + "\n[...]"
            context_parts.append(f"[GPT_Summary:{key}]\n{summary}")
        if key in roi_blobs:
            roi_txt = roi_blobs[key].strip()
            if len(roi_txt) > 1200:
                roi_txt = roi_txt[:1200].rsplit("\n", 1)[0] + "\n[...]"
            context_parts.append(f"[ROI_Summary:{key}]\n{roi_txt}")

    # SHAP help if explicitly asked
    if any(kw in q for kw in ["shap", "explain visual", "feature importance"]):
        context_parts.append(
            "[SHAP_Primer]\n"
            "SHAP shows per‑feature contribution to the predicted investment score. "
            "Positive values push the score up; negative values push it down. "
            "When a SHAP image is unavailable for a LAD, state that clearly."
        )

    return "\n\n".join([p for p in context_parts if p]), matched


def build_prompt(context: str, user_query: str) -> str:
    return f"""Role: UK care home investment analyst.
Style: concise, decision‑oriented, no speculation, no invented numbers.

Ground rules:
- Use only the facts in [Context]. If data is missing, state that it is not available.
- Keep recommendations practical. Where risk exists (low scores, weak CQC, etc.), flag it.
- When comparing LADs, cite the relevant numeric fields from the context.
- If SHAP is requested, describe the implication of feature directionality and note availability.

[User_Query]
{user_query.strip()}

[Context]
{context if context else "No specific LAD context was located."}
"""


# ----------------------------- #
#              UI               #
# ----------------------------- #

st.title("🏡 Care Home Investment Dashboard (UK)")
st.caption("MSc Project – AI for Care Home Investment Support (2025)")

# Load data
df = load_base_data()
geojson_data = load_geojson()
roi_table = load_roi_table()
roi_blobs = load_text_blob(ROI_FOLDER)
gpt_blobs = load_text_blob(GPT_FOLDER)

# Sidebar filters
st.sidebar.header("Filters")
grade_filter = st.sidebar.multiselect(
    "Investment grade",
    options=sorted(df["Investment_Grade"].dropna().unique().tolist()),
    default=sorted(df["Investment_Grade"].dropna().unique().tolist()),
)
score_range = st.sidebar.slider("Score range", SCORE_MIN, SCORE_MAX, (SCORE_MIN, SCORE_MAX), step=1.0)

filtered = df[
    df["Investment_Grade"].isin(grade_filter)
    & df["Investment_Potential_Score"].between(score_range[0], score_range[1])
].copy()

# Tabs
tab_overview, tab_map, tab_assistant = st.tabs(["Overview", "Map & Details", "LLM Assistant"])

# ---- Overview Tab ----
with tab_overview:
    st.subheader("National snapshot")
    col1, col2, col3 = st.columns(3)
    col1.metric("LADs in dataset", len(df))
    col2.metric("Median score", f"{df['Investment_Potential_Score'].median():.1f}")
    col3.metric("Top grade share", f"{(df['Investment_Grade'].eq('Good').mean()*100):.1f}%")

    st.markdown("#### Top 15 LADs by score (current filter)")
    top = (
        filtered[[
            "Local_Authority", "Investment_Potential_Score", "Investment_Grade",
            "Percent_65plus", "%_CQC_Good", "House_Price_Growth_%"
        ]]
        .sort_values("Investment_Potential_Score", ascending=False)
        .head(15)
    )
    st.dataframe(top, use_container_width=True)

# ---- Map & Details Tab ----
with tab_map:
    st.subheader("Clickable LAD map")
    # Choropleth by Investment Score
    m = folium.Map(location=DEFAULT_MAP_CENTER, zoom_start=DEFAULT_MAP_ZOOM, tiles="cartodbpositron")

    # Prepare mapping of LAD name -> score for Folium
    score_map = {r["Local_Authority"]: float(r["Investment_Potential_Score"]) for _, r in df.iterrows()}
    # Join key for geojson uses LAD25NM per the provided file
    folium.Choropleth(
        geo_data=geojson_data,
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
        geojson_data,
        name="LADs",
        style_function=lambda x: {"fillOpacity": 0, "weight": 0.3, "color": "#333"},
        highlight_function=lambda x: {"weight": 2, "color": "#111"},
        tooltip=folium.GeoJsonTooltip(fields=["LAD25NM"], aliases=["LAD:"]),
    ).add_to(m)

    map_output = st_folium(m, height=540, returned_objects=["last_active_drawing"])

    st.markdown("---")

    # Selection
    lad_names = df["Local_Authority"].sort_values().unique().tolist()
    if "selected_lad" not in st.session_state:
        st.session_state.selected_lad = lad_names[0]

    # Click -> update selection
    if map_output and map_output.get("last_active_drawing"):
        clicked_lad = map_output["last_active_drawing"]["properties"]["LAD25NM"]
        if clicked_lad in lad_names:
            st.session_state.selected_lad = clicked_lad

    # Dropdown (always shown)
    selected_lad = st.selectbox("Select a Local Authority District:", lad_names, index=lad_names.index(st.session_state.selected_lad))

    # Display panel
    sel = df.loc[df["Local_Authority"] == selected_lad]
    if not sel.empty:
        r = sel.iloc[0]
        key = r["norm_lad"]

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Score", f"{safe_number(r.get('Investment_Potential_Score'), 2)} / 100")
        colB.metric("Grade", str(r.get("Investment_Grade")))
        colC.metric("% aged 65+", safe_number(r.get("Percent_65plus"), 2))
        colD.metric("House price growth %", safe_number(r.get("House_Price_Growth_%"), 2))

        st.markdown("##### Core indicators")
        core = pd.DataFrame(
            {
                "Indicator": [
                    "GDHI per head (2022)",
                    "Care homes (count)",
                    "Care homes per 10k",
                    "CQC Good %",
                    "CQC Requires Improvement %",
                ],
                "Value": [
                    f"£{safe_number(r.get('GDHI_per_head_2022'), 0)}",
                    safe_number(r.get("Care_Homes_Count"), 0),
                    safe_number(r.get("Care_Homes_per_10k"), 2),
                    safe_number(r.get("%_CQC_Good"), 2),
                    safe_number(r.get("%_CQC_RequiresImprovement"), 2),
                ],
            }
        )
        st.dataframe(core, hide_index=True, use_container_width=True)

        st.markdown("##### SHAP visual")
        shap_path = os.path.join(SHAP_FOLDER, f"{key}.png")
        if os.path.exists(shap_path):
            st.image(shap_path, use_column_width=True)
        else:
            st.info("No SHAP image is available for this LAD.")

        st.markdown("##### LLM summary")
        gpt_path = os.path.join(GPT_FOLDER, f"{key}.txt")
        if os.path.exists(gpt_path):
            with open(gpt_path, "r", encoding="utf-8") as fh:
                st.markdown(fh.read())
        else:
            st.info("No LLM summary was provided for this LAD.")

        st.markdown("##### ROI simulation summary")
        roi_path = os.path.join(ROI_FOLDER, f"{key}.txt")
        if os.path.exists(roi_path):
            with open(roi_path, "r", encoding="utf-8") as fh:
                st.markdown(fh.read())
        else:
            st.info("No ROI simulation summary was provided for this LAD.")

# ---- LLM Assistant Tab ----
with tab_assistant:
    st.subheader("🧠 Investment Assistant (GPT‑5)")
    st.write(
        "Example prompts: "
        "_What is the ROI in York?_, _Compare Camden and Southwark_, _Explain the SHAP visual for Leeds_."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Show history
    for role, content in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").markdown(content)
        else:
            st.chat_message("assistant").markdown(content)

    user_query = st.chat_input("Type a question about any LAD(s) …")
    if user_query:
        st.chat_message("user").markdown(user_query)
        st.session_state.chat_history.append(("user", user_query))

        with st.spinner("Generating grounded answer…"):
            context, matched = build_context_from_query(df, roi_table, roi_blobs, gpt_blobs, user_query)
            prompt = build_prompt(context, user_query)
            answer = ask_gpt5_with_retry(prompt)

        st.chat_message("assistant").markdown(answer)
        st.session_state.chat_history.append(("assistant", answer))


# ----------------------------- #
#            FOOTER             #
# ----------------------------- #

st.markdown("---")
st.caption("Created for the MSc Project (2025). Data includes demographic, financial, and quality-of-care features. SHAP images and text blobs are loaded when available.")

