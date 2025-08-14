# -*- coding: utf-8 -*-
"""
Care Home Investment Dashboard (GPT‑5, hardened)
- Robust OpenAI v1 client usage with graceful degradation
- ROI name normalisation from `District`
- Consistent LAD normalisation across data/files
- Fuzzy LAD matching for LLM context
- Choropleth + details + grounded assistant
"""

from __future__ import annotations
import os, json, time, re
from typing import Dict, List, Tuple
import pandas as pd
import streamlit as st
import folium
from difflib import get_close_matches
from streamlit_folium import st_folium

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Care Home Investment – UK", page_icon="🏡", layout="wide")

DATA_PATH       = "final_model_data_with_grade.csv"
GEOJSON_PATH    = "LAD_MAY_2025_Simplified_5.geojson"
SHAP_FOLDER     = "shap_visuals"
GPT_FOLDER      = "gpt_explanation"
ROI_FOLDER      = "roi_gpt"
ROI_TABLE_PATH  = "roi_by_district.csv"

OPENAI_API_KEY        = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
OPENAI_MODEL_PRIMARY  = "gpt-5"
OPENAI_MODEL_FALLBACK = "gpt-4o-mini"
TEMPERATURE           = 0.3
MAX_RETRIES           = 2
RETRY_BACKOFF_SECONDS = 2.0

DEFAULT_MAP_CENTER = [54.5, -3.0]
DEFAULT_MAP_ZOOM   = 5

# =========================
# UTILITIES
# =========================
def normalise(name: str) -> str:
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
    try:
        if pd.isna(val):
            return default
        return f"{float(val):.{digits}f}"
    except Exception:
        return default

def shap_available(key: str) -> bool:
    return os.path.exists(os.path.join(SHAP_FOLDER, f"{key}.png"))

@st.cache_data(show_spinner=False)
def load_base_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        st.error(f"Missing data file: {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    if "Local_Authority" not in df.columns:
        st.error("Column 'Local_Authority' is missing from the model data.")
        st.stop()
    df = df.copy()
    df["norm_lad"] = df["Local_Authority"].apply(normalise)
    for col in [
        "Investment_Potential_Score","Percent_65plus","GDHI_per_head_2022",
        "House_Price_Growth_%","Care_Homes_Count","Care_Homes_per_10k",
        "%_CQC_Good","%_CQC_RequiresImprovement"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_geojson() -> dict:
    if not os.path.exists(GEOJSON_PATH):
        st.error(f"Missing GeoJSON: {GEOJSON_PATH}")
        st.stop()
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_roi_table() -> pd.DataFrame:
    if not os.path.exists(ROI_TABLE_PATH):
        return pd.DataFrame(columns=["norm_lad","ROI (%)"])
    roi = pd.read_csv(ROI_TABLE_PATH)
    # Normalise from District name (as per your CSV)
    if "District" in roi.columns:
        roi["norm_lad"] = roi["District"].apply(normalise)
    elif "LAD" in roi.columns:
        roi["norm_lad"] = roi["LAD"].apply(normalise)
    else:
        # best-effort: try first string column
        name_col = next((c for c in roi.columns if roi[c].dtype == object), None)
        roi["norm_lad"] = roi[name_col].apply(normalise) if name_col else ""
    return roi

@st.cache_data(show_spinner=False)
def load_text_blob(folder: str) -> Dict[str, str]:
    data: Dict[str,str] = {}
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

INTENT_KEYWORDS = {
    "compare": ["compare", "vs", "versus", "better than", "which is better"],
    "shap": ["shap", "feature importance", "explain visual", "drivers"],
    "roi": ["roi", "return", "returns", "appreciation", "yield"],
    "why": ["why", "rationale", "reason"],
    "action": ["invest", "proceed", "avoid", "recommend", "should we"]
}

def infer_intent(user_query: str, matched_lads: list, context: str) -> dict:
    q = user_query.lower()
    flags = {
        "is_compare": len(matched_lads) >= 2 or any(k in q for k in INTENT_KEYWORDS["compare"]),
        "wants_shap": any(k in q for k in INTENT_KEYWORDS["shap"]) or "SHAP_Primer" in context,
        "wants_roi": any(k in q for k in INTENT_KEYWORDS["roi"]) or "ROI%" in context or "ROI_Summary" in context,
        "wants_rationale": any(k in q for k in INTENT_KEYWORDS["why"]),
        "wants_action": any(k in q for k in INTENT_KEYWORDS["action"]),
        "has_rich_context": len(context) > 800  # simple proxy
    }
    return flags


# =========================
# OPENAI (v1) – graceful fallback
# =========================
def call_openai(prompt: str, model: str) -> str:
    if not OPENAI_API_KEY:
        return "LLM error: no API key. Set OPENAI_API_KEY in Streamlit secrets."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Act as a UK care-home investment analyst. "
                        "Write concise, decision-focused answers. Use only supplied context. "
                        "If a fact is missing, say so."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {e}"

def ask_gpt_with_retry(prompt: str) -> str:
    for attempt in range(MAX_RETRIES):
        out = call_openai(prompt, OPENAI_MODEL_PRIMARY)
        if not out.startswith("LLM error:"):
            return out
        time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    return call_openai(prompt, OPENAI_MODEL_FALLBACK)

# =========================
# LLM CONTEXT
# =========================
def build_lad_snapshot(df: pd.DataFrame, roi: pd.DataFrame, lad_key: str) -> str:
    row = df.loc[df["norm_lad"] == lad_key]
    if row.empty:
        return ""
    r = row.iloc[0]
    roi_row = roi.loc[roi["norm_lad"] == lad_key] if "norm_lad" in roi.columns else pd.DataFrame()
    roi_val = None if roi_row.empty or "ROI (%)" not in roi_row.columns else roi_row.iloc[0]["ROI (%)"]

    lines = [
        f"[{r['Local_Authority']}]",
        f"Score={safe_number(r.get('Investment_Potential_Score'),2)}/100",
        f"Grade={str(r.get('Investment_Grade'))}",
        f"%65+={safe_number(r.get('Percent_65plus'),2)}",
        f"GDHI_2022=£{safe_number(r.get('GDHI_per_head_2022'),0)}",
        f"Price_Growth%={safe_number(r.get('House_Price_Growth_%'),2)}",
        f"CareHomes={safe_number(r.get('Care_Homes_Count'),0)}",
        f"Per10k={safe_number(r.get('Care_Homes_per_10k'),2)}",
        f"CQC_Good%={safe_number(r.get('%_CQC_Good'),2)}",
        f"CQC_RI%={safe_number(r.get('%_CQC_RequiresImprovement'),2)}",
        f"ROI%={safe_number(roi_val,2) if roi_val is not None else 'N/A'}",
        f"SHAP={'available' if shap_available(lad_key) else 'missing'}",
    ]
    return " | ".join(lines)

def extract_possible_lads(text: str) -> List[str]:
    # simple token pass + fuzzy against known LADs
    tokens = re.findall(r"[A-Za-z][A-Za-z\s\-\']{2,}", text)
    return [normalise(t) for t in tokens]

def build_context_from_query(
    df: pd.DataFrame,
    roi: pd.DataFrame,
    roi_blobs: Dict[str, str],
    gpt_blobs: Dict[str, str],
    user_query: str,
    max_lads: int = 6,
) -> Tuple[str, List[str]]:
    q = user_query.strip().lower()
    known = df["norm_lad"].unique().tolist()

    # direct contains over norm names
    matched = [k for k in known if k in q]

    # if none, fuzzy on tokens
    if not matched:
        candidates = extract_possible_lads(q)
        for c in candidates:
            close = get_close_matches(c, known, n=1, cutoff=0.93)
            if close:
                matched.append(close[0])

    # if still none, try display-name contains
    if not matched:
        for _, r in df.iterrows():
            if str(r["Local_Authority"]).lower() in q:
                matched.append(r["norm_lad"])

    # dedupe & cap
    seen = set(); ordered=[]
    for k in matched:
        if k not in seen:
            seen.add(k); ordered.append(k)
    matched = ordered[:max_lads]

    parts: List[str] = []
    if len(matched) >= 2:
        parts.append("[Comparison]")
        for key in matched:
            parts.append(build_lad_snapshot(df, roi, key))

    for key in matched:
        parts.append(build_lad_snapshot(df, roi, key))
        if key in gpt_blobs:
            s = gpt_blobs[key].strip()
            parts.append(f"[LLM_Summary:{key}]\n" + (s[:1500] + ("\n[...]" if len(s) > 1500 else "")))
        if key in roi_blobs:
            s = roi_blobs[key].strip()
            parts.append(f"[ROI_Summary:{key}]\n" + (s[:1200] + ("\n[...]" if len(s) > 1200 else "")))

    if any(k in q for k in ["shap", "explain visual", "feature importance"]):
        parts.append(
            "[SHAP_Primer]\n"
            "SHAP shows per‑feature contribution to the investment score. "
            "Positive values increase the score; negative values reduce it. "
            "Only describe visuals when available; otherwise state 'missing'."
        )

    return "\n\n".join(p for p in parts if p), matched

def build_prompt(context: str, user_query: str, matched: list | None = None) -> str:
    matched = matched or []
    intent = infer_intent(user_query, matched, context)

    # Core rules always on
    core_rules = [
        "Use only [Context]. Do not invent data.",
        "Write clearly and professionally, prioritising decision usefulness.",
    ]

    # Adaptive rules – included only when relevant
    adaptive_rules = []
    if intent["is_compare"]:
        adaptive_rules.append(
            "If multiple LADs are relevant, include a compact comparison table: LAD | Score/100 | Grade | ROI% (if available), then discuss the trade‑offs."
        )
    if intent["wants_shap"]:
        adaptive_rules.append(
            "If SHAP is available, summarise the top 2–4 drivers and their direction; if missing, state this plainly."
        )
    if intent["wants_roi"]:
        adaptive_rules.append(
            "Where ROI% is present in context, interpret it briefly (trend/level) and relate it to the investment score."
        )
    if intent["wants_rationale"]:
        adaptive_rules.append("Include a brief rationale that ties the numbers to the conclusion.")
    if intent["wants_action"]:
        adaptive_rules.append("Finish with one clear action: Proceed / Proceed with caution / Defer, and a one‑line reason.")
    # Allow the model to add structure only when it helps
    if intent["has_rich_context"]:
        adaptive_rules.append(
            "If an extra section would materially help (e.g., risks, upside catalysts), add it succinctly."
        )

    # Target depth: short when simple, deeper when needed
    depth_line = (
        "Keep to 3–6 sentences if the query is simple; expand to several short paragraphs only when comparison or explanation merits it."
    )

    rules_block = "\n- ".join(core_rules + adaptive_rules + [depth_line])

    return f"""Role: UK care‑home investment analyst.

Output guidance:
- {rules_block}

[User_Query]
{user_query.strip()}

[Context]
{context if context else "No specific LAD context was located."}
"""


# =========================
# UI
# =========================
st.title("🏡 Care Home Investment Dashboard (UK)")
st.caption("MSc Project – AI for Care Home Investment Support (2025)")

df          = load_base_data()
geojson     = load_geojson()
roi_table   = load_roi_table()
roi_blobs   = load_text_blob(ROI_FOLDER)
gpt_blobs   = load_text_blob(GPT_FOLDER)

# Sidebar
st.sidebar.header("Filters")
grades = sorted(df["Investment_Grade"].dropna().unique().tolist())
chosen_grades = st.sidebar.multiselect("Investment grade", options=grades, default=grades)
score_min, score_max = st.sidebar.slider("Score range", 0.0, 100.0, (0.0, 100.0), step=1.0)

filtered = df[
    df["Investment_Grade"].isin(chosen_grades)
    & df["Investment_Potential_Score"].between(score_min, score_max)
].copy()

tab_overview, tab_map, tab_assistant = st.tabs(["Overview", "Map & Details", "LLM Assistant"])

# Overview
with tab_overview:
    st.subheader("National snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("LADs in dataset", len(df))
    c2.metric("Median score", f"{df['Investment_Potential_Score'].median():.1f}")
    c3.metric("Top grade share", f"{(df['Investment_Grade'].eq('Good').mean()*100):.1f}%")

    st.markdown("#### Top 15 LADs by score (current filter)")
    top = (
        filtered[
            ["Local_Authority","Investment_Potential_Score","Investment_Grade",
             "Percent_65plus","%_CQC_Good","House_Price_Growth_%"]
        ]
        .sort_values("Investment_Potential_Score", ascending=False)
        .head(15)
    )
    st.dataframe(top, use_container_width=True)

# Map & details
with tab_map:
    st.subheader("Clickable LAD map")
    m = folium.Map(location=DEFAULT_MAP_CENTER, zoom_start=DEFAULT_MAP_ZOOM, tiles="cartodbpositron")
    score_map = {r["Local_Authority"]: float(r["Investment_Potential_Score"]) for _, r in df.iterrows()}

    folium.Choropleth(
        geo_data=geojson,
        data=pd.DataFrame({"LAD": list(score_map.keys()), "Score": list(score_map.values())}),
        columns=["LAD","Score"],
        key_on="feature.properties.LAD25NM",
        fill_color="YlGnBu", nan_fill_color="#efefef",
        fill_opacity=0.7, line_opacity=0.2,
        legend_name="Investment Potential Score",
    ).add_to(m)

    folium.GeoJson(
        geojson, name="LADs",
        style_function=lambda x: {"fillOpacity": 0, "weight": 0.3, "color": "#333"},
        highlight_function=lambda x: {"weight": 2, "color": "#111"},
        tooltip=folium.GeoJsonTooltip(fields=["LAD25NM"], aliases=["LAD:"]),
    ).add_to(m)

    map_output = st_folium(m, height=540, returned_objects=["last_active_drawing"])
    st.markdown("---")

    lad_names = df["Local_Authority"].sort_values().unique().tolist()
    if "selected_lad" not in st.session_state:
        st.session_state.selected_lad = lad_names[0]

    if map_output and map_output.get("last_active_drawing"):
        clicked_lad = map_output["last_active_drawing"]["properties"]["LAD25NM"]
        if clicked_lad in lad_names:
            st.session_state.selected_lad = clicked_lad

    selected_lad = st.selectbox("Select a Local Authority District:", lad_names, index=lad_names.index(st.session_state.selected_lad))

    sel = df.loc[df["Local_Authority"] == selected_lad]
    if not sel.empty:
        r   = sel.iloc[0]
        key = r["norm_lad"]

        cA,cB,cC,cD = st.columns(4)
        cA.metric("Score", f"{safe_number(r.get('Investment_Potential_Score'),2)} / 100")
        cB.metric("Grade", str(r.get("Investment_Grade")))
        cC.metric("% aged 65+", safe_number(r.get("Percent_65plus"),2))
        cD.metric("House price growth %", safe_number(r.get("House_Price_Growth_%"),2))

        st.markdown("##### Core indicators")
        core = pd.DataFrame(
            {
                "Indicator": [
                    "GDHI per head (2022)","Care homes (count)","Care homes per 10k",
                    "CQC Good %","CQC Requires Improvement %"
                ],
                "Value": [
                    f"£{safe_number(r.get('GDHI_per_head_2022'),0)}",
                    safe_number(r.get("Care_Homes_Count"),0),
                    safe_number(r.get("Care_Homes_per_10k"),2),
                    safe_number(r.get("%_CQC_Good"),2),
                    safe_number(r.get("%_CQC_RequiresImprovement"),2),
                ],
            }
        )
        st.dataframe(core, hide_index=True, use_container_width=True)

        st.markdown("##### SHAP visual")
        shap_path = os.path.join(SHAP_FOLDER, f"{key}.png")
        st.image(shap_path, use_column_width=True) if os.path.exists(shap_path) else st.info("No SHAP image is available for this LAD.")

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

# LLM Assistant
with tab_assistant:
    st.subheader("🧠 Investment Assistant (GPT‑5)")
    st.write("Examples: *What is the ROI in York?* · *Compare Camden and Southwark* · *Explain the SHAP visual for Leeds*.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, content in st.session_state.chat_history:
        st.chat_message(role).markdown(content)

    user_query = st.chat_input("Type a question about any LAD(s) …")
    if user_query:
        st.chat_message("user").markdown(user_query)
        st.session_state.chat_history.append(("user", user_query))
        with st.spinner("Generating grounded answer…"):
            context, matched = build_context_from_query(df, roi_table, roi_blobs, gpt_blobs, user_query)
            prompt = build_prompt(context, user_query)
            answer = ask_gpt_with_retry(prompt)
        st.chat_message("assistant").markdown(answer)
        st.session_state.chat_history.append(("assistant", answer))

# Footer
st.markdown("---")
st.caption("Created for the MSc Project (2025). Data includes demographic, financial, and quality-of-care features. SHAP images and text blobs are loaded when available.")
