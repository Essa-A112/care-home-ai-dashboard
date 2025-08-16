# -*- coding: utf-8 -*-
"""
Care Home Investment Dashboard (GPT-5, hardened)
- Robust OpenAI v1 client usage with graceful degradation
- ROI name normalisation from `District`
- Consistent LAD normalisation across data/files
- Fuzzy LAD matching for LLM context
- Choropleth + details + grounded assistant
- SHAP knowledge injection for LLM (global + per-LAD, enriched files)
- Zoning reports section (per-LAD MD + JSON meta), also available to LLM
"""

from __future__ import annotations
import os, json, time, re, io
from typing import Dict, List, Tuple, Optional
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

# already used in your app for SHAP images
SHAP_IMG_FOLDER = "shap_visuals"

# NEW: structured SHAP assets (global + per_LAD + guides)
SHAP_ROOT       = "SHAP"
SHAP_PER_LAD    = os.path.join(SHAP_ROOT, "per_LAD")
SHAP_TOP_DRIVERS_PATH         = os.path.join(SHAP_ROOT, "lad_shap_top_drivers.csv")
SHAP_GLOBAL_SUMMARY_PATH      = os.path.join(SHAP_ROOT, "global_summary.csv")
SHAP_GLOBAL_SUMMARY_ENR_PATH  = os.path.join(SHAP_ROOT, "global_summary_enriched.csv")
SHAP_FEATURE_GUIDE_TXT        = os.path.join(SHAP_ROOT, "feature_guide.txt")
SHAP_FEATURE_GUIDE_JSON       = os.path.join(SHAP_ROOT, "feature_guide.json")

GPT_FOLDER      = "gpt_explanation"
ROI_FOLDER      = "roi_gpt"
ROI_TABLE_PATH  = "roi_by_district.csv"

# NEW: Zoning reports
ZONING_FOLDER   = "zoning_planning_summary"  # <lad>_zoning_planning_summary.md + <lad>_zoning_planning_meta.json
ZONING_INDEX_CSV= os.path.join(ZONING_FOLDER, "zoning_planning_summaries_index.csv")

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

def shap_img_available(key: str) -> bool:
    return os.path.exists(os.path.join(SHAP_IMG_FOLDER, f"{key}.png"))

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
    """Load loose .txt files as {norm_name: text}."""
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

@st.cache_data(show_spinner=False)
def load_markdown_blobs(folder: str, suffix: str) -> Dict[str, str]:
    """
    Read per-LAD *.md files with a specific suffix (e.g. '_enriched', '_zoning_planning_summary').
    Returns { lad_key : md_text }.
    """
    out: Dict[str, str] = {}
    if not os.path.isdir(folder):
        return out
    for f in os.listdir(folder):
        if f.lower().endswith(".md") and f.lower().endswith(f"{suffix}.md"):
            key = normalise(f.replace(suffix + ".md", ""))  # strip the suffix
            p = os.path.join(folder, f)
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    out[key] = fh.read()
            except Exception:
                pass
    return out

@st.cache_data(show_spinner=False)
def load_json_meta(folder: str, suffix: str) -> Dict[str, dict]:
    """
    Read per-LAD *.json meta files with a specific suffix, returns { lad_key : dict }.
    """
    out: Dict[str, dict] = {}
    if not os.path.isdir(folder):
        return out
    for f in os.listdir(folder):
        if f.lower().endswith(".json") and f.lower().endswith(f"{suffix}.json"):
            key = normalise(f.replace(suffix + ".json", ""))
            p = os.path.join(folder, f)
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    out[key] = json.load(fh)
            except Exception:
                pass
    return out

# ---------- SHAP structured assets ----------
@st.cache_data(show_spinner=False)
def load_shap_top_table() -> pd.DataFrame:
    """Wide table you created: one row per LAD, Top1/2/3 drivers etc."""
    if not os.path.exists(SHAP_TOP_DRIVERS_PATH):
        return pd.DataFrame()
    df = pd.read_csv(SHAP_TOP_DRIVERS_PATH)
    # Ensure LAD key column normalised
    # Accept either 'LAD_key' or a first column that looks like LAD name
    key_col = "LAD_key" if "LAD_key" in df.columns else df.columns[0]
    df["lad_key"] = df[key_col].apply(normalise)
    return df

@st.cache_data(show_spinner=False)
def load_shap_global_guides() -> Dict[str, str]:
    """
    Load global SHAP summaries and feature guide.
    Returns a small dict of strings that can be inserted into LLM context.
    """
    out = {}

    def safe_read(path: str, label: str):
        if os.path.exists(path):
            try:
                text = pd.read_csv(path).to_csv(index=False)
                out[label] = text[:2500]  # cap for context
            except Exception:
                pass

    # CSV global summaries as text blocks (compact)
    safe_read(SHAP_GLOBAL_SUMMARY_PATH, "SHAP_Global_Summary")
    safe_read(SHAP_GLOBAL_SUMMARY_ENR_PATH, "SHAP_Global_Summary_Enriched")

    # Feature guide (prefer TXT, fall back to JSON)
    if os.path.exists(SHAP_FEATURE_GUIDE_TXT):
        with open(SHAP_FEATURE_GUIDE_TXT, "r", encoding="utf-8") as fh:
            out["Feature_Guide"] = fh.read()[:3000]
    elif os.path.exists(SHAP_FEATURE_GUIDE_JSON):
        try:
            j = json.load(open(SHAP_FEATURE_GUIDE_JSON, "r", encoding="utf-8"))
            out["Feature_Guide"] = json.dumps(j, indent=2)[:3000]
        except Exception:
            pass

    return out

@st.cache_data(show_spinner=False)
def load_shap_perlad_blobs() -> Tuple[Dict[str, str], Dict[str, dict]]:
    """
    Read SHAP/per_LAD/<lad>_enriched.md and <lad>_enriched.json (optional).
    """
    md = load_markdown_blobs(SHAP_PER_LAD, "_enriched")
    meta = load_json_meta(SHAP_PER_LAD, "_enriched")
    return md, meta

def extract_top_drivers_for_lad(lad_key: str, top_df: pd.DataFrame) -> Optional[str]:
    if top_df.empty:
        return None
    row = top_df.loc[top_df["lad_key"] == lad_key]
    if row.empty:
        return None
    r = row.iloc[0]
    parts = []
    for i in (1, 2, 3):
        feat = r.get(f"Top{i}_Feature")
        if pd.isna(feat) or feat == "":
            continue
        direction = r.get(f"Top{i}_Dir", "")
        absval = r.get(f"Top{i}_AbsSHAP", "")
        parts.append(f"{i}) {feat} ({direction}, |SHAP|={safe_number(absval, 3)})")
    return "; ".join(parts) if parts else None

# ---------- Zoning assets ----------
@st.cache_data(show_spinner=False)
def load_zoning_assets() -> Tuple[Dict[str, str], Dict[str, dict], pd.DataFrame]:
    """
    Returns:
      zoning_md:   { lad_key : markdown }
      zoning_meta: { lad_key : dict }
      zoning_index (optional CSV if present)
    """
    md   = load_markdown_blobs(ZONING_FOLDER, "_zoning_planning_summary")
    meta = load_json_meta(ZONING_FOLDER, "_zoning_planning_meta")
    idx  = pd.read_csv(ZONING_INDEX_CSV) if os.path.exists(ZONING_INDEX_CSV) else pd.DataFrame()
    return md, meta, idx

# =========================
# INTENT
# =========================
INTENT_KEYWORDS = {
    "compare": ["compare", "vs", "versus", "better than", "which is better"],
    "shap": ["shap", "feature importance", "explain visual", "drivers"],
    "roi": ["roi", "return", "returns", "appreciation", "yield"],
    "why": ["why", "rationale", "reason"],
    "action": ["invest", "proceed", "avoid", "recommend", "should we"],
    "zoning": ["zoning", "planning", "permission", "class use", "policy", "local plan"],
}

def infer_intent(user_query: str, matched_lads: list, context: str) -> dict:
    q = user_query.lower()
    flags = {
        "is_compare": len(matched_lads) >= 2 or any(k in q for k in INTENT_KEYWORDS["compare"]),
        "wants_shap": any(k in q for k in INTENT_KEYWORDS["shap"]) or "SHAP_Primer" in context,
        "wants_roi": any(k in q for k in INTENT_KEYWORDS["roi"]) or "ROI%" in context or "ROI_Summary" in context,
        "wants_rationale": any(k in q for k in INTENT_KEYWORDS["why"]),
        "wants_action": any(k in q for k in INTENT_KEYWORDS["action"]),
        "wants_zoning": any(k in q for k in INTENT_KEYWORDS["zoning"]) or "[Zoning_Summary:" in context,
        "has_rich_context": len(context) > 800,
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
        f"SHAP_IMG={'available' if shap_img_available(lad_key) else 'missing'}",
    ]
    return " | ".join(lines)

def extract_possible_lads(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z\s\-\']{2,}", text)
    return [normalise(t) for t in tokens]

def build_context_from_query(
    df: pd.DataFrame,
    roi: pd.DataFrame,
    roi_blobs: Dict[str, str],
    gpt_blobs: Dict[str, str],
    user_query: str,
    max_lads: int = 6,
    # NEW SHAP/zoning inputs:
    shap_top: Optional[pd.DataFrame] = None,
    shap_global_guides: Optional[Dict[str, str]] = None,
    shap_md: Optional[Dict[str, str]] = None,
    zoning_md: Optional[Dict[str, str]] = None,
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

        # --- LLM summaries you already had
        if key in gpt_blobs:
            s = gpt_blobs[key].strip()
            parts.append(f"[LLM_Summary:{key}]\n" + (s[:1500] + ("\n[...]" if len(s) > 1500 else "")))
        if key in roi_blobs:
            s = roi_blobs[key].strip()
            parts.append(f"[ROI_Summary:{key}]\n" + (s[:1200] + ("\n[...]" if len(s) > 1200 else "")))

        # --- NEW: SHAP top drivers from lad_shap_top_drivers.csv
        if shap_top is not None and not shap_top.empty:
            drivers = extract_top_drivers_for_lad(key, shap_top)
            if drivers:
                parts.append(f"[SHAP_Drivers:{key}]\nTop drivers: {drivers}")

        # --- NEW: SHAP per-LAD enriched notes (markdown)
        if shap_md and key in shap_md:
            s = shap_md[key].strip()
            parts.append(f"[SHAP_Enriched:{key}]\n" + (s[:1200] + ("\n[...]" if len(s) > 1200 else "")))

        # --- NEW: Zoning summaries (only if available)
        if zoning_md and key in zoning_md:
            s = zoning_md[key].strip()
            parts.append(f"[Zoning_Summary:{key}]\n" + (s[:1200] + ("\n[...]" if len(s) > 1200 else "")))

    # If user hints at SHAP, add a concise primer & a compact global context
    if any(k in q for k in ["shap", "explain visual", "feature importance", "drivers"]):
        parts.append(
            "[SHAP_Primer]\n"
            "SHAP shows per-feature contribution to the investment score. "
            "Positive values increase the score; negative values reduce it. "
            "Summaries below reflect mean |SHAP| (impact magnitude) and signed effects."
        )
        if shap_global_guides:
            if "SHAP_Global_Summary" in shap_global_guides:
                parts.append("[Global_SHAP]\n" + shap_global_guides["SHAP_Global_Summary"])
            if "Feature_Guide" in shap_global_guides:
                parts.append("[Feature_Guide]\n" + shap_global_guides["Feature_Guide"])

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
            "If multiple LADs are relevant, include a compact comparison table: LAD | Score/100 | Grade | ROI% (if available), then discuss the trade-offs."
        )
    if intent["wants_shap"]:
        adaptive_rules.append(
            "If SHAP is available, summarise the top 2–4 drivers and their direction; if missing, state this plainly."
        )
    if intent["wants_roi"]:
        adaptive_rules.append(
            "Where ROI% is present in context, interpret it briefly (trend/level) and relate it to the investment score."
        )
    if intent["wants_zoning"]:
        adaptive_rules.append(
            "Where zoning summaries are present, highlight any constraints/opportunities that materially alter investment attractiveness."
        )
    if intent["wants_rationale"]:
        adaptive_rules.append("Include a brief rationale that ties the numbers to the conclusion.")
    if intent["wants_action"]:
        adaptive_rules.append("Finish with one clear action: Proceed / Proceed with caution / Defer, and a one-line reason.")
    if intent["has_rich_context"]:
        adaptive_rules.append("If an extra section (risks/upside catalysts) helps, add it succinctly.")

    depth_line = (
        "Keep to 3–6 sentences if the query is simple; expand to several short paragraphs only when comparison or explanation merits it."
    )
    rules_block = "\n- ".join(core_rules + adaptive_rules + [depth_line])

    return f"""Role: UK care-home investment analyst.

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

# NEW: SHAP structured assets & Zoning
shap_top_df          = load_shap_top_table()
shap_global_guides   = load_shap_global_guides()
shap_md, shap_meta   = load_shap_perlad_blobs()

zoning_md, zoning_meta, zoning_idx = load_zoning_assets()

# Sidebar
st.sidebar.header("Filters")
grades = sorted(df["Investment_Grade"].dropna().unique().tolist())
chosen_grades = st.sidebar.multiselect("Investment grade", options=grades, default=grades)
score_min, score_max = st.sidebar.slider("Score range", 0.0, 100.0, (0.0, 100.0), step=1.0)

filtered = df[
    df["Investment_Grade"].isin(chosen_grades)
    & df["Investment_Potential_Score"].between(score_min, score_max)
].copy()

tab_overview, tab_map, tab_zoning, tab_assistant = st.tabs(
    ["Overview", "Map & Details", "Zoning reports", "LLM Assistant"]
)

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
        shap_path = os.path.join(SHAP_IMG_FOLDER, f"{key}.png")
        st.image(shap_path, width=800) 
        if os.path.exists(shap_path) 
        else st.info("No SHAP image is available for this LAD.")

        # show compact top drivers if available (from lad_shap_top_drivers.csv)
        if not shap_top_df.empty:
            drivers = extract_top_drivers_for_lad(key, shap_top_df)
            if drivers:
                st.caption(f"Top SHAP drivers: {drivers}")

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

# NEW: Zoning reports section
with tab_zoning:
    st.subheader("Zoning & planning summaries (selected LADs)")

    # Build a robust list of LADs for which we actually have zoning content
    zoning_index = df[norm_lad].isin(zoning_md.keys())
    
    available_lads = (
        df.loc[zoning_index, "Local_Authority"]
          .dropna()
          .astype(str)
          .drop_duplicates()
          .sort_values()
          .tolist()
    )
    if not available_lads:
        st.info("No zoning reports available for the current filter.")
        st.stop()  # or `return` if you're inside a function


    else:
        zlad = st.selectbox("LAD with zoning summary:", available_lads, index=0)
        key  = normalise(zlad)

        # left: markdown summary; right: meta table
        c1, c2 = st.columns((2,1))
        with c1:
            st.markdown("#### Summary")
            st.markdown(zoning_md.get(key, "_No summary text found._"))

        with c2:
            meta = zoning_meta.get(key, {})
            st.markdown("#### Meta")
            if meta:
                # flatten (one level) then render
                flat = {}
                for k, v in meta.items():
                    if isinstance(v, (dict, list)):
                        flat[k] = json.dumps(v, ensure_ascii=False)
                    else:
                        flat[k] = v
                meta_df = pd.DataFrame(list(flat.items()), columns=["Field", "Value"])
                st.dataframe(meta_df, hide_index=True, use_container_width=True)
            else:
                st.info("No meta JSON found for this LAD.")

        # optional index CSV (e.g., to show which are covered)
        if not zoning_idx.empty:
            st.markdown("#### Coverage index")
            st.dataframe(zoning_idx, use_container_width=True)

# LLM Assistant
with tab_assistant:
    st.subheader("🧠 Investment Assistant (GPT-5)")
    st.write("Examples: *What is the ROI in York?* · *Compare Camden and Southwark* · *Explain the SHAP visual for Leeds* · *Any zoning constraints in Gravesham?*")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, content in st.session_state.chat_history:
        st.chat_message(role).markdown(content)

    user_query = st.chat_input("Type a question about any LAD(s) …")
    if user_query:
        st.chat_message("user").markdown(user_query)
        st.session_state.chat_history.append(("user", user_query))
        with st.spinner("Generating grounded answer…"):
            context, matched = build_context_from_query(
                df=df,
                roi=roi_table,
                roi_blobs=roi_blobs,
                gpt_blobs=gpt_blobs,
                user_query=user_query,
                shap_top=shap_top_df,
                shap_global_guides=shap_global_guides,
                shap_md=shap_md,
                zoning_md=zoning_md,
            )
            prompt = build_prompt(context, user_query, matched)
            answer = ask_gpt_with_retry(prompt)
        st.chat_message("assistant").markdown(answer)
        st.session_state.chat_history.append(("assistant", answer))

# Footer
st.markdown("---")
st.caption(
    "Created for the MSc Project (2025). Data includes demographic, financial, and quality-of-care features. "
    "SHAP images and structured SHAP assets are loaded when available. Zoning summaries render for LADs with uploaded reports."
)
