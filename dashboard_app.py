# -*- coding: utf-8 -*-
"""
Care Home Investment Dashboard (GPT-5, hardened)
- Robust OpenAI v1 client usage with graceful degradation
- ROI name normalisation from `District`
- Consistent LAD normalisation across data/files
- Fuzzy LAD matching for LLM context
- Choropleth + details + grounded assistant
- NEW: SHAP explanations wired into the chatbot context (per-LAD + feature guide)
- NEW: 'Zoning Reports' tab for the LADs you have zoning summaries for (top-5 by score)
"""

from __future__ import annotations
import os, json, time, re
from typing import Dict, List, Tuple, Iterable, Optional
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
SHAP_FOLDER     = "shap_visuals"               # existing PNGs for Map tab
GPT_FOLDER      = "gpt_explanation"            # existing per-LAD LLM summaries (.txt)
ROI_FOLDER      = "roi_gpt"                    # existing ROI summaries (.txt)
ROI_TABLE_PATH  = "roi_by_district.csv"

# NEW paths (non-breaking; all optional and fail-safe)
PER_LAD_SHAP_DIR    = "SHAP/per_LAD"           # per-LAD JSON/CSV/MD (e.g., adur_enriched.*)
FEATURE_GUIDE_JSON  = "feature_guide.json"     # feature dictionary (aliases, definitions, ranges)
FEATURE_GUIDE_TXT   = "feature_guide.txt"      # fallback readable text
GLOBAL_LAD_SHAP_TOP = "lad_shap_top_drivers.csv"  # optional fallback per-LAD top features
ZONING_DIR_CANDIDATES = [
    "zoning_planning_summary",
    "zoning_reports",
    "Outputs/gpt_summaries",                   # if copied in-repo
]

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

def best_match_norm(key_like: str, known: Iterable[str], cutoff: float = 0.84) -> Optional[str]:
    """Find closest LAD key; also try dropping common suffixes like '_enriched'."""
    k = normalise(key_like)
    if k in known:
        return k
    # strip common suffixes
    for suf in ["_enriched", "_lad", "_district"]:
        if k.endswith(suf):
            base = k[: -len(suf)]
            if base in known:
                return base
            close = get_close_matches(base, list(known), n=1, cutoff=cutoff)
            if close:
                return close[0]
    close = get_close_matches(k, list(known), n=1, cutoff=cutoff)
    return close[0] if close else None
    
def clean_stem(stem: str) -> str:
    s = normalise(stem)
    SUFFIXES = [
        "_zoning_planning_summary", "_zoning_summary", "_planning_summary",
        "_zoning_report", "_planning_report", "_zoning", "_planning",
        "_summary", "_report", "_reports", "_llm", "_gpt", "_notes",
        "_enriched"  # last so "adur_enriched" -> "adur"
    ]
    changed = True
    while changed:
        changed = False
        for suf in SUFFIXES:
            if s.endswith(suf):
                s = s[: -len(suf)]
                changed = True
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def strip_enriched(s: str) -> str:
    s = normalise(s)
    return s[:-10] if s.endswith("_enriched") else s


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

# ---------- NEW: SHAP + feature-guide + zoning loaders (optional & safe) ----------
@st.cache_data(show_spinner=False)
def load_feature_guide() -> Dict[str, dict]:
    """
    Returns alias->entry mapping. Each entry keeps fields like:
    dataset_column, definition, expected_direction, why_it_matters, unit
    """
    guide: Dict[str, dict] = {}
    if os.path.exists(FEATURE_GUIDE_JSON):
        try:
            with open(FEATURE_GUIDE_JSON, "r", encoding="utf-8") as fh:
                items = json.load(fh)
            for it in items:
                # map canonical + aliases
                names = [it.get("dataset_column","")]
                names += it.get("aliases", []) or []
                for nm in names:
                    if nm:
                        guide[normalise(nm)] = it
        except Exception:
            pass
    # TXT is kept only for display; not used for alias mapping to keep it robust
    return guide

@st.cache_data(show_spinner=False)
def load_per_lad_shap_drivers() -> Dict[str, List[dict]]:
    """
    Load per-LAD SHAP drivers from JSON/CSV under PER_LAD_SHAP_DIR.
    Keys are stored under BOTH the raw stem (e.g. 'halton_enriched') and the
    cleaned stem (e.g. 'halton') so lookups by LAD work.
    CSVs can have various column names; we normalise them.
    """
    out: Dict[str, List[dict]] = {}
    if not os.path.isdir(PER_LAD_SHAP_DIR):
        return out

    def _normalise_csv(df: pd.DataFrame) -> List[dict]:
        # unify column names we’ve seen in your repo variants
        ren = {}
        for c in df.columns:
            lc = c.strip().lower()
            if lc in ("feature", "term", "name"):
                ren[c] = "term"
            elif lc in ("value", "signed", "shap_value", "shap", "contribution", "impact"):
                ren[c] = "signed"
            elif lc in ("dir", "direction"):
                ren[c] = "direction"
            elif lc in ("strength", "importance", "abs", "magnitude"):
                ren[c] = "strength"
        if ren:
            df = df.rename(columns=ren)

        # if we still don’t have 'signed' but do have a numeric candidate, pick the first
        if "signed" not in df.columns:
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    df = df.rename(columns={c: "signed"})
                    break

        # fill direction/strength if missing
        if "direction" not in df.columns and "signed" in df.columns:
            df["direction"] = df["signed"].apply(lambda v: "up" if pd.notna(v) and float(v) >= 0 else "down")
        if "strength" not in df.columns and "signed" in df.columns:
            # simple magnitude string
            df["strength"] = df["signed"].abs().apply(lambda v: f"{float(v):.3g}" if pd.notna(v) else "n/a")

        # keep only rows that have a feature/term
        keep_cols = [c for c in ["term", "signed", "direction", "strength"] if c in df.columns]
        df = df.dropna(subset=[c for c in ["term"] if c in df.columns])
        return df[keep_cols].to_dict("records")

    for f in os.listdir(PER_LAD_SHAP_DIR):
        path = os.path.join(PER_LAD_SHAP_DIR, f)
        base, ext = os.path.splitext(f)
        key_raw   = normalise(base)           # e.g. 'halton_enriched'
        key_clean = strip_enriched(key_raw)   # e.g. 'halton'

        try:
            drivers: List[dict] = []
            if ext.lower() == ".json":
                with open(path, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                # flexible JSON shape
                if isinstance(obj, dict) and "drivers" in obj:
                    drivers = obj["drivers"]
                elif isinstance(obj, list):
                    drivers = obj
            elif ext.lower() == ".csv":
                df_csv = pd.read_csv(path)
                drivers = _normalise_csv(df_csv)
            else:
                continue

            if drivers:
                out[key_raw] = drivers              # 'halton_enriched'
                out.setdefault(key_clean, drivers)  # also store under 'halton'
        except Exception:
            # keep going if a single file fails
            pass

    return out


@st.cache_data(show_spinner=False)
def load_per_lad_shap_notes() -> Dict[str, str]:
    """
    Optional readable markdown notes per LAD in PER_LAD_SHAP_DIR (*.md).
    """
    out: Dict[str,str] = {}
    if not os.path.isdir(PER_LAD_SHAP_DIR):
        return out
    for f in os.listdir(PER_LAD_SHAP_DIR):
        if f.lower().endswith(".md"):
            base = os.path.splitext(f)[0]
            try:
                with open(os.path.join(PER_LAD_SHAP_DIR, f), "r", encoding="utf-8") as fh:
                    out[normalise(base)] = fh.read()
            except Exception:
                pass
    return out

@st.cache_data(show_spinner=False)
def load_global_lad_top_drivers() -> Dict[str, List[str]]:
    """
    Optional fallback CSV mapping LAD -> top feature names (strings).
    """
    out: Dict[str, List[str]] = {}
    if not os.path.exists(GLOBAL_LAD_SHAP_TOP):
        return out
    try:
        df = pd.read_csv(GLOBAL_LAD_SHAP_TOP)
        # heuristics: first column = lad name; next few columns are features
        if df.shape[1] >= 2:
            name_col = df.columns[0]
            for _, r in df.iterrows():
                key = normalise(r[name_col])
                feats = [str(v) for v in r.tolist()[1:] if isinstance(v, str) and v.strip()]
                if feats:
                    out[key] = feats
    except Exception:
        pass
    return out

@st.cache_data(show_spinner=False)
def load_zoning_blobs() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for folder in ZONING_DIR_CANDIDATES:
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if not f.lower().endswith((".md", ".txt")):
                continue  # only render text/markdown
            stem = os.path.splitext(f)[0]
            path = os.path.join(folder, f)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    text = fh.read()
            except Exception:
                continue
            key_orig  = normalise(stem)
            key_clean = clean_stem(stem)
            out[key_orig]  = text
            out[key_clean] = text
    return out


# ---------- INTENT ----------
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
        "wants_shap": any(k in q for k in INTENT_KEYWORDS["shap"]) or "SHAP_Primer" in context or "SHAP_Drivers" in context,
        "wants_roi": any(k in q for k in INTENT_KEYWORDS["roi"]) or "ROI%" in context or "ROI_Summary" in context,
        "wants_rationale": any(k in q for k in INTENT_KEYWORDS["why"]),
        "wants_action": any(k in q for k in INTENT_KEYWORDS["action"]),
        "has_rich_context": len(context) > 800
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
# LLM CONTEXT HELPERS
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
    tokens = re.findall(r"[A-Za-z][A-Za-z\s\-\']{2,}", text)
    return [normalise(t) for t in tokens]

# Globals filled later (safe to reference inside functions)
_feature_guide: Dict[str, dict] = {}
_shap_perlad: Dict[str, List[dict]] = {}
_shap_notes: Dict[str, str] = {}
_global_top_drivers: Dict[str, List[str]] = {}
_zoning_blobs: Dict[str, str] = {}

def _explain_driver(term: str) -> str:
    """Return 'name — quick why/how' using feature guide if known."""
    tkey = normalise(term)
    info = _feature_guide.get(tkey)
    if not info:
        # also try mapping via canonical name if aliases are capitalised with symbols
        # e.g., transform "House=Price Growth %" -> "house_price_growth_%"
        tkey2 = normalise(term.replace("=", " ").replace("%", "percent"))
        info = _feature_guide.get(tkey2)
    if info:
        exp_dir = info.get("expected_direction", "")
        why = info.get("why_it_matters", "")
        unit = info.get("unit", "")
        canon = info.get("dataset_column", term)
        bits = [canon]
        if unit: bits.append(f"[{unit}]")
        if exp_dir: bits.append(f"({exp_dir.replace('_',' ')})")
        line_head = " ".join(bits)
        if why:
            return f"{line_head}: {why}"
        return line_head
    return term

def _format_shap_drivers_for_context(lad_key: str, max_items: int = 6) -> Optional[str]:
    """
    Build a SHAP driver snippet for the LLM context.
    Try multiple keys: exact, + '_enriched', and stripped version.
    """
    candidates = [
        lad_key,
        f"{lad_key}_enriched",
        strip_enriched(lad_key),
    ]
    drivers: Optional[List[dict]] = None
    for c in candidates:
        if c in _shap_perlad and _shap_perlad[c]:
            drivers = _shap_perlad[c]
            break

    lines: List[str] = []
    if drivers:
        for d in drivers[:max_items]:
            term = (d.get("term") or d.get("feature") or "").strip()
            if not term:
                continue
            # compute direction/strength if not provided
            signed = d.get("signed")
            direction = d.get("direction")
            strength  = d.get("strength")
            if direction is None and signed is not None:
                try:
                    direction = "up" if float(signed) >= 0 else "down"
                except Exception:
                    direction = "n/a"
            if strength is None and signed is not None:
                try:
                    strength = f"{abs(float(signed)):.3g}"
                except Exception:
                    strength = "n/a"

            desc = _explain_driver(term)
            parts = [desc]
            if direction: parts.append(f"→ {direction}")
            if strength:  parts.append(f"({strength})")
            lines.append("- " + " ".join(parts))
    else:
        # fallback: global list of features (no signs here)
        feats = _global_top_drivers.get(lad_key) or _global_top_drivers.get(f"{lad_key}_enriched") or _global_top_drivers.get(strip_enriched(lad_key)) or []
        for f in feats[:max_items]:
            desc = _explain_driver(f)
            lines.append(f"- {desc} (top driver)")

    if not lines:
        return None
    return "Top SHAP drivers:\n" + "\n".join(lines)


def _attach_zoning_if_available(lad_key: str, df: pd.DataFrame) -> Optional[str]:
    # try exact, then fuzzy against file stems held in _zoning_blobs
    if lad_key in _zoning_blobs:
        return _zoning_blobs[lad_key][:1600] + ("\n[...]" if len(_zoning_blobs[lad_key]) > 1600 else "")
    # try by display name variants
    lad_name = df.loc[df["norm_lad"] == lad_key, "Local_Authority"].iloc[0] if not df.loc[df["norm_lad"] == lad_key].empty else ""
    guess = best_match_norm(lad_name, _zoning_blobs.keys(), cutoff=0.8)
    if guess and guess in _zoning_blobs:
        txt = _zoning_blobs[guess]
        return txt[:1600] + ("\n[...]" if len(txt) > 1600 else "")
    return None

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
        # NEW: SHAP drivers (per-LAD or fallback) + concise feature explanations
        shap_txt = _format_shap_drivers_for_context(key)
        if shap_txt:
            parts.append(f"[SHAP_Drivers:{key}]\n{shap_txt}")
        # NEW: attach zoning report excerpt if available
        ztxt = _attach_zoning_if_available(key, df)
        if ztxt:
            parts.append(f"[Zoning_Report:{key}]\n{ztxt}")

    # keep SHAP primer when relevant to the query
    if any(k in q for k in ["shap", "explain visual", "feature importance"]):
        parts.append(
            "[SHAP_Primer]\n"
            "SHAP shows per-feature contribution to the investment score. "
            "Positive values increase the score; negative values reduce it."
        )

    return "\n\n".join(p for p in parts if p), matched

def build_prompt(context: str, user_query: str, matched: list | None = None) -> str:
    matched = matched or []
    intent = infer_intent(user_query, matched, context)

    core_rules = [
        "Use only [Context]. Do not invent data.",
        "Write clearly and professionally, prioritising decision usefulness.",
    ]

    adaptive_rules = []
    if intent["is_compare"]:
        adaptive_rules.append(
            "If multiple LADs are relevant, include a compact comparison table: LAD | Score/100 | Grade | ROI% (if available), then discuss the trade-offs."
        )
    if intent["wants_shap"]:
        adaptive_rules.append(
            "If SHAP is available, summarise the top 2–4 drivers and their direction; where present, use the brief feature explanations in [SHAP_Drivers]."
        )
    if intent["wants_roi"]:
        adaptive_rules.append(
            "Where ROI% is present in context, interpret it briefly (trend/level) and relate it to the investment score."
        )
    if intent["wants_rationale"]:
        adaptive_rules.append("Include a brief rationale that ties the numbers to the conclusion.")
    if intent["wants_action"]:
        adaptive_rules.append("Finish with one clear action: Proceed / Proceed with caution / Defer, and a one-line reason.")
    if intent["has_rich_context"]:
        adaptive_rules.append("Add short sections only if they sharpen the decision (e.g., risks, catalysts).")

    depth_line = "Keep to 3–6 sentences if the query is simple; expand only when comparison/explanation merits it."
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

df           = load_base_data()
geojson      = load_geojson()
roi_table    = load_roi_table()
roi_blobs    = load_text_blob(ROI_FOLDER)
gpt_blobs    = load_text_blob(GPT_FOLDER)

# NEW: optional data (safe if missing)
_feature_guide        = load_feature_guide()
_shap_perlad          = load_per_lad_shap_drivers()
_shap_notes           = load_per_lad_shap_notes()
_global_top_drivers   = load_global_lad_top_drivers()
_zoning_blobs         = load_zoning_blobs()

# Sidebar
st.sidebar.header("Filters")
grades = sorted(df["Investment_Grade"].dropna().unique().tolist())
chosen_grades = st.sidebar.multiselect("Investment grade", options=grades, default=grades)
score_min, score_max = st.sidebar.slider("Score range", 0.0, 100.0, (0.0, 100.0), step=1.0)

filtered = df[
    df["Investment_Grade"].isin(chosen_grades)
    & df["Investment_Potential_Score"].between(score_min, score_max)
].copy()

# ---------- Tabs (existing + NEW 'Zoning Reports') ----------
tab_overview, tab_map, tab_assistant, tab_zoning = st.tabs(
    ["Overview", "Map & Details", "LLM Assistant", "Zoning Reports"]
)

# Overview (unchanged)
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

# Map & details (unchanged)
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
        if os.path.exists(shap_path):
            st.image(shap_path, width=800)
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

# LLM Assistant (extended only by context enrichment; UI unchanged)
with tab_assistant:
    st.subheader("🧠 Investment Assistant (GPT-5)")
    st.write("Examples: *What is the ROI in York?* · *Compare Camden and Southwark* · *Explain the SHAP for Leeds* · *Any zoning constraints in Sheffield?*")

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
            prompt = build_prompt(context, user_query, matched)
            answer = ask_gpt_with_retry(prompt)
        st.chat_message("assistant").markdown(answer)
        st.session_state.chat_history.append(("assistant", answer))

with tab_zoning:
    st.subheader("📑 Zoning & Planning Reports")
    if not _zoning_blobs:
        st.info("No zoning reports found in the repository paths.")
    else:
        known = df["norm_lad"].unique().tolist()
        mapped: List[Tuple[str, str, float, str]] = []  # (lad_key, display_name, score, grade)

        for zkey in _zoning_blobs.keys():
            candidates = [
                zkey,
                clean_stem(zkey),
                zkey.replace("_", " "),
                clean_stem(zkey).replace("_", " "),
            ]
            lad_key = None
            for cand in candidates:
                if cand in known:
                    lad_key = cand; break
                if cand.endswith("_enriched") and cand[:-10] in known:
                    lad_key = cand[:-10]; break
                if (cand + "_enriched") in known:
                    lad_key = cand + "_enriched"; break
                close = get_close_matches(cand, known, n=1, cutoff=0.75)
                if close:
                    lad_key = close[0]; break

            if lad_key:
                row = df.loc[df["norm_lad"] == lad_key]
                if not row.empty:
                    r = row.iloc[0]
                    mapped.append(
                        (lad_key, r["Local_Authority"], float(r["Investment_Potential_Score"]), str(r["Investment_Grade"]))
                    )

        # --- de-duplicate by LAD key (keep highest score) ---
        unique: Dict[str, Tuple[str, float, str]] = {}
        for lad_key, name, score, grade in mapped:
            if lad_key not in unique or score > unique[lad_key][1]:
                unique[lad_key] = (name, score, grade)

        mapped_sorted = sorted([(k, v[0], v[1], v[2]) for k, v in unique.items()], key=lambda x: x[2], reverse=True)

        if not mapped_sorted:
            st.info("Zoning files were found but could not be matched to LADs in the dataset.")
            st.stop()

        top5 = mapped_sorted[:5]
        st.markdown("These are the LADs with available zoning summaries (top 5 by score shown):")
        st.dataframe(
            pd.DataFrame([{"LAD": name, "Score": score, "Grade": grade} for _, name, score, grade in top5]),
            use_container_width=True, hide_index=True
        )

        # Selection among unique LADs only (no duplicates)
        options = list(dict.fromkeys([name for _, name, _, _ in mapped_sorted]))
        pick = st.selectbox("Select a LAD with a zoning report:", options, index=0)

        # recover lad_key for picked name
        sel = next(item for item in mapped_sorted if item[1] == pick)
        lad_key = sel[0]
        r = df.loc[df["norm_lad"] == lad_key].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score", f"{safe_number(r.get('Investment_Potential_Score'),2)} / 100")
        c2.metric("Grade", str(r.get("Investment_Grade")))
        c3.metric("% aged 65+", safe_number(r.get("Percent_65plus"),2))
        c4.metric("House price growth %", safe_number(r.get("House_Price_Growth_%"),2))

        st.markdown("#### Zoning summary")
        z_exact_key = lad_key if lad_key in _zoning_blobs else best_match_norm(r["Local_Authority"], _zoning_blobs.keys(), cutoff=0.8)
        if z_exact_key and z_exact_key in _zoning_blobs:
            st.markdown(_zoning_blobs[z_exact_key])
        else:
            st.info("Zoning summary matched, but file could not be retrieved.")

        st.markdown("#### SHAP context (text)")
        shap_txt = _format_shap_drivers_for_context(lad_key, max_items=6)
        if shap_txt:
            st.markdown(shap_txt)
        else:
            st.info("No textual SHAP drivers were found for this LAD.")
        note_key = lad_key if lad_key in _shap_notes else f"{lad_key}_enriched" if f"{lad_key}_enriched" in _shap_notes else strip_enriched(lad_key)
        if note_key in _shap_notes:
            with st.expander("More SHAP notes"):
                st.markdown(_shap_notes[note_key])
                
# Footer (unchanged)
st.markdown("---")
st.caption("Created for the MSc Project (2025). Data includes demographic, financial, and quality-of-care features. SHAP images and text blobs are loaded when available.")
