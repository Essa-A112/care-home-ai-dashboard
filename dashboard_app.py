# -*- coding: utf-8 -*-
"""
AI-Powered Care Home Investment Dashboard

Author: Essa Abikar
MSc Robotics – King's College London

This application visualises the output of a machine learning model
designed to identify UK locations with strong potential for care
home investment.

The dashboard combines:
• demographic indicators
• economic indicators
• care quality data
• machine learning investment scores
• SHAP explainability

It provides investors with an interpretable, data-driven overview
of the UK care home market.
"""

import os
import json
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# ==========================================================
# CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="UK Care Home Investment Analysis",
    page_icon="🏡",
    layout="wide"
)

DATA_PATH = "final_model_data_with_grade.csv"
GEOJSON_PATH = "LAD_MAY_2025_Simplified_5.geojson"
SHAP_FOLDER = "shap_visuals"

DEFAULT_MAP_CENTER = [54.5, -3.0]
DEFAULT_MAP_ZOOM = 5


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def safe_number(value, digits=2):
    try:
        return f"{float(value):.{digits}f}"
    except:
        return "N/A"


def compute_risk(row):
    """
    Simple market risk indicator based on
    supply saturation and quality indicators.
    """

    risk = 0

    if row["%_CQC_RequiresImprovement"] > 25:
        risk += 1

    if row["Care_Homes_per_10k"] > 6:
        risk += 1

    if row["House_Price_Growth_%"] < 2:
        risk += 1

    return risk


# ==========================================================
# DATA LOADING
# ==========================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data
def load_geojson():

    with open(GEOJSON_PATH) as f:
        geojson = json.load(f)

    return geojson


df = load_data()
geojson = load_geojson()


# ==========================================================
# SIDEBAR FILTERS
# ==========================================================

st.sidebar.header("Filters")

grades = sorted(df["Investment_Grade"].dropna().unique())

selected_grades = st.sidebar.multiselect(
    "Investment Grade",
    grades,
    default=grades
)

score_range = st.sidebar.slider(
    "Investment Score Range",
    0.0,
    100.0,
    (0.0, 100.0)
)

filtered = df[
    (df["Investment_Grade"].isin(selected_grades)) &
    (df["Investment_Potential_Score"].between(score_range[0], score_range[1]))
]


# ==========================================================
# MAIN TITLE
# ==========================================================

st.title("🏡 AI-Powered Care Home Investment Decision System")

st.caption(
    "MSc Robotics Project – Decision Support for UK Care Home Investment"
)


# ==========================================================
# TABS
# ==========================================================

tab_overview, tab_map, tab_compare, tab_method = st.tabs([
    "Overview",
    "Map",
    "Compare Locations",
    "Methodology"
])


# ==========================================================
# OVERVIEW TAB
# ==========================================================

with tab_overview:

    st.subheader("National Investment Snapshot")

    col1, col2, col3 = st.columns(3)

    col1.metric("Local Authorities Analysed", len(df))
    col2.metric("Median Investment Score", f"{df['Investment_Potential_Score'].median():.1f}")
    col3.metric("High Grade Areas (%)",
                f"{(df['Investment_Grade']=='Good').mean()*100:.1f}")


    st.markdown("### Top Investment Opportunities")

    top5 = filtered.sort_values(
        "Investment_Potential_Score",
        ascending=False
    ).head(5)

    for _, row in top5.iterrows():

        st.markdown(
            f"""
**{row['Local_Authority']}**

Score: {safe_number(row['Investment_Potential_Score'])}/100  
Grade: {row['Investment_Grade']}  
House Price Growth: {safe_number(row['House_Price_Growth_%'])}%  
Care Homes per 10k: {safe_number(row['Care_Homes_per_10k'])}
"""
        )

    st.markdown("### Top 15 Areas")

    table = filtered[
        [
            "Local_Authority",
            "Investment_Potential_Score",
            "Investment_Grade",
            "Percent_65plus",
            "%_CQC_Good",
            "House_Price_Growth_%"
        ]
    ].sort_values("Investment_Potential_Score", ascending=False).head(15)

    st.dataframe(table, use_container_width=True)


# ==========================================================
# MAP TAB
# ==========================================================

with tab_map:

    st.subheader("Interactive Investment Map")

    m = folium.Map(
        location=DEFAULT_MAP_CENTER,
        zoom_start=DEFAULT_MAP_ZOOM,
        tiles="cartodbpositron"
    )

    choropleth = folium.Choropleth(
        geo_data=geojson,
        data=filtered,
        columns=["Local_Authority", "Investment_Potential_Score"],
        key_on="feature.properties.LAD25NM",
        fill_color="YlGnBu",
        legend_name="Investment Score"
    ).add_to(m)

    st_folium(m, height=550)

    st.markdown("---")

    lad_list = filtered["Local_Authority"].unique()

    selected_lad = st.selectbox(
        "Select Location",
        lad_list
    )

    row = filtered[filtered["Local_Authority"] == selected_lad].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Score", f"{safe_number(row['Investment_Potential_Score'])}/100")
    col2.metric("Grade", row["Investment_Grade"])
    col3.metric("% aged 65+", safe_number(row["Percent_65plus"]))
    col4.metric("House Price Growth", safe_number(row["House_Price_Growth_%"]))


    risk = compute_risk(row)

    if risk == 0:
        st.success("Low investment risk")
    elif risk == 1:
        st.warning("Moderate investment risk")
    else:
        st.error("Higher investment risk")


    shap_path = os.path.join(
        SHAP_FOLDER,
        f"{selected_lad.lower().replace(' ','_')}.png"
    )

    st.markdown("### Model Explanation")

    if os.path.exists(shap_path):
        st.image(shap_path, width=800)
    else:
        st.info("No SHAP explanation available for this location.")


# ==========================================================
# COMPARISON TAB
# ==========================================================

with tab_compare:

    st.subheader("Compare Investment Locations")

    col1, col2 = st.columns(2)

    lad1 = col1.selectbox("Location 1", df["Local_Authority"])
    lad2 = col2.selectbox("Location 2", df["Local_Authority"], index=1)

    row1 = df[df["Local_Authority"] == lad1].iloc[0]
    row2 = df[df["Local_Authority"] == lad2].iloc[0]

    comparison = pd.DataFrame({

        "Metric": [
            "Investment Score",
            "% Population 65+",
            "House Price Growth %",
            "Care Homes per 10k",
            "CQC Good %"
        ],

        lad1: [
            row1["Investment_Potential_Score"],
            row1["Percent_65plus"],
            row1["House_Price_Growth_%"],
            row1["Care_Homes_per_10k"],
            row1["%_CQC_Good"]
        ],

        lad2: [
            row2["Investment_Potential_Score"],
            row2["Percent_65plus"],
            row2["House_Price_Growth_%"],
            row2["Care_Homes_per_10k"],
            row2["%_CQC_Good"]
        ]

    })

    st.dataframe(comparison, use_container_width=True)


# ==========================================================
# METHODOLOGY TAB
# ==========================================================

with tab_method:

    st.subheader("Methodology")

    st.markdown("""

### Model Overview

This dashboard presents results from a machine learning model designed
to identify UK Local Authority Districts with strong potential for
care home investment.

The model combines demographic, economic and healthcare quality
indicators.

### Key Features Used

• Percentage of population aged 65+  
• Gross Disposable Household Income (GDHI)  
• House price growth  
• Care home supply per capita  
• Care Quality Commission ratings  

### Explainability

SHAP (Shapley Additive Explanations) values are used to interpret how
each feature influences the predicted investment score for a given
location.

This enables transparent and interpretable investment analysis.

""")


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Developed as part of MSc research into AI-driven investment decision support systems."
)
