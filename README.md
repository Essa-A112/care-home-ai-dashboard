# Care Home Investment AI Dashboard

AI-powered decision support system for identifying high-potential care home investment locations across the United Kingdom.

This project combines machine learning, explainable AI, and interactive visualisation to evaluate the investment potential of UK Local Authority Districts (LADs). The system analyses demographic trends, economic indicators, care home supply, and regulatory quality metrics to generate interpretable investment scores.

The dashboard provides investors and analysts with a transparent way to explore the factors influencing care home investment opportunities.

---

# Project Overview

The system evaluates each Local Authority District using a predictive model trained on multiple regional indicators. Results are presented through an interactive Streamlit dashboard that enables users to explore investment potential geographically and analytically.

Key capabilities include:

- Investment potential scoring for UK Local Authority Districts
- Interactive geographic visualisation of investment potential
- SHAP-based model explainability
- GPT-generated natural language investment explanations
- ROI simulation summaries for each district
- Zoning and planning report summaries where available

The aim is to demonstrate how AI-driven analysis can support strategic decision-making in healthcare real estate investment.

---

# Dashboard Features

## Investment Score Analysis
Each Local Authority District receives a machine learning–derived investment potential score based on demographic, economic, and healthcare indicators.

## Interactive Map
Users can explore investment scores geographically across the UK using an interactive choropleth map.

## SHAP Explainability
The model's predictions are interpreted using SHAP (Shapley Additive Explanations), allowing users to understand which factors drive the investment score for each region.

## ROI Simulation
Simulated return-on-investment estimates provide an additional financial perspective on potential care home developments.

## GPT Investment Assistant
A natural language assistant allows users to ask questions about specific regions, compare locations, or request explanations of model predictions.

## Zoning and Planning Reports
Planning summaries and zoning context are provided for selected high-potential districts.

---

# Data Sources

The model integrates data from several UK public datasets, including:

- Office for National Statistics (ONS)
- Care Quality Commission (CQC)
- UK House Price Index
- Regional Gross Disposable Household Income (GDHI)
- Population demographic estimates

These datasets are merged and processed at the Local Authority District level.

---

# Machine Learning Pipeline

The investment scoring model was trained using regional indicators including:

- Population aged 65+
- Care home supply per capita
- CQC quality ratings
- Regional income levels
- House price growth trends

Explainability is provided through SHAP analysis, enabling feature-level interpretation of model predictions.

---

# Project Structure
