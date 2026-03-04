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

```
care-home-ai-dashboard/
│
├── dashboard_app.py
│   Main Streamlit dashboard application
│
├── final_model_data_with_grade.csv
│   Processed dataset used for investment analysis
│
├── roi_by_district.csv
│   ROI simulation results by Local Authority District
│
├── requirements.txt
│   Python dependencies required to run the dashboard
│
├── README.md
│   Project documentation
│
├── shap_visuals/
│   SHAP explanation plots for each Local Authority District
│
├── SHAP/
│   Additional SHAP feature importance data and drivers
│
├── gpt_explanation/
│   GPT-generated explanations of investment potential for each district
│
├── roi_gpt/
│   GPT-generated summaries explaining ROI simulations
│
├── zoning_planning_summary/
│   Zoning and planning context summaries for selected districts
│
├── LAD_MAY_2025_Simplified.geojson
│   UK Local Authority District boundary file
│
├── LAD_MAY_2025_Simplified_5.geojson
│   Simplified boundary file used for faster dashboard map rendering
│
└── .devcontainer/
    Development container configuration
```

---

# Running the Dashboard

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch the Streamlit application

```bash
streamlit run dashboard_app.py
```

The dashboard will open locally in your browser.

### Website Link

https://care-home-ai-dashboard.streamlit.app/

---

# Deployment

The dashboard is deployed using **Streamlit Community Cloud**, allowing the application to run directly from the GitHub repository without additional infrastructure.

Users can interact with the dashboard through a web interface to explore investment opportunities across UK Local Authority Districts.

---

# Research Context

This project was developed as part of an MSc research project exploring how artificial intelligence and explainable machine learning can support investment decision-making in healthcare real estate.

The system demonstrates how predictive modelling, interpretability techniques, and natural language interfaces can be combined to improve transparency and usability in AI-driven analytics.

The project integrates:

- Machine learning prediction models  
- Explainable AI (SHAP)  
- Natural language explanation using large language models  
- Interactive decision support through a Streamlit dashboard  

---

# Author

Essa Abikar  
MSc Robotics  
King's College London

---

# Future Improvements

Potential extensions include:

- Incorporating additional regional economic indicators  
- Improving ROI modelling with historical investment performance  
- Integrating planning constraints directly into the scoring model  
- Expanding the LLM assistant for deeper investment analysis  
- Adding additional explainability methods alongside SHAP  
