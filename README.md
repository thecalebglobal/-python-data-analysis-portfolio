# 🗂️ Python Data Analysis Portfolio
### Arogundade Caleb Oluwadamilola | TCG Analytics
---

## Overview
Five end-to-end Python data analysis projects spanning healthcare and business domains,
demonstrating advanced skills in EDA, machine learning, time series, customer segmentation,
and health economics analytics.

---

## Projects

### 📁 Project 1 — Maternal Health Risk Prediction
**File:** `project1_maternal_health.py`
**Domain:** Healthcare / Clinical Data Science
**Key Skills:** EDA, Multi-class Classification (Random Forest, Gradient Boosting, Logistic Regression),
Feature Importance, Model Evaluation (Cross-Validation, Confusion Matrix)
**Business Question:** Can clinical vitals (BP, glucose, temp) predict maternal risk level accurately?
**Output Charts:** Feature distributions, correlation heatmap, boxplots, model results
**Key Finding:** Blood Glucose + Systolic BP are the top predictors; RF achieves >90% accuracy

---

### 📁 Project 2 — Nigerian Retail Business Intelligence
**File:** `project2_nigerian_retail.py`
**Domain:** Business / Retail Analytics
**Key Skills:** Time Series Revenue Analysis, RFM Customer Segmentation,
K-Means Clustering, Cohort Analysis, Market Intelligence
**Business Question:** Who are our most valuable customers and what drives revenue growth?
**Output Charts:** Revenue trends, market intelligence, RFM segmentation
**Key Finding:** Champions segment (13% of customers) drive 40%+ of total revenue

---

### 📁 Project 3 — COVID-19 West Africa Epidemiological Analysis
**File:** `project3_covid_west_africa.py`
**Domain:** Healthcare / Public Health / Epidemiology
**Key Skills:** Time Series Analysis, Rolling Statistics, Comparative Country Analysis,
Case Fatality Rate (CFR) Modelling, Vaccination Impact Analysis
**Business Question:** How did COVID-19 spread across West Africa and what reduced transmission?
**Output Charts:** Epidemic curves (7-day rolling), normalized per-million comparison, CFR heatmap
**Key Finding:** Vaccination at 35%+ coverage correlated with significant case decline in Nigeria

---

### 📁 Project 4 — Telecom Customer Churn Prediction
**File:** `project4_churn_prediction.py`
**Domain:** Business / Telecommunications
**Key Skills:** Binary Classification Pipeline, Feature Engineering,
ROC-AUC Comparison, Business Impact Quantification, Revenue-at-Risk Analysis
**Business Question:** Which customers are most likely to churn and what is the revenue risk?
**Output Charts:** EDA (6 panels), ROC curves (4 models), feature importance
**Key Finding:** Month-to-Month contracts churn 3x more; Gradient Boosting AUC = best performer

---

### 📁 Project 5 — Hospital Financial & Clinical Operations Analytics
**File:** `project5_hospital_analytics.py`
**Domain:** Healthcare + Business (Health Economics)
**Key Skills:** Healthcare KPIs, Revenue Cycle Analysis, Department Benchmarking,
Insurance Claim Analytics, Readmission Cost Modelling, Diagnosis Cost Profiling
**Business Question:** Which departments and diagnoses drive revenue, and where is quality risk highest?
**Output Charts:** Department dashboard (4 panels), diagnosis cost + quarterly revenue trends
**Key Finding:** Oncology leads revenue; Emergency has highest readmission rate (22%)

---

## Tech Stack
| Library        | Usage                                    |
|----------------|------------------------------------------|
| `pandas`       | Data manipulation, groupby, time series  |
| `numpy`        | Numerical computation, simulation        |
| `matplotlib`   | Custom visualizations, dashboards        |
| `seaborn`      | Statistical plots, heatmaps              |
| `scikit-learn` | ML pipelines, evaluation, preprocessing  |
| `scipy`        | Statistical tests                        |

---

## How to Run
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy

# Run any project
python project1_maternal_health.py
python project2_nigerian_retail.py
python project3_covid_west_africa.py
python project4_churn_prediction.py
python project5_hospital_analytics.py
```

---

*Built by Arogundade Caleb Oluwadamilola — TCG Analytics | "Turning Data Into Decisions. Intelligence Into Impact."*
