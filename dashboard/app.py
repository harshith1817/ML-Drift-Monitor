import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="ML Drift Monitor", layout="wide")

st.title("ML Drift Monitoring Dashboard")

with open("../reports/data_drift.json") as f:
    feature_report=json.load(f)
    
with open("../reports/prediction_drift.json") as f:
    prediction_report=json.load(f)

with open("../reports/concept_drift.json") as f:
    concept_report=json.load(f)

st.header("Feature Drift")

rows=[]

for feature, stats in feature_report.items():
    rows.append({
        "Feature": feature,
        "Baseline Mean": stats["baseline_mean"],
        "Production Mean": stats["production_mean"],
        "KS p-value": stats["ks_test"]["p_value"],
        "KS Drift": stats["ks_test"]["drift_detected"],
        "PSI": stats["psi"]
    })

df=pd.DataFrame(rows)
st.dataframe(df)
st.subheader("PSI Distribution")
st.bar_chart(df.set_index("Feature")["PSI"])



st.header("Prediction Drift")

ks_drift=prediction_report["ks_test"]["drift_detected"]
psi_val=prediction_report["psi"]

col1, col2=st.columns(2)

col1.metric("Prediction KS Drift", ks_drift)
col2.metric("Preiction PSI", round(psi_val,3))

if ks_drift or psi_val>0.2:
    st.error("⚠ Prediction Drift Detected")
else:
    st.success("No Prediction Drift")
    
    
    
    
st.header("Concept Drift")

baseline_acc=concept_report["baseline_accuracy"]
prod_acc=concept_report["production_accuracy"]

col1,col2=st.columns(2)

col1.metric("Baseline Acuuracy", round(baseline_acc,3))
col2.metric("Production Accuracy", round(prod_acc,3))

if concept_report["concept_drift_detected"]:
    st.error("⚠ Concept Drift Detected")
else:
    st.success("No Concept Drift")