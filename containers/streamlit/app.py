import streamlit as st
import requests

st.title("MLOps Product Dashboard")

# Example MLflow Configuration
st.header("MLflow Configuration")
tracking_uri = st.text_input("MLflow Tracking URI", "http://localhost:5000")
artifact_uri = st.text_input("Artifact Store URI", "s3://mlflow")

if st.button("Test Connection"):
    try:
        r = requests.get(f"{tracking_uri}/health")
        st.success("Connection successful!" if r.status_code == 200 else "Failed")
    except Exception as e:
        st.error(f"Error: {e}")
