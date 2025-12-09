# ui/app.py
import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "https://onboarding-api-637557748829.us-central1.run.app")

st.set_page_config(page_title="Onboarding UI", layout="centered")

st.title("Onboarding UI")
st.write("Replace this file with your production Streamlit UI. This is a working example that queries the API.")

query = st.text_input("Query", value="What is GitLab?")
if st.button("Ask API"):
    with st.spinner("Calling API..."):
        try:
            resp = requests.post(f"{API_URL}/query", json={"query": query}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            st.success("Got response")
            st.json(data)
        except Exception as e:
            st.error(f"API call failed: {e}")
            st.write("Tried:", f"{API_URL}/query")
