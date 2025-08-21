import streamlit as st

st.set_page_config(page_title="Terrain Distance Checker", layout="wide")

st.title("✅ Terrain Distance Checker")
st.write("App launched successfully!")

uploaded_file = st.file_uploader("Upload a KMZ file", type=["kmz"])

if uploaded_file:
    st.success(f"File `{uploaded_file.name}` uploaded successfully!")
