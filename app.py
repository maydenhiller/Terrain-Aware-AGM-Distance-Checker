import streamlit as st
from fastkml import kml
import requests
from xml.etree import ElementTree as ET

def parse_kml_coords(kml_file):
    coords = []
    content = kml_file.read().decode("utf-8")
    k = kml.KML()
    k.from_string(content)
    features = list(k.features())
    placemarks = list(features[0].features())
    for placemark in placemarks:
        geom = placemark.geometry
        if hasattr(geom, "coords"):
            coords.extend(geom.coords)
    return coords

def query_opentopo(lat, lon):
    url = f"https://portal.opentopography.org/API/globaldem?demtype=AW3D30&lat={lat}&lon={lon}&outputFormat=json"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("elevation", "No elevation data found")
        else:
            return f"HTTP {response.status_code} error"
    except Exception as e:
        return f"Request failed: {e}"

st.title("OpenTopography Diagnostic Tool")
uploaded_file = st.file_uploader("Upload KML file", type=["kml"])

if uploaded_file:
    st.success("KML file uploaded successfully.")
    coordinates = parse_kml_coords(uploaded_file)
    st.write(f"📌 Found {len(coordinates)} coordinate points.")

    st.subheader("Elevation Diagnostics:")
    for idx, (lon, lat) in enumerate(coordinates[:10]):  # Sample first 10
        elevation = query_opentopo(lat, lon)
        st.write(f"{idx+1}. ({lat:.6f}, {lon:.6f}) → Elevation: `{elevation}`")
