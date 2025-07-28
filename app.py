import streamlit as st
from fastkml import kml
import requests
from xml.etree import ElementTree as ET

def parse_kml_coords(kml_file):
    try:
        content = kml_file.read().decode("utf-8")
        ET.fromstring(content)  # Quick XML sanity check

        k = kml.KML()
        k.from_string(content)

        coords = []
        features = list(k.features())
        for feature in features:
            placemarks = list(feature.features())
            for placemark in placemarks:
                geom = placemark.geometry
                if hasattr(geom, "coords"):
                    coords.extend(geom.coords)

        return coords

    except Exception as e:
        st.error(f"⚠️ Failed to parse KML file: {e}")
        return []

def query_opentopo(lat, lon):
    url = f"https://portal.opentopography.org/API/globaldem?demtype=AW3D30&lat={lat}&lon={lon}&outputFormat=json"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("elevation", "No elevation data returned")
        else:
            return f"HTTP error {response.status_code}"
    except Exception as e:
        return f"Request failed: {e}"

# Streamlit UI
st.title("🧭 OpenTopography Elevation Diagnostic Tool")
uploaded_file = st.file_uploader("Upload a KML file for elevation sampling", type=["kml"])

if uploaded_file:
    st.success("KML file uploaded successfully.")
    coordinates = parse_kml_coords(uploaded_file)
    num_coords = len(coordinates)
    
    if num_coords == 0:
        st.warning("No valid coordinates found.")
    else:
        st.write(f"📌 Found `{num_coords}` coordinates. Showing elevation for first 10:")
        st.subheader("🗻 Elevation Diagnostics")
        
        for idx, (lon, lat) in enumerate(coordinates[:10]):
            elevation = query_opentopo(lat, lon)
            st.write(f"{idx+1}. ({lat:.6f}, {lon:.6f}) → Elevation: `{elevation}`")
