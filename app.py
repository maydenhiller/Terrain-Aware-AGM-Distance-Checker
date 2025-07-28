import streamlit as st
from fastkml import kml
import requests
from xml.etree import ElementTree as ET

def parse_kml_coords(kml_file):
    try:
        # Read raw bytes (keep XML declaration intact)
        content = kml_file.read()

        # Quick XML sanity check on bytes
        ET.fromstring(content)

        # Parse KML bytes via FastKML
        k = kml.KML()
        k.from_string(content)

        coords = []
        # Iterate through all features and placemarks
        for feature in k.features():
            for placemark in feature.features():
                geom = placemark.geometry
                if hasattr(geom, "coords"):
                    coords.extend(geom.coords)

        return coords

    except Exception as e:
        st.error(f"⚠️ Failed to parse KML file: {e}")
        return []

def query_opentopo(lat, lon):
    url = (
        "https://portal.opentopography.org/API/globaldem"
        f"?demtype=AW3D30&lat={lat}&lon={lon}&outputFormat=json"
    )
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("elevation", "No elevation data returned")
        return f"HTTP error {resp.status_code}"
    except Exception as e:
        return f"Request failed: {e}"

# Streamlit UI
st.title("🧭 OpenTopography Elevation Diagnostic Tool")

uploaded_file = st.file_uploader(
    "Upload a KML file for elevation sampling", type=["kml"]
)

if uploaded_file:
    st.success("KML file uploaded successfully.")
    coordinates = parse_kml_coords(uploaded_file)
    total = len(coordinates)

    if total == 0:
        st.warning("No valid coordinates found in the KML.")
    else:
        st.write(f"📌 Found `{total}` coordinate points. Showing first 10:")
        st.subheader("🗻 Elevation Diagnostics")
        for idx, (lon, lat) in enumerate(coordinates[:10]):
            elev = query_opentopo(lat, lon)
            st.write(f"{idx+1}. ({lat:.6f}, {lon:.6f}) → Elevation: `{elev}`")
