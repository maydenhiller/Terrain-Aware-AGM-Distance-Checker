import streamlit as st
from fastkml import kml
import requests
from xml.etree import ElementTree as ET

def get_features(obj):
    feat = getattr(obj, "features", None)
    if feat is None:
        return []
    return list(feat()) if callable(feat) else feat

def parse_kml_coords(kml_file):
    try:
        content = kml_file.read()  # keep bytes for XML declaration
        ET.fromstring(content)     # fast sanity check

        doc = kml.KML()
        doc.from_string(content)

        coords = []
        for feature in get_features(doc):
            for placemark in get_features(feature):
                geom = getattr(placemark, "geometry", None)
                if geom and hasattr(geom, "coords"):
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
            return resp.json().get("elevation", "No elevation data returned")
        return f"HTTP error {resp.status_code}"
    except Exception as e:
        return f"Request failed: {e}"

st.title("🧭 OpenTopography Elevation Diagnostic Tool")
uploaded_file = st.file_uploader("Upload a KML file for elevation sampling", type=["kml"])

if uploaded_file:
    st.success("KML file uploaded successfully.")
    coords = parse_kml_coords(uploaded_file)
    total = len(coords)

    if total == 0:
        st.warning("No valid coordinates found in the KML.")
    else:
        st.write(f"📌 Found `{total}` coordinate points. Showing first 10:")
        st.subheader("🗻 Elevation Diagnostics")
        for idx, (lon, lat) in enumerate(coords[:10]):
            elev = query_opentopo(lat, lon)
            st.write(f"{idx+1}. ({lat:.6f}, {lon:.6f}) → Elevation: `{elev}`")
