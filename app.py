import streamlit as st
import zipfile
import tempfile
import xml.etree.ElementTree as ET
import folium
from streamlit_folium import st_folium
import os

st.set_page_config(page_title="Terrain Distance Checker", layout="wide")
st.title("📍 Terrain Distance Checker")

uploaded_file = st.file_uploader("Upload a KML or KMZ file", type=["kml", "kmz"])

if uploaded_file:
    try:
        # Handle KML vs KMZ
        if uploaded_file.name.endswith(".kml"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".kml") as tmp:
                tmp.write(uploaded_file.read())
                kml_path = tmp.name
        else:  # KMZ
            with tempfile.NamedTemporaryFile(delete=False, suffix=".kmz") as tmp:
                tmp.write(uploaded_file.read())
                kmz_path = tmp.name
            with zipfile.ZipFile(kmz_path, "r") as z:
                kml_filename = [f for f in z.namelist() if f.endswith(".kml")][0]
                z.extract(kml_filename, tempfile.gettempdir())
                kml_path = os.path.join(tempfile.gettempdir(), kml_filename)

        # Parse KML
        tree = ET.parse(kml_path)
        root = tree.getroot()
        ns = {"kml": "http://www.opengis.net/kml/2.2"}

        placemarks = []
        centerline = []

        # Collect placemarks (only numeric names)
        for placemark in root.findall(".//kml:Placemark", ns):
            name_elem = placemark.find("kml:name", ns)
            point = placemark.find(".//kml:Point/kml:coordinates", ns)
            if name_elem is not None and point is not None:
                name = name_elem.text.strip()
                if name.isnumeric():
                    lon, lat, *_ = point.text.strip().split(",")
                    placemarks.append((name, float(lat), float(lon)))

        # Look specifically for the CENTERLINE folder
        for folder in root.findall(".//kml:Folder", ns):
            fname = folder.find("kml:name", ns)
            if fname is not None and fname.text.strip().upper().startswith("CENTERLI
