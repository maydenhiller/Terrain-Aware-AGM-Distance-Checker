import streamlit as st
import zipfile
import tempfile
import xml.etree.ElementTree as ET
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Terrain Distance Checker", layout="wide")
st.title("📍 Terrain Distance Checker")

uploaded_file = st.file_uploader("Upload a KMZ file", type=["kmz"])

if uploaded_file:
    # Save uploaded KMZ to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".kmz") as tmp:
        tmp.write(uploaded_file.read())
        kmz_path = tmp.name

    # Extract KML from KMZ
    with zipfile.ZipFile(kmz_path, "r") as z:
        kml_filename = [f for f in z.namelist() if f.endswith(".kml")][0]
        z.extract(kml_filename, tempfile.gettempdir())
        kml_path = tempfile.gettempdir() + "/" + kml_filename

    # Parse KML
    tree = ET.parse(kml_path)
    root = tree.getroot()

    # KML uses namespaces, so we define it
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    fmap = folium.Map(location=[39, -98], zoom_start=4)

    # Draw all LineStrings in red
    for linestring in root.findall(".//kml:LineString", ns):
        coords_text = linestring.find("kml:coordinates", ns).text.strip()
        coords = []
        for c in coords_text.split():
            lon, lat, *_ = c.split(",")
            coords.append((float(lat), float(lon)))
        folium.PolyLine(coords, color="red", weight=3).add_to(fmap)

    # Plot numerical placemarks
    for placemark in root.findall(".//kml:Placemark", ns):
        name_elem = placemark.find("kml:name", ns)
        point = placemark.find(".//kml:Point/kml:coordinates", ns)
        if name_elem is not None and point is not None:
            name = name_elem.text.strip()
            if name.isnumeric():
                lon, lat, *_ = point.text.strip().split(",")
                folium.Marker(
                    location=(float(lat), float(lon)),
                    popup=name,
                    icon=folium.Icon(color="blue", icon="info-sign"),
                ).add_to(fmap)

    st.subheader("🗺️ Map Preview")
    st_folium(fmap, width=800, height=600)
