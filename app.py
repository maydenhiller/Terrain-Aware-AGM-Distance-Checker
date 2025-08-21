import streamlit as st
import zipfile
import simplekml
import tempfile
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Terrain Distance Checker", layout="wide")
st.title("📍 Terrain Distance Checker")

uploaded_file = st.file_uploader("Upload a KMZ file", type=["kmz"])

if uploaded_file:
    # Save uploaded KMZ to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".kmz") as tmp:
        tmp.write(uploaded_file.read())
        kmz_path = tmp.name

    # Unzip KMZ to get KML
    with zipfile.ZipFile(kmz_path, "r") as z:
        kml_filename = [f for f in z.namelist() if f.endswith(".kml")][0]
        z.extract(kml_filename, tempfile.gettempdir())
        kml_path = tempfile.gettempdir() + "/" + kml_filename

    # Parse KML with simplekml
    kml = simplekml.Kml()
    with open(kml_path, "rb") as f:
        kml.from_kml(f.read())

    # Create Folium map
    fmap = folium.Map(location=[39, -98], zoom_start=4)

    # Plot red line (assuming first linestring is centerline)
    for ls in kml.linestrings:
        coords = [(p[1], p[0]) for p in ls.coords]  # lat, lon
        folium.PolyLine(coords, color="red", weight=3).add_to(fmap)

    # Plot numerical placemarks
    for pnt in kml.points:
        if pnt.name and pnt.name.isnumeric():
            folium.Marker(
                location=[pnt.coords[0][1], pnt.coords[0][0]],
                popup=pnt.name,
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(fmap)

    st.subheader("🗺️ Map Preview")
    st_data = st_foli
