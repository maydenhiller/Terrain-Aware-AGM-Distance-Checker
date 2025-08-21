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
        lines = []

        # Collect placemarks
        for placemark in root.findall(".//kml:Placemark", ns):
            name_elem = placemark.find("kml:name", ns)
            point = placemark.find(".//kml:Point/kml:coordinates", ns)
            if name_elem is not None and point is not None:
                name = name_elem.text.strip()
                if name.isnumeric():
                    lon, lat, *_ = point.text.strip().split(",")
                    placemarks.append((name, float(lat), float(lon)))

        # Collect line coords
        for linestring in root.findall(".//kml:LineString", ns):
            coords_text = linestring.find("kml:coordinates", ns).text.strip()
            coords = []
            for c in coords_text.split():
                lon, lat, *_ = c.split(",")
                coords.append((float(lat), float(lon)))
            lines.append(coords)

        # Debug info
        st.write(f"✅ Found {len(placemarks)} placemarks and {len(lines)} line(s).")
        if lines:
            st.write(f"First line has {len(lines[0])} points.")

        # Build map
        fmap = folium.Map(location=[39, -98], zoom_start=4)

        # Cap huge line draw for stability
        for coords in lines:
            if len(coords) > 5000:
                st.warning(f"Line has {len(coords)} points. Showing only first 5000 for preview.")
                coords = coords[:5000]
            folium.PolyLine(coords, color="red", weight=3).add_to(fmap)

        # Add markers
        for name, lat, lon in placemarks:
            folium.Marker(
                location=(lat, lon),
                popup=name,
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(fmap)

        st.subheader("🗺️ Map Preview")
        st_folium(fmap, width=800, height=600)

    except Exception as e:
        st.error(f"⚠️ Error while processing file: {e}")
