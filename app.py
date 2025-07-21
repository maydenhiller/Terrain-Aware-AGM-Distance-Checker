import streamlit as st
from fastkml import kml
from shapely.geometry import LineString, Point
import zipfile
import tempfile
import os

st.set_page_config(page_title="Terrain-Aware AGM Distance Checker", layout="centered")
st.title("Terrain-Aware AGM Distance Checker")

uploaded_file = st.file_uploader("Upload a KMZ or KML file with a red centerline and numbered AGMs", type=["kmz", "kml"])

def parse_kml(kml_data):
    k = KML()
    k.from_string(kml_data)
    features = list(k.features()) if callable(k.features) else k.feature
    document = features[0]
    folder = list(document.features())[0]

    centerline = None
    agms = []

    for feature in folder.features():
        if isinstance(feature.geometry, LineString):
            centerline = feature.geometry
        elif isinstance(feature.geometry, Point):
            name = feature.name
            if name.isnumeric():  # Only include purely numeric placemarks
                agms.append((name, feature.geometry))

    return centerline, agms

if uploaded_file:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.TemporaryDirectory() as tmpdirname:
        if file_extension == ".kmz":
            with zipfile.ZipFile(uploaded_file, 'r') as kmz:
                kmz.extractall(tmpdirname)
                kml_files = [f for f in os.listdir(tmpdirname) if f.endswith('.kml')]
                if not kml_files:
                    st.error("No .kml file found inside the .kmz archive.")
                else:
                    kml_path = os.path.join(tmpdirname, kml_files[0])
                    with open(kml_path, 'rb') as f:
                        kml_data = f.read()
        else:
            kml_data = uploaded_file.read()

        try:
            centerline, agms = parse_kml(kml_data)
            st.success("File successfully parsed!")
            st.write(f"Centerline found: {'Yes' if centerline else 'No'}")
            st.write(f"Number of AGMs: {len(agms)}")
        except Exception as e:
            st.error(f"Failed to parse KML: {e}")
