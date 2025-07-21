import streamlit as st
import zipfile
import os
import tempfile
from fastkml import kml
from shapely.geometry import LineString, Point
from pyproj import Geod
from xml.etree import ElementTree as ET

# Haversine with terrain (placeholder for real DEM lookup)
def haversine_with_terrain(p1, p2):
    g = Geod(ellps="WGS84")
    distance_2d = g.line_length([p1[0], p2[0]], [p1[1], p2[1]])
    return distance_2d  # terrain adjustment can go here

def extract_kml_from_kmz(uploaded_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(uploaded_file, 'r') as z:
            z.extractall(tmpdir)
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith('.kml'):
                        return open(os.path.join(root, file), 'rb').read()
    return None

def parse_kml(kml_data):
    k = kml.KML()
    k.from_string(kml_data)
    features = list(k.features())
    
    centerline = None
    agms = []

    def recursive_parse(features):
        nonlocal centerline, agms
        for f in features:
            if hasattr(f, 'geometry'):
                if isinstance(f.geometry, LineString):
                    if f.name.lower().startswith("red") or 'red' in f.name.lower():
                        centerline = f.geometry
                elif isinstance(f.geometry, Point):
                    if f.name and f.name.isnumeric():
                        agms.append((f.name, f.geometry))
            if hasattr(f, 'features'):
                recursive_parse(list(f.features()))

    recursive_parse(features)
    agms.sort(key=lambda x: int(x[0]))
    return centerline, agms

def calculate_distances(centerline, agms):
    points = [pt.coords[0] for _, pt in agms]
    segment_distances = []
    cumulative_distances = []

    total = 0
    for i in range(1, len(points)):
        dist = haversine_with_terrain(points[i - 1], points[i])
        total += dist
        segment_distances.append(dist)
        cumulative_distances.append(total)

    return segment_distances, cumulative_distances

# --- Streamlit UI ---
st.title("Terrain-Aware AGM Distance Checker")
uploaded_file = st.file_uploader("Upload a KMZ or KML file with a red centerline and numbered AGMs", type=["kmz", "kml"])

if uploaded_file:
    if uploaded_file.name.endswith(".kmz"):
        kml_data = extract_kml_from_kmz(uploaded_file)
    else:
        kml_data = uploaded_file.read()

    if kml_data:
        try:
            centerline, agms = parse_kml(kml_data)
            if centerline is None or len(agms) < 2:
                st.error("Could not find a valid red centerline or enough numbered AGMs.")
            else:
                seg_dists, cum_dists = calculate_distances(centerline, agms)

                st.subheader("Segment Distances (in meters)")
                for i, d in enumerate(seg_dists):
                    st.write(f"{agms[i][0]} to {agms[i+1][0]}: {d:.2f} m")

                st.subheader("Cumulative Distances (in meters)")
                for i, d in enumerate(cum_dists):
                    st.write(f"Up to {agms[i+1][0]}: {d:.2f} m")
        except Exception as e:
            st.error(f"Failed to parse KML: {e}")
    else:
        st.error("Could not extract KML from uploaded file.")
