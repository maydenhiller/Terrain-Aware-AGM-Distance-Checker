import streamlit as st
import zipfile
import tempfile
import os
from fastkml import kml
from shapely.geometry import LineString, Point
import pyproj
import pandas as pd
import folium
from streamlit_folium import st_folium
import xml.etree.ElementTree as ET
import requests
import math

st.set_page_config(page_title="Terrain-Aware AGM Distance Checker", layout="wide")

st.title("🗺️ Terrain-Aware AGM Distance Checker")
st.markdown("Upload a **KML or KMZ file** with a **red centerline** and **numbered AGMs** (e.g. 000, 010, etc.). Placemarks starting with 'SP' will be ignored.")

uploaded_file = st.file_uploader("Upload .kmz or .kml", type=["kmz", "kml"])

def extract_kml_from_kmz(file_path):
    with zipfile.ZipFile(file_path, 'r') as z:
        for f in z.namelist():
            if f.endswith(".kml"):
                with z.open(f) as kml_file:
                    return kml_file.read()
    return None

def parse_kml(kml_data):
    k = kml.KML()
    k.from_string(kml_data)
    features = list(k.features)

    centerline = None
    agms = []

    def recursive_parse(features):
        nonlocal centerline, agms
        for f in features:
            if hasattr(f, 'geometry') and f.geometry:
                if isinstance(f.geometry, LineString):
                    if f.name and 'red' in f.name.lower():
                        centerline = f.geometry
                elif isinstance(f.geometry, Point):
                    if f.name and f.name.isnumeric():
                        agms.append((f.name, f.geometry))
            if hasattr(f, 'features') and f.features:
                recursive_parse(list(f.features))

    recursive_parse(features)
    agms.sort(key=lambda x: int(x[0]))
    return centerline, agms

def get_elevation(lat, lon):
    try:
        url = f"https://epqs.nationalmap.gov/v1/json?x={lon}&y={lat}&units=Feet&wkid=4326"
        response = requests.get(url)
        return response.json()['value']
    except:
        return 0

def haversine_3d(p1, p2):
    lat1, lon1, ele1 = p1
    lat2, lon2, ele2 = p2
    R = 6371000  # radius of Earth in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi/2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    horizontal = R * c
    vertical = ele2 - ele1
    return math.sqrt(horizontal**2 + vertical**2)

def get_nearest_point_on_line(line, point):
    return line.interpolate(line.project(point))

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".kmz" if uploaded_file.name.endswith(".kmz") else ".kml") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    kml_data = extract_kml_from_kmz(tmp_path) if uploaded_file.name.endswith(".kmz") else open(tmp_path, 'rb').read()

    try:
        centerline, agms = parse_kml(kml_data)
        if not centerline or not agms:
            st.error("Centerline or AGMs not found. Make sure the line is red and AGM points are purely numeric.")
        else:
            path_coords = list(centerline.coords)
            distances = []
            cumulative = 0
            previous_agm = None

            agm_coords = []

            for agm_name, agm_geom in agms:
                nearest = get_nearest_point_on_line(centerline, agm_geom)
                agm_coords.append((agm_name, nearest.y, nearest.x))

            agm_coords.sort(key=lambda x: int(x[0]))
            points_with_elev = []

            for name, lat, lon in agm_coords:
                elev = get_elevation(lat, lon)
                points_with_elev.append((name, lat, lon, elev))

            for i in range(1, len(points_with_elev)):
                a = points_with_elev[i - 1]
                b = points_with_elev[i]
                seg_dist_m = haversine_3d((a[1], a[2], a[3]), (b[1], b[2], b[3]))
                seg_dist_ft = seg_dist_m * 3.28084
                cumulative += seg_dist_ft
                distances.append({
                    "From": a[0],
                    "To": b[0],
                    "Segment Distance (ft)": round(seg_dist_ft, 2),
                    "Cumulative Distance (ft)": round(cumulative, 2),
                    "Cumulative Distance (mi)": round(cumulative / 5280, 4)
                })

            df = pd.DataFrame(distances)
            st.success("✅ Distances calculated successfully.")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "terrain_distances.csv", "text/csv")

            m = folium.Map(location=[points_with_elev[0][1], points_with_elev[0][2]], zoom_start=13)
            folium.PolyLine([(pt[1], pt[2]) for pt in points_with_elev], color="red").add_to(m)

            for pt in points_with_elev:
                folium.Marker(location=[pt[1], pt[2]], popup=f"{pt[0]} (Elev: {round(pt[3])} ft)").add_to(m)

            st_folium(m, height=500)

    except Exception as e:
        st.error(f"Failed to parse KML: {e}")

