import streamlit as st
import zipfile
import os
import tempfile
from fastkml import kml
from shapely.geometry import LineString, Point
import xml.etree.ElementTree as ET
import requests
import math
import pandas as pd

# === CONFIG ===
GOOGLE_ELEVATION_API_KEY = "AIzaSyB9HxznAvlGb02e-K1rhld_CPeAm_wvPWU"
ELEVATION_BASE_URL = "https://maps.googleapis.com/maps/api/elevation/json"

# === STREAMLIT SETUP ===
st.set_page_config(layout="wide")
st.title("📏 Terrain-Aware Distance Calculator")

uploaded_file = st.file_uploader("Upload KMZ or KML file", type=["kmz", "kml"])

def unzip_kmz(kmz_file):
    with tempfile.TemporaryDirectory() as tmpdirname:
        kmz_path = os.path.join(tmpdirname, "temp.kmz")
        with open(kmz_path, "wb") as f:
            f.write(kmz_file.getbuffer())
        with zipfile.ZipFile(kmz_path, "r") as zip_ref:
            zip_ref.extractall(tmpdirname)
        for root, _, files in os.walk(tmpdirname):
            for file in files:
                if file.endswith(".kml"):
                    return os.path.join(root, file)
    return None

def parse_kml(kml_path):
    with open(kml_path, "rb") as f:
        kml_data = f.read()
    k = kml.KML()
    k.from_string(kml_data)
    return k

def find_folder_by_name(kml_obj, target):
    for doc in kml_obj.features():
        for folder in doc.features():
            if folder.name.strip().upper() == target.upper():
                return folder
    return None

def extract_agms(agms_folder):
    agms = []
    for placemark in agms_folder.features():
        if placemark.name.strip().isdigit():
            point = placemark.geometry
            agms.append((placemark.name.strip(), point.y, point.x))
    agms.sort(key=lambda x: int(x[0]))
    return agms

def extract_red_centerline(centerline_folder):
    for placemark in centerline_folder.features():
        if hasattr(placemark, 'styleUrl') and placemark.styleUrl:
            if placemark.styleUrl.lower() == "#red":
                geom = placemark.geometry
                if isinstance(geom, LineString):
                    return list(geom.coords)
    return None

def interpolate_elevation(lat, lon):
    url = f"{ELEVATION_BASE_URL}?locations={lat},{lon}&key={GOOGLE_ELEVATION_API_KEY}"
    response = requests.get(url).json()
    if "results" in response and len(response["results"]) > 0:
        return response["results"][0]["elevation"]
    return 0

def terrain_distance(p1, p2):
    lat1, lon1 = p1[:2]
    lat2, lon2 = p2[:2]
    ele1 = interpolate_elevation(lat1, lon1)
    ele2 = interpolate_elevation(lat2, lon2)
    d = math.dist((lat1, lon1), (lat2, lon2))
    horiz = haversine(lat1, lon1, lat2, lon2)
    elev_diff = ele2 - ele1
    return math.sqrt(horiz**2 + elev_diff**2)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def find_closest_index_on_centerline(point, centerline):
    min_dist = float("inf")
    min_index = 0
    for i, coord in enumerate(centerline):
        dist = haversine(point[0], point[1], coord[1], coord[0])
        if dist < min_dist:
            min_dist = dist
            min_index = i
    return min_index

if uploaded_file:
    with st.spinner("Processing file..."):
        try:
            kml_path = unzip_kmz(uploaded_file) if uploaded_file.name.endswith(".kmz") else uploaded_file
            if not kml_path:
                st.error("❌ Could not find a KML inside the KMZ file.")
                st.stop()
            kml_obj = parse_kml(kml_path)
            centerline_folder = find_folder_by_name(kml_obj, "CENTERLINE")
            agms_folder = find_folder_by_name(kml_obj, "AGMs")

            if not centerline_folder or not agms_folder:
                st.error("❌ Error: CENTERLINE or AGMs folder not found.")
                st.stop()

            centerline_coords = extract_red_centerline(centerline_folder)
            if not centerline_coords:
                st.error("❌ Error: No red centerline found inside the CENTERLINE folder.")
                st.stop()

            agms = extract_agms(agms_folder)
            if len(agms) < 2:
                st.error("❌ Need at least two AGMs to measure distances.")
                st.stop()

            results = []
            cumulative = 0

            for i in range(len(agms) - 1):
                name1, lat1, lon1 = agms[i]
                name2, lat2, lon2 = agms[i + 1]
                idx1 = find_closest_index_on_centerline((lat1, lon1), centerline_coords)
                idx2 = find_closest_index_on_centerline((lat2, lon2), centerline_coords)

                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1

                seg_coords = centerline_coords[idx1:idx2 + 1]
                segment_dist = 0

                for j in range(len(seg_coords) - 1):
                    pt1 = seg_coords[j]
                    pt2 = seg_coords[j + 1]
                    latlon1 = (pt1[1], pt1[0])
                    latlon2 = (pt2[1], pt2[0])
                    segment_dist += terrain_distance(latlon1, latlon2)

                cumulative += segment_dist
                results.append({
                    "From": name1,
                    "To": name2,
                    "Segment (ft)": round(segment_dist * 3.28084, 2),
                    "Segment (mi)": round(segment_dist * 3.28084 / 5280, 4),
                    "Cumulative (ft)": round(cumulative * 3.28084, 2),
                    "Cumulative (mi)": round(cumulative * 3.28084 / 5280, 4)
                })

            df = pd.DataFrame(results)
            st.success("✅ Terrain-aware distances calculated!")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, "terrain_distances.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
