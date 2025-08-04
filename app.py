import streamlit as st
import zipfile
import simplekml
import io
import polyline
import requests
import pandas as pd
import math
from pykml import parser
from lxml import etree
from geopy.distance import geodesic

# 📍 Elevation API config (Open-Elevation)
def fetch_elevation(lat, lon):
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    try:
        response = requests.get(url)
        data = response.json()
        return data['results'][0]['elevation']
    except Exception:
        return 0  # fallback if API fails

# 📍 3D distance calculator using terrain
def terrain_distance(path):
    total = 0
    for i in range(len(path) - 1):
        lat1, lon1 = path[i]
        lat2, lon2 = path[i+1]
        elev1 = fetch_elevation(lat1, lon1)
        elev2 = fetch_elevation(lat2, lon2)
        horizontal = geodesic((lat1, lon1), (lat2, lon2)).feet
        vertical = elev2 - elev1
        total += math.sqrt(horizontal**2 + vertical**2)
    return total

# 📍 KMZ/KML parser
def extract_kml_from_kmz(kmz_file):
    with zipfile.ZipFile(kmz_file, 'r') as zf:
        for name in zf.namelist():
            if name.endswith('.kml'):
                return zf.read(name)
    return None

def extract_agms_and_centerline(kml_data):
    agms = {}
    centerline_coords = []

    kml_root = parser.fromstring(kml_data)
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    for folder in kml_root.Document.Folder:
        if hasattr(folder, 'name') and folder.name == 'AGMs':
            for pm in folder.Placemark:
                name = str(pm.name)
                coord_text = str(pm.Point.coordinates).strip()
                lon, lat, _ = map(float, coord_text.split(','))
                agms[name] = (lat, lon)

        if hasattr(folder, 'name') and folder.name == 'Centerline':
            for pm in folder.Placemark:
                if hasattr(pm, 'LineString'):
                    coords_text = str(pm.LineString.coordinates).strip()
                    for line in coords_text.split():
                        lon, lat, _ = map(float, line.split(','))
                        centerline_coords.append((lat, lon))

    return agms, centerline_coords

# 📍 AGM path segment extractor
def get_path_segment(centerline, pt1, pt2):
    def closest_index(point):
        return min(range(len(centerline)), key=lambda i: geodesic(centerline[i], point).feet)
    i1 = closest_index(pt1)
    i2 = closest_index(pt2)
    return centerline[min(i1, i2):max(i1, i2)+1]

# 🖥️ Streamlit App
st.title("AGM Terrain-Aware Distance Calculator")
uploaded_file = st.file_uploader("Upload KMZ file with 'AGMs' and 'Centerline' folders", type="kmz")

if uploaded_file:
    kml_raw = extract_kml_from_kmz(uploaded_file)
    agms, centerline = extract_agms_and_centerline(kml_raw)

    ordered_agms = sorted(agms.items(), key=lambda x: int(x[0].split()[-1]))
    distances = []

    st.write(f"Processing {len(ordered_agms)} AGMs…")
    for i in range(len(ordered_agms) - 1):
        name1, pt1 = ordered
