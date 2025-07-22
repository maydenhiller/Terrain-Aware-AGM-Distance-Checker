import streamlit as st
import zipfile
import xml.etree.ElementTree as ET
import tempfile
import os
import pandas as pd
import math
import requests
from io import BytesIO
import numpy as np

# ✅ Prefilled API Key
GOOGLE_ELEVATION_API_KEY = "AIzaSyB9HxznAvlGb02e-K1rhld_CPeAm_wvPWU"

st.set_page_config(layout="wide")
st.title("📏 Terrain-Aware AGM Distance Calculator")

# ✅ Extract raw KML string from KMZ or KML upload
def extract_kml(uploaded_file):
    content = uploaded_file.getvalue()
    if uploaded_file.name.endswith(".kmz"):
        with zipfile.ZipFile(BytesIO(content), 'r') as kmz:
            for f in kmz.namelist():
                if f.endswith(".kml"):
                    return kmz.read(f).decode("utf-8")
        raise ValueError("No KML found inside KMZ.")
    else:
        return content.decode("utf-8")

# ✅ Parse the KML and extract AGMs and red CENTERLINE
def parse_kml(kml_string):
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    root = ET.fromstring(kml_string)
    agm_coords = {}
    centerline_coords = []

    folders = root.findall(".//kml:Folder", ns)
    for folder in folders:
        name_elem = folder.find("kml:name", ns)
        if name_elem is None:
            continue
        name = name_elem.text.strip().upper()

        if name == "AGMS":
            for placemark in folder.findall("kml:Placemark", ns):
                label = placemark.find("kml:name", ns)
                point = placemark.find(".//kml:Point/kml:coordinates", ns)
                if label is not None and point is not None:
                    if label.text.strip().isdigit():  # Skip SP labels
                        lon, lat, *_ = map(float, point.text.strip().split(","))
                        agm_coords[label.text.strip()] = (lat, lon)

        elif name == "CENTERLINE":
            for placemark in folder.findall("kml:Placemark", ns):
                style_url = placemark.find("kml:styleUrl", ns)
                if style_url is not None and "red" in style_url.text.lower():
                    coords_elem = placemark.find(".//kml:LineString/kml:coordinates", ns)
                    if coords_elem is not None:
                        coords = []
                        for line in coords_elem.text.strip().split():
                            lon, lat, *_ = map(float, line.split(","))
                            coords.append((lat, lon))
                        centerline_coords.extend(coords)

    if not agm_coords:
        raise ValueError("AGMs not found under AGMs folder.")
    if not centerline_coords:
        raise ValueError("No red centerline found inside the CENTERLINE folder.")

    return agm_coords, centerline_coords

# ✅ Sample elevation using Google Elevation API
def get_elevation(lat, lon):
    url = (
        f"https://maps.googleapis.com/maps/api/elevation/json?"
        f"locations={lat},{lon}&key={GOOGLE_ELEVATION_API_KEY}"
    )
    r = requests.get(url)
    result = r.json()
    if result["status"] == "OK":
        return result["results"][0]["elevation"]
    else:
        return 0.0

# ✅ Terrain-aware distance using 3D hypotenuse
def terrain_distance(p1, p2):
    lat1, lon1 = p1
    lat2, lon2 = p2
    elev1 = get_elevation(lat1, lon1)
    elev2 = get_elevation(lat2, lon2)
    R = 6371000  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    flat_dist = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    elev_diff = elev2 - elev1
    return math.sqrt(flat_dist**2 + elev_diff**2)  # meters

# ✅ Find nearest centerline vertex to a point
def find_nearest_point(point, line_coords):
    lat0, lon0 = point
    min_dist = float("inf")
    best_idx = 0
    for i, (lat1, lon1) in enumerate(line_coords):
        d = (lat0 - lat1)**2 + (lon0 - lon1)**2
        if d < min_dist:
            min_dist = d
            best_idx = i
    return best_idx

# ✅ Segment terrain distances between AGM labels
def compute_distances(agm_dict, centerline):
    sorted_agms = sorted((int(k), v) for k, v in agm_dict.items())
    results = []
    cumulative_meters = 0

    for i in range(len(sorted_agms) - 1):
        label_start = str(sorted_agms[i][0]).zfill(3)
        label_end = str(sorted_agms[i+1][0]).zfill(3)
        pt_start = sorted_agms[i][1]
        pt_end = sorted_agms[i+1][1]

        idx_start = find_nearest_point(pt_start, centerline)
        idx_end = find_nearest_point(pt_end, centerline)

        if idx_start > idx_end:
            idx_start, idx_end = idx_end, idx_start

        segment = centerline[idx_start:idx_end + 1]
        segment_dist = 0.0
        for j in range(len(segment) - 1):
            segment_dist += terrain_distance(segment[j], segment[j + 1])

        cumulative_meters += segment_dist
        segment_feet = segment_dist * 3.28084
        cumulative_feet = cumulative_meters * 3.28084
        segment_miles = segment_feet / 5280

        results.append({
            "From": label_start,
            "To": label_end,
            "Segment Distance (feet)": round(segment_feet, 2),
            "Segment Distance (miles)": round(segment_miles, 4),
            "Cumulative Distance (feet)": round(cumulative_feet, 2),
        })

    return pd.DataFrame(results)

# ✅ Upload and Process
uploaded = st.file_uploader("Upload KML/KMZ file", type=["kml", "kmz"])
if uploaded:
    try:
        kml_string = extract_kml(uploaded)
        agms, centerline = parse_kml(kml_string)
        df = compute_distances(agms, centerline)
        st.success("✅ Distances computed!")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "terrain_distances.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
