import zipfile 
import os
import io
import base64
import tempfile
import pandas as pd
import streamlit as st
from fastkml import kml
from shapely.geometry import LineString, Point
import numpy as np
import requests

st.set_page_config(page_title="Terrain Aware AGM Distance Checker")
st.title("Terrain Aware AGM Distance Checker")
st.markdown("Upload a KMZ file with a red centerline LineString and AGM point placemarks. The app will calculate terrain-aware distances between AGMs along the centerline and generate a CSV.")

def extract_kml_from_kmz(kmz_file):
    with zipfile.ZipFile(kmz_file) as kmz:
        for name in kmz.namelist():
            if name.endswith('.kml'):
                return kmz.read(name)
    return None

def parse_kml(kml_data):
    k = kml.KML()
    k.from_string(kml_data)
    features = list(k.features())
    document = list(features[0].features())
    agms = []
    centerline = None

    for f in document:
        if hasattr(f, 'geometry'):
            geom = f.geometry
            if isinstance(geom, LineString):
                if hasattr(f, 'styleUrl') and 'red' in str(f.styleUrl).lower():
                    centerline = geom
            elif isinstance(geom, Point):
                name = f.name.strip()
                if name.isdigit():  # Only include numeric AGMs
                    agms.append((name, geom))
    agms.sort(key=lambda x: int(x[0]))  # Sort by number
    return centerline, agms

def interpolate_points_along_line(line, num_points):
    distances = np.linspace(0, line.length, num_points)
    return [line.interpolate(d) for d in distances]

def get_terrain_distance_between(line, pt1, pt2, n=50):
    # Sample points along the segment of the line from pt1 to pt2
    proj1 = line.project(pt1)
    proj2 = line.project(pt2)
    if proj1 > proj2:
        proj1, proj2 = proj2, proj1
    segment = line.segmentize(max(1, int((proj2 - proj1) * 10)))
    sample_distances = np.linspace(proj1, proj2, n)
    sample_points = [line.interpolate(d) for d in sample_distances]
    lats_lons = [(pt.y, pt.x) for pt in sample_points]

    # Use open elevation API (SRTM-based, public, free)
    elevations = []
    for lat, lon in lats_lons:
        try:
            r = requests.get(f"https://api.opentopodata.org/v1/srtm90m?locations={lat},{lon}")
            result = r.json()
            elevations.append(result['results'][0]['elevation'])
        except:
            elevations.append(0)

    # Compute 3D terrain-aware distance
    total_distance = 0.0
    for i in range(1, len(sample_points)):
        dx = sample_points[i].x - sample_points[i-1].x
        dy = sample_points[i].y - sample_points[i-1].y
        dz = elevations[i] - elevations[i-1]
        d = np.sqrt((dx**2 + dy**2) * (111139**2) + dz**2)  # Convert deg to meters approx
        total_distance += d

    return total_distance * 3.28084  # convert to feet

def generate_csv(centerline, agms):
    rows = []
    cumulative_ft = 0.0
    for i in range(len(agms) - 1):
        start_name, start_pt = agms[i]
        end_name, end_pt = agms[i+1]
        dist_ft = get_terrain_distance_between(centerline, start_pt, end_pt)
        cumulative_ft += dist_ft
        rows.append({
            "Start AGM": start_name,
            "End AGM": end_name,
            "Distance (feet)": round(dist_ft, 2),
            "Distance (miles)": round(dist_ft / 5280, 4),
        })
    return pd.DataFrame(rows)

uploaded_file = st.file_uploader("Upload KMZ file", type="kmz")

if uploaded_file:
    kml_data = extract_kml_from_kmz(uploaded_file)
    if kml_data:
        centerline, agms = parse_kml(kml_data)
        if centerline and agms:
            st.success(f"Found centerline and {len(agms)} AGMs. Calculating distances...")
            df = generate_csv(centerline, agms)
            st.dataframe(df)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv_bytes, "terrain_distances.csv")

            st.download_button("Download Original KMZ", uploaded_file.read(), uploaded_file.name)
        else:
            st.error("Could not find a red centerline or valid AGMs in the KMZ.")
    else:
        st.error("Failed to extract KML from KMZ.")
