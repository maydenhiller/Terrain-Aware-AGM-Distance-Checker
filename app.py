import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import requests
from shapely.geometry import LineString, Point
from fastkml import kml
from pyproj import Transformer

# --- CONFIG ---
API_KEY = "AIzaSyCd7sfheaJIbB8_J9Q9cxWb5jnv4U0K0LA"
ELEVATION_URL = "https://maps.googleapis.com/maps/api/elevation/json"
transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)

# --- AGM SORTING ---
def agm_sort_key(name_geom):
    name = name_geom[0]
    base = ''.join(filter(str.isdigit, name))
    suffix = ''.join(filter(str.isalpha, name))
    return (int(base), suffix)

# --- FILE PARSING ---
def parse_kml_kmz(uploaded_file):
    if uploaded_file.name.endswith(".kmz"):
        with zipfile.ZipFile(uploaded_file) as zf:
            kml_file = next((f for f in zf.namelist() if f.endswith(".kml")), None)
            with zf.open(kml_file) as f:
                kml_data = f.read()
    else:
        kml_data = uploaded_file.read()

    k = kml.KML()
    k.from_string(kml_data)
    agms = []
    centerline = None

    def extract_from_folder(folder):
        nonlocal agms, centerline
        folder_name = folder.name.strip().lower()
        if folder_name == "agms":
            for f in folder.features():
                if isinstance(f.geometry, Point):
                    agms.append((f.name.strip(), f.geometry))
        elif folder_name == "centerline":
            for f in folder.features():
                if isinstance(f.geometry, LineString):
                    centerline = f.geometry

    def recurse_features(features):
        for f in features:
            if hasattr(f, "features") and f.features():
                recurse_features(f.features())
            if hasattr(f, "name") and f.name:
                folder_name = f.name.strip().lower()
                if folder_name in ["agms", "centerline"]:
                    extract_from_folder(f)

    for feature in list(k.features()):
        recurse_features([feature])

    agms.sort(key=agm_sort_key)
    return agms, centerline

# --- CENTERLINE SLICING ---
def slice_centerline(centerline, p1, p2):
    coords = list(centerline.coords)
    idx1 = min(range(len(coords)), key=lambda i: Point(coords[i]).distance(p1))
    idx2 = min(range(len(coords)), key=lambda i: Point(coords[i]).distance(p2))
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    return LineString(coords[idx1:idx2+1])

# --- INTERPOLATION ---
def interpolate_line(line, spacing_m=1.0):
    coords = list(line.coords)
    points = [Point(coords[0])]
    for i in range(1, len(coords)):
        seg = LineString([coords[i-1], coords[i]])
        dist = seg.length
        steps = max(int(dist / spacing_m), 1)
        for j in range(1, steps + 1):
            points.append(seg.interpolate(j * spacing_m))
    return points

# --- ELEVATION API ---
def get_elevations(points):
    elevations = []
    batch_size = 512
    for i in range(0, len(points), batch_size):
        chunk = points[i:i+batch_size]
        locs = "|".join([f"{p.y},{p.x}" for p in chunk])
        response = requests.get(ELEVATION_URL, params={"locations": locs, "key": API_KEY})
        data = response.json()
        elevations.extend([r["elevation"] for r in data["results"]])
    return elevations

# --- 3D DISTANCE ---
def distance_3d(p1, p2, e1, e2):
    x1, y1 = transformer.transform(p1.x, p1.y)
    x2, y2 = transformer.transform(p2.x, p2.y)
    dz = e2 - e1
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + dz**2)

# --- STREAMLIT UI ---
st.title("Terrain-Aware AGM Distance Calculator")

uploaded_file = st.file_uploader("Upload KML or KMZ file", type=["kml", "kmz"])
if uploaded_file:
    agms, centerline = parse_kml_kmz(uploaded_file)

    if not centerline or len(agms) < 2:
        st.error("Missing CENTERLINE or insufficient AGM points.")
    else:
        rows = []
        cumulative_miles = 0.0

        for i in range(len(agms) - 1):
            name1, pt1 = agms[i]
            name2, pt2 = agms[i + 1]
            segment = slice_centerline(centerline, pt1, pt2)
            interp_points = interpolate_line(segment, spacing_m=1.0)
            elevations = get_elevations(interp_points)

            dist_m = sum(distance_3d(interp_points[j], interp_points[j+1], elevations[j], elevations[j+1]) for j in range(len(interp_points)-1))
            dist_ft = dist_m * 3.28084
            dist_mi = dist_ft / 5280
            cumulative_miles += dist_mi

            rows.append({
                "From AGM": name1,
                "To AGM": name2,
                "Distance (feet)": round(dist_ft, 2),
                "Distance (miles)": round(dist_mi, 6),
                "Cumulative Distance (miles)": round(cumulative_miles, 6)
            })

        df = pd.DataFrame(rows)
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "terrain_distances.csv", "text/csv")
