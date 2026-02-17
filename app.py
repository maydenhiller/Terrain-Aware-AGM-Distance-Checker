import streamlit as st
import zipfile
import tempfile
import os
import math
import requests
from shapely.geometry import LineString, Point
from fastkml import kml
import pandas as pd

st.set_page_config(layout="wide")

# ---------------- MAPBOX TOKEN ----------------
MAPBOX_TOKEN = st.secrets["mapbox"]["token"]

st.title("Terrain-Aware AGM Distance Checker")

uploaded_file = st.file_uploader("Upload KMZ", type=["kmz"])

if uploaded_file:

    # ---------- UNZIP KMZ ----------
    with tempfile.TemporaryDirectory() as tmpdir:
        kmz_path = os.path.join(tmpdir, "file.kmz")
        with open(kmz_path, "wb") as f:
            f.write(uploaded_file.read())

        with zipfile.ZipFile(kmz_path, 'r') as z:
            z.extractall(tmpdir)

        kml_file = [f for f in os.listdir(tmpdir) if f.endswith(".kml")][0]
        with open(os.path.join(tmpdir, kml_file), 'rb') as f:
            doc_bytes = f.read()

    # ---------- PARSE KML ----------
    k = kml.KML()
    k.from_string(doc_bytes)

    # --- WALK DOWN UNTIL WE FIND FOLDERS ---
    def get_all_features(container):
        feats = []
        if hasattr(container, "features") and container.features:
            for f in container.features:
                feats.append(f)
                feats.extend(get_all_features(f))
        return feats

    all_features = get_all_features(k)

    centerline = None
    agms_folder = None

    for f in all_features:
        if hasattr(f, "name") and f.name:
            name = f.name.lower()

            if name == "centerline":
                for feat in f.features:
                    if isinstance(feat.geometry, LineString):
                        centerline = feat.geometry

            if name == "agms":
                agms_folder = f

    if centerline is None or agms_folder is None:
        st.error("Missing Centerline or AGMs folder.")
        st.stop()

    # ---------- GET ALL PLACEMARKS IN AGMs ----------
    agm_points = []

    for placemark in agms_folder.features:
        if isinstance(placemark.geometry, Point):
            agm_points.append({
                "name": placemark.name,
                "lon": placemark.geometry.x,
                "lat": placemark.geometry.y
            })

    # ---------- PROJECT ONTO CENTERLINE ----------
    def project_onto_line(point):
        p = Point(point["lon"], point["lat"])
        d = centerline.project(p)
        snap = centerline.interpolate(d)
        return {
            "name": point["name"],
            "distance": d,
            "lon": snap.x,
            "lat": snap.y
        }

    projected = [project_onto_line(p) for p in agm_points]

    # ---------- SORT BY POSITION ALONG CENTERLINE ----------
    projected.sort(key=lambda x: x["distance"])

    # ---------- ELEVATION CACHE ----------
    @st.cache_data
    def get_elevation(lon, lat):
        url = f"https://api.mapbox.com/v4/mapbox.mapbox-terrain-v2/tilequery/{lon},{lat}.json"
        params = {
            "layers": "contour",
            "limit": 50,
            "access_token": MAPBOX_TOKEN
        }
        r = requests.get(url, params=params)
        data = r.json()
        return data["features"][0]["properties"]["ele"]

    # ---------- TERRAIN DISTANCE ----------
    def terrain_distance(p1, p2, samples=25):

        total = 0
        prev = None

        for i in range(samples + 1):
            t = i / samples
            lon = p1["lon"] + (p2["lon"] - p1["lon"]) * t
            lat = p1["lat"] + (p2["lat"] - p1["lat"]) * t
            ele = get_elevation(lon, lat)

            current = (lon, lat, ele)

            if prev:
                dx = (current[0] - prev[0]) * 364000
                dy = (current[1] - prev[1]) * 364000
                dz = current[2] - prev[2]
                total += math.sqrt(dx*dx + dy*dy + dz*dz)

            prev = current

        return total

    # ---------- CALCULATE DISTANCES ----------
    rows = []
    cumulative = 0

    for i in range(1, len(projected)):
        d = terrain_distance(projected[i-1], projected[i])
        cumulative += d

        rows.append({
            "From": projected[i-1]["name"],
            "To": projected[i]["name"],
            "Segment Feet": round(d, 2),
            "Cumulative Feet": round(cumulative, 2),
            "Segment Miles": round(d / 5280, 4),
            "Cumulative Miles": round(cumulative / 5280, 4)
        })

    df = pd.DataFrame(rows)

    st.dataframe(df)

    csv = df.to_csv(index=False).encode()
    st.download_button(
        "Download CSV",
        csv,
        "AGM_Terrain_Distances.csv",
        "text/csv"
    )
