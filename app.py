import streamlit as st
import zipfile
import os
import tempfile
from fastkml import kml
from shapely.geometry import LineString, Point
import requests
import math
import pandas as pd

# === CONFIGURATION ===
GOOGLE_ELEVATION_API_KEY = "AIzaSyB9HxznAvlGb02e-K1rhld_CPeAm_wvPWU"
ELEVATION_URL = "https://maps.googleapis.com/maps/api/elevation/json"

st.set_page_config(layout="wide")
st.title("📐 Terrain-Aware AGM Distance Checker")

uploaded_file = st.file_uploader("Upload KMZ or KML file", type=["kmz", "kml"])

def unzip_kmz(uploaded):
    with tempfile.TemporaryDirectory() as tmpdir:
        kmz_path = os.path.join(tmpdir, "temp.kmz")
        with open(kmz_path, "wb") as f:
            f.write(uploaded.getbuffer())
        with zipfile.ZipFile(kmz_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        for root, _, files in os.walk(tmpdir):
            for file in files:
                if file.endswith(".kml"):
                    return os.path.join(root, file)
    return None

def save_kml(uploaded):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".kml") as tmpfile:
        tmpfile.write(uploaded.getbuffer())
        return tmpfile.name

def parse_kml(path):
    with open(path, "rb") as f:
        doc = f.read()
    k = kml.KML()
    k.from_string(doc)
    return k

def find_folder(kml_obj, name):
    for doc in kml_obj.features():
        for folder in doc.features():
            if folder.name.strip().upper() == name.upper():
                return folder
    return None

def extract_agms(folder):
    agms = []
    for placemark in folder.features():
        if placemark.name.strip().isdigit():
            pt = placemark.geometry
            agms.append((placemark.name.strip(), pt.y, pt.x))
    agms.sort(key=lambda x: int(x[0]))
    return agms

def extract_red_line(folder):
    for placemark in folder.features():
        if hasattr(placemark, "styleUrl") and placemark.styleUrl and placemark.styleUrl.lower() == "#red":
            geom = placemark.geometry
            if isinstance(geom, LineString):
                return list(geom.coords)
    return None

def get_elevation(lat, lon):
    url = f"{ELEVATION_URL}?locations={lat},{lon}&key={GOOGLE_ELEVATION_API_KEY}"
    try:
        r = requests.get(url).json()
        return r["results"][0]["elevation"] if "results" in r and r["results"] else 0
    except:
        return 0

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def terrain_distance(p1, p2):
    lat1, lon1 = p1
    lat2, lon2 = p2
    elev1 = get_elevation(lat1, lon1)
    elev2 = get_elevation(lat2, lon2)
    flat = haversine(lat1, lon1, lat2, lon2)
    return math.sqrt(flat**2 + (elev2 - elev1)**2)

def nearest_index(pt, line):
    min_dist = float("inf")
    index = 0
    for i, coord in enumerate(line):
        d = haversine(pt[0], pt[1], coord[1], coord[0])
        if d < min_dist:
            min_dist = d
            index = i
    return index

if uploaded_file:
    with st.spinner("⏳ Processing file..."):
        try:
            if uploaded_file.name.endswith(".kmz"):
                kml_path = unzip_kmz(uploaded_file)
            else:
                kml_path = save_kml(uploaded_file)

            if not kml_path:
                st.error("❌ Could not extract KML from uploaded file.")
                st.stop()

            kml_obj = parse_kml(kml_path)
            centerline_folder = find_folder(kml_obj, "CENTERLINE")
            agm_folder = find_folder(kml_obj, "AGMs")

            if not centerline_folder or not agm_folder:
                st.error("❌ CENTERLINE or AGMs folder not found.")
                st.stop()

            centerline = extract_red_line(centerline_folder)
            agms = extract_agms(agm_folder)

            if not centerline:
                st.error("❌ No red centerline found in CENTERLINE folder.")
                st.stop()
            if len(agms) < 2:
                st.error("❌ Need at least two AGMs.")
                st.stop()

            results = []
            cumulative = 0
            for i in range(len(agms) - 1):
                name1, lat1, lon1 = agms[i]
                name2, lat2, lon2 = agms[i + 1]
                idx1 = nearest_index((lat1, lon1), centerline)
                idx2 = nearest_index((lat2, lon2), centerline)
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                seg_coords = centerline[idx1:idx2+1]
                dist = 0
                for j in range(len(seg_coords)-1):
                    pt1 = (seg_coords[j][1], seg_coords[j][0])
                    pt2 = (seg_coords[j+1][1], seg_coords[j+1][0])
                    dist += terrain_distance(pt1, pt2)
                cumulative += dist
                results.append({
                    "From": name1,
                    "To": name2,
                    "Segment (ft)": round(dist * 3.28084, 2),
                    "Segment (mi)": round(dist * 3.28084 / 5280, 4),
                    "Cumulative (ft)": round(cumulative * 3.28084, 2),
                    "Cumulative (mi)": round(cumulative * 3.28084 / 5280, 4)
                })

            df = pd.DataFrame(results)
            st.success("✅ Distances calculated!")
            st.dataframe(df)
            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Download CSV", csv, "terrain_distances.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")
