import streamlit as st
import zipfile
import tempfile
import os
import xml.etree.ElementTree as ET
from math import radians, cos, sin, sqrt
import pandas as pd
import requests

# --- Constants ---
ELEVATION_API_KEY = "AIzaSyB9HxznAvlGb02e-K1rhld_CPeAm_wvPWU"
ELEVATION_API_URL = "https://maps.googleapis.com/maps/api/elevation/json"

# --- Functions ---

def parse_kml_from_kmz(uploaded_file):
    with tempfile.TemporaryDirectory() as tmpdirname:
        kmz_path = os.path.join(tmpdirname, "uploaded.kmz")
        with open(kmz_path, "wb") as f:
            f.write(uploaded_file.read())

        with zipfile.ZipFile(kmz_path, "r") as kmz:
            for name in kmz.namelist():
                if name.endswith(".kml"):
                    kmz.extract(name, tmpdirname)
                    return os.path.join(tmpdirname, name)
    return None

def extract_coords(coord_text):
    coords = []
    for line in coord_text.strip().split():
        parts = line.split(",")
        if len(parts) >= 2:
            lon, lat = float(parts[0]), float(parts[1])
            coords.append((lat, lon))
    return coords

def get_coords_from_folder(folder, target_name):
    for sub in folder.findall(".//{http://www.opengis.net/kml/2.2}Folder"):
        name = sub.find("{http://www.opengis.net/kml/2.2}name")
        if name is not None and name.text.strip().upper() == target_name:
            return sub
    return None

def get_centerline_coords(folder):
    for pm in folder.findall(".//{http://www.opengis.net/kml/2.2}Placemark"):
        line = pm.find("{http://www.opengis.net/kml/2.2}LineString")
        if line is not None:
            coords = line.find("{http://www.opengis.net/kml/2.2}coordinates")
            if coords is not None:
                return extract_coords(coords.text)
    return []

def get_agm_points(folder):
    agms = {}
    for pm in folder.findall(".//{http://www.opengis.net/kml/2.2}Placemark"):
        name_el = pm.find("{http://www.opengis.net/kml/2.2}name")
        pt = pm.find(".//{http://www.opengis.net/kml/2.2}Point")
        if name_el is not None and pt is not None:
            try:
                name = name_el.text.strip()
                if name.isnumeric():
                    coord = pt.find("{http://www.opengis.net/kml/2.2}coordinates")
                    if coord is not None:
                        lon, lat = map(float, coord.text.strip().split(",")[:2])
                        agms[name] = (lat, lon)
            except Exception:
                continue
    return dict(sorted(agms.items(), key=lambda x: int(x[0])))

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2*R*sqrt(a)

def interpolate_point_on_line(p1, p2, p):
    x1, y1 = p1[1], p1[0]
    x2, y2 = p2[1], p2[0]
    x0, y0 = p[1], p[0]

    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0:
        return p1, 0
    t = ((x0 - x1)*dx + (y0 - y1)*dy) / (dx*dx + dy*dy)
    t = max(0, min(1, t))
    interp = (y1 + t*dy, x1 + t*dx)
    return interp, t

def project_onto_centerline(agm, centerline):
    min_dist = float('inf')
    closest_point = None
    segment_index = 0
    for i in range(len(centerline) - 1):
        proj, t = interpolate_point_on_line(centerline[i], centerline[i+1], agm)
        dist = haversine(agm[0], agm[1], proj[0], proj[1])
        if dist < min_dist:
            min_dist = dist
            closest_point = proj
            segment_index = i + t
    return closest_point, segment_index

def get_elevation(lat, lon):
    try:
        url = f"{ELEVATION_API_URL}?locations={lat},{lon}&key={ELEVATION_API_KEY}"
        response = requests.get(url).json()
        if response["status"] == "OK":
            return response["results"][0]["elevation"]
    except:
        pass
    return 0

def terrain_distance(p1, p2):
    flat_dist = haversine(p1[0], p1[1], p2[0], p2[1])
    elev1 = get_elevation(*p1)
    elev2 = get_elevation(*p2)
    elev_diff = elev2 - elev1
    return sqrt(flat_dist**2 + elev_diff**2)

# --- Streamlit UI ---

st.set_page_config(layout="wide")
st.title("📏 Terrain-Aware Distance Calculator")

uploaded_file = st.file_uploader("Upload a KMZ or KML file", type=["kmz", "kml"])

if uploaded_file:
    if uploaded_file.name.endswith(".kmz"):
        kml_path = parse_kml_from_kmz(uploaded_file)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".kml") as tmp:
            tmp.write(uploaded_file.read())
            kml_path = tmp.name

    if not kml_path or not os.path.exists(kml_path):
        st.error("❌ Error: Failed to extract KML from KMZ.")
        st.stop()

    try:
        tree = ET.parse(kml_path)
        root = tree.getroot()
        ns = {"kml": "http://www.opengis.net/kml/2.2"}

        folders = root.findall(".//kml:Folder", ns)

        agm_folder = get_coords_from_folder(root, "AGMs")
        cl_folder = get_coords_from_folder(root, "CENTERLINE")

        if agm_folder is None or cl_folder is None:
            st.error("❌ Error: 'CENTERLINE' or 'AGMs' folder not found.")
            st.stop()

        centerline = get_centerline_coords(cl_folder)
        agm_points = get_agm_points(agm_folder)

        if not centerline or not agm_points:
            st.error("❌ Error: Red centerline or AGMs not found.")
            st.stop()

        # Project AGMs
        agm_proj = {}
        for name, pt in agm_points.items():
            proj, idx = project_onto_centerline(pt, centerline)
            agm_proj[name] = (proj, idx)

        sorted_agms = sorted(agm_proj.items(), key=lambda x: x[1][1])
        distances = []
        total_dist_ft = 0

        for i in range(1, len(sorted_agms)):
            prev = sorted_agms[i-1][1][0]
            curr = sorted_agms[i][1][0]
            dist = terrain_distance(prev, curr)
            dist_ft = dist * 3.28084
            dist_mi = dist_ft / 5280
            total_dist_ft += dist_ft
            distances.append({
                "From": sorted_agms[i-1][0],
                "To": sorted_agms[i][0],
                "Segment Distance (ft)": round(dist_ft, 2),
                "Segment Distance (mi)": round(dist_mi, 4),
                "Cumulative Distance (mi)": round(total_dist_ft / 5280, 4)
            })

        df = pd.DataFrame(distances)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, file_name="Terrain_Distances.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
