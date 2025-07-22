import streamlit as st
import zipfile
import simplekml
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import math
import os

# --- App Config ---
st.set_page_config(layout="wide")
st.title("📏 Terrain-Aware Distance Calculator")

# --- Helper Functions ---

def haversine_3d(lat1, lon1, ele1, lat2, lon2, ele2):
    R = 6371000  # Radius of Earth in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    dx = R * d_lambda * math.cos((phi1 + phi2) / 2)
    dy = R * d_phi
    dz = ele2 - ele1
    return math.sqrt(dx**2 + dy**2 + dz**2)

def parse_kml_root(root):
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    centerline_coords = None
    agm_points = []

    for folder in root.findall(".//kml:Folder", ns):
        name = folder.find("kml:name", ns)
        if name is None:
            continue
        if name.text.strip().upper() == "CENTERLINE":
            for pl in folder.findall(".//kml:Placemark", ns):
                style = pl.find("kml:Style", ns)
                if style is not None:
                    line_style = style.find("kml:LineStyle/kml:color", ns)
                    if line_style is not None and line_style.text.strip().lower() == "ff0000ff":  # Red (ABGR)
                        coords = pl.find(".//kml:coordinates", ns)
                        if coords is not None:
                            raw = coords.text.strip().split()
                            centerline_coords = [tuple(map(float, pt.split(',')[:3])) for pt in raw]
        elif name.text.strip().upper() == "AGMS":
            for pl in folder.findall(".//kml:Placemark", ns):
                pname = pl.find("kml:name", ns)
                if pname is None or not pname.text.strip().isdigit():
                    continue
                coords = pl.find(".//kml:coordinates", ns)
                if coords is not None:
                    lon, lat, ele = map(float, coords.text.strip().split(',')[:3])
                    agm_points.append((pname.text.strip(), lat, lon, ele))

    return centerline_coords, sorted(agm_points, key=lambda x: int(x[0]))

def load_kml_or_kmz(file) -> tuple:
    try:
        ext = os.path.splitext(file.name)[1].lower()
        with tempfile.TemporaryDirectory() as tmpdir:
            if ext == ".kmz":
                with zipfile.ZipFile(file) as kmz:
                    kmz.extractall(tmpdir)
                    for fname in os.listdir(tmpdir):
                        if fname.endswith(".kml"):
                            path = os.path.join(tmpdir, fname)
                            tree = ET.parse(path)
                            return parse_kml_root(tree.getroot())
            elif ext == ".kml":
                tree = ET.parse(file)
                return parse_kml_root(tree.getroot())
    except Exception as e:
        st.error(f"❌ Failed to parse KMZ/KML: {e}")
        return None, None

# --- Upload ---
uploaded = st.file_uploader("📂 Upload your KML or KMZ file", type=["kml", "kmz"])

if uploaded:
    centerline, agms = load_kml_or_kmz(uploaded)

    if not centerline:
        st.warning("⚠️ No red centerline found inside the CENTERLINE folder.")
    elif not agms:
        st.warning("⚠️ No numeric AGMs found in AGMs folder.")
    else:
        # Interpolate points along the centerline to follow its path between AGMs
        def snap_to_path(lat, lon, path):
            return min(path, key=lambda pt: math.hypot(lat - pt[1], lon - pt[0]))

        snapped_agms = [(name, *snap_to_path(lat, lon, centerline)) for name, lat, lon, ele in agms]

        distances = []
        cum_dist = 0
        for i in range(1, len(snapped_agms)):
            prev = snapped_agms[i - 1]
            curr = snapped_agms[i]
            seg_dist = haversine_3d(prev[2], prev[1], prev[3], curr[2], curr[1], curr[3])
            cum_dist += seg_dist
            distances.append({
                "From": prev[0],
                "To": curr[0],
                "Segment_Distance_ft": round(seg_dist * 3.28084, 2),
                "Cumulative_Distance_mi": round(cum_dist * 0.000621371, 3)
            })

        df = pd.DataFrame(distances)
        st.subheader("📊 Distance Table")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv, file_name="terrain_distances.csv", mime="text/csv")

        # --- Map ---
        st.subheader("🗺️ Map View")
        m = folium.Map(location=[agms[0][1], agms[0][2]], zoom_start=13)
        folium.PolyLine([(lat, lon) for _, lat, lon, _ in agms], color="red").add_to(m)
        MarkerCluster().add_to(m)
        for name, lat, lon, ele in agms:
            folium.Marker(location=(lat, lon), tooltip=name).add_to(m)
        st_folium(m, width=1000, height=500)
