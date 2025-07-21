import streamlit as st
import zipfile
import os
import tempfile
from fastkml import kml
from shapely.geometry import LineString, Point
import folium
from streamlit_folium import st_folium
import io
import simplekml
import math
import base64

st.set_page_config(page_title="Terrain-Aware AGM Distance Checker", layout="wide")
st.title("📏 Terrain-Aware AGM Distance Checker")

# Helper: safe features getter (handles list vs method)
def safe_features(obj):
    try:
        return list(obj.features())
    except TypeError:
        return list(obj.features) if hasattr(obj, "features") else []

# Helper: parse KMZ or KML
def extract_kml_from_upload(upload):
    try:
        if upload.name.endswith(".kmz"):
            with zipfile.ZipFile(upload) as z:
                for name in z.namelist():
                    if name.endswith(".kml"):
                        return z.read(name)
        else:
            return upload.read()
    except Exception as e:
        st.error(f"❌ Failed to extract KML/KMZ: {e}")
    return None

# Helper: parse KML and extract centerline and AGMs
def parse_kml(kml_data):
    agms = []
    centerline = []

    try:
        k = kml.KML()
        if isinstance(kml_data, bytes):
            k.from_string(kml_data)
        else:
            k.from_string(kml_data.encode("utf-8"))

        for doc in safe_features(k):
            for folder in safe_features(doc):
                for feature in safe_features(folder):
                    if hasattr(feature, "geometry"):
                        if feature.geometry.geom_type == "LineString":
                            # Look for red (#ff0000)
                            if (
                                hasattr(feature, "style_url")
                                and "ff0000" in feature.style_url.lower()
                            ) or (
                                hasattr(feature, "description")
                                and "ff0000" in feature.description.lower()
                            ):
                                centerline = list(feature.geometry.coords)
                        elif feature.geometry.geom_type == "Point":
                            name = feature.name.strip()
                            if name.isnumeric():
                                coord = list(feature.geometry.coords)[0]
                                agms.append((name, coord))
    except Exception as e:
        st.error(f"❌ Failed to parse KML: {e}")
        return [], []

    agms.sort(key=lambda x: int(x[0]))
    return centerline, agms

# Haversine + elevation (3D terrain distance)
def compute_3d_distance(p1, p2):
    from math import radians, cos, sin, sqrt, atan2

    lat1, lon1, ele1 = p1
    lat2, lon2, ele2 = p2

    R = 6371000  # radius of Earth in meters

    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    flat_distance = R * c
    elev_diff = ele2 - ele1
    return sqrt(flat_distance ** 2 + elev_diff ** 2)

# Distance calculation
def compute_distances(centerline, agms):
    results = []
    cumulative = 0

    # Interpolate elevations
    elevations = [0] * len(centerline)
    for i in range(len(centerline)):
        if len(centerline[i]) == 3:
            elevations[i] = centerline[i][2]
        else:
            elevations[i] = 0

    # Get closest point index for each AGM
    def nearest_index(pt):
        return min(
            range(len(centerline)),
            key=lambda i: math.dist([pt[0], pt[1]], [centerline[i][0], centerline[i][1]]),
        )

    agm_indexes = [nearest_index(agm[1]) for agm in agms]

    for i in range(1, len(agm_indexes)):
        start_idx = agm_indexes[i - 1]
        end_idx = agm_indexes[i]

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        segment_distance = 0
        for j in range(start_idx, end_idx):
            p1 = (*centerline[j], elevations[j])
            p2 = (*centerline[j + 1], elevations[j + 1])
            segment_distance += compute_3d_distance(p1, p2)

        cumulative += segment_distance
        results.append(
            {
                "From": agms[i - 1][0],
                "To": agms[i][0],
                "Segment (ft)": round(segment_distance * 3.28084, 2),
                "Cumulative (mi)": round(cumulative * 0.000621371, 4),
            }
        )

    return results

# App UI
uploaded = st.file_uploader("Upload a KMZ or KML file with red centerline and numeric AGMs", type=["kmz", "kml"])

if uploaded:
    raw_kml = extract_kml_from_upload(uploaded)
    if raw_kml:
        centerline, agms = parse_kml(raw_kml)
        if not centerline or not agms:
            st.warning("⚠️ Centerline or AGMs not found. Ensure the line is red (#ff0000) and placemark names are numeric.")
        else:
            distances = compute_distances(centerline, agms)
            st.success("✅ Parsed successfully. Distance table below:")

            st.dataframe(distances)

            # CSV download
            import pandas as pd
            df = pd.DataFrame(distances)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, "terrain_distances.csv", "text/csv")

            # Map preview
            with st.expander("🌍 Show map preview"):
                m = folium.Map(location=agms[0][1][::-1], zoom_start=14)
                folium.PolyLine(locations=[(lat, lon) for lon, lat, *_ in centerline], color="red").add_to(m)
                for name, (lon, lat, *_) in agms:
                    folium.Marker([lat, lon], popup=name).add_to(m)
                st_folium(m, width=800, height=500)
