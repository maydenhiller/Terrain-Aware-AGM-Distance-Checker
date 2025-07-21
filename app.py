import streamlit as st
import zipfile
import io
from fastkml import kml
from shapely.geometry import LineString, Point
import folium
from folium.plugins import MarkerCluster
import base64
import csv
import tempfile
import re

st.set_page_config(page_title="Terrain-Aware Distance Checker", layout="wide")
st.title("📍 Terrain-Aware AGM Distance Checker")
st.markdown("Upload a `.kmz` or `.kml` file with a red centerline (`#ff0000`) and numbered AGMs.")

# Utility: Read KML content from uploaded file
def extract_kml_content(uploaded_file):
    if uploaded_file.name.endswith('.kmz'):
        with zipfile.ZipFile(uploaded_file, 'r') as zf:
            for name in zf.namelist():
                if name.endswith('.kml'):
                    return zf.read(name).decode('utf-8')
    elif uploaded_file.name.endswith('.kml'):
        return uploaded_file.read().decode('utf-8')
    return None

# Parse KML: Find red centerline and numeric AGMs
def parse_kml(kml_data):
    k = kml.KML()
    k.from_string(kml_data)
    ns = '{http://www.opengis.net/kml/2.2}'

    def find_features(obj):
        if hasattr(obj, 'features'):
            for f in obj.features():
                yield from find_features(f)
        else:
            yield obj

    centerline = None
    agms = []

    for f in find_features(k):
        if isinstance(f.geometry, LineString):
            style = getattr(f, 'styleUrl', '') or getattr(f, 'style', None)
            if 'ff0000ff' in str(style).lower() or '#ff0000' in str(style).lower():
                centerline = f.geometry
        elif isinstance(f.geometry, Point):
            name = f.name.strip()
            if name.isdigit():
                agms.append((int(name), f.geometry))

    agms.sort()
    return centerline, agms

# Terrain-aware 3D distance using mock elevation (replace later with real USGS API)
def haversine_3d(p1, p2, elev1, elev2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371000  # radius of Earth in meters
    lat1, lon1 = radians(p1.y), radians(p1.x)
    lat2, lon2 = radians(p2.y), radians(p2.x)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    flat_dist = R * 2 * atan2(sqrt(a), sqrt(1 - a))
    dz = elev2 - elev1
    return sqrt(flat_dist**2 + dz**2)

# Mock elevation function (replace with USGS later)
def get_mock_elevation(lat, lon):
    return 100.0  # flat terrain mock

# Build CSV + Map
def calculate_distances(centerline, agms):
    points = [pt[1] for pt in agms]
    names = [pt[0] for pt in agms]
    data = []
    total = 0

    for i in range(len(points)-1):
        p1, p2 = points[i], points[i+1]
        elev1 = get_mock_elevation(p1.y, p1.x)
        elev2 = get_mock_elevation(p2.y, p2.x)
        dist_m = haversine_3d(p1, p2, elev1, elev2)
        dist_ft = dist_m * 3.28084
        total += dist_ft
        data.append({
            'From': names[i],
            'To': names[i+1],
            'Segment Distance (ft)': round(dist_ft, 2),
            'Cumulative Distance (ft)': round(total, 2),
            'Cumulative Distance (mi)': round(total / 5280, 4)
        })
    return data

def generate_csv(data):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()

def render_map(centerline, agms):
    m = folium.Map(location=[agms[0][1].y, agms[0][1].x], zoom_start=14)
    folium.PolyLine([(pt.y, pt.x) for pt in centerline.coords], color='red', weight=3).add_to(m)
    for label, pt in agms:
        folium.Marker(location=[pt.y, pt.x], popup=str(label)).add_to(m)
    return m

# Upload and process
uploaded_file = st.file_uploader("Upload a KMZ or KML file", type=["kmz", "kml"])
if uploaded_file:
    kml_text = extract_kml_content(uploaded_file)
    if kml_text:
        try:
            centerline, agms = parse_kml(kml_text)
            if not centerline or len(agms) < 2:
                st.error("❌ Could not find a red centerline or at least 2 numeric AGMs.")
            else:
                st.success("✅ Centerline and AGMs loaded.")
                results = calculate_distances(centerline, agms)
                st.dataframe(results)

                csv_data = generate_csv(results)
                st.download_button("📥 Download CSV", data=csv_data, file_name="terrain_distances.csv", mime="text/csv")

                st.subheader("🗺️ Map Preview")
                map_html = render_map(centerline, agms)._repr_html_()
                st.components.v1.html(map_html, height=500, scrolling=True)
        except Exception as e:
            st.error(f"Failed to parse KML: {e}")
    else:
        st.error("Failed to extract KML from the file.")
