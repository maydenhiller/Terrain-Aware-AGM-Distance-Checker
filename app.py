import streamlit as st
import zipfile
import os
import tempfile
import xml.etree.ElementTree as ET
import simplekml
from fastkml import kml
import math
import folium
from streamlit_folium import st_folium
import base64
from io import BytesIO

st.set_page_config(page_title="Terrain-Aware AGM Distance Checker", layout="wide")
st.title("🗺️ Terrain-Aware AGM Distance Checker")

# Accept KMZ or KML file
uploaded_file = st.file_uploader("Upload a KMZ or KML file with a red centerline and numbered AGMs", type=["kmz", "kml"])

def extract_kml_from_kmz(kmz_file):
    with zipfile.ZipFile(kmz_file, 'r') as zf:
        for name in zf.namelist():
            if name.endswith('.kml'):
                return zf.read(name)
    return None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2.0)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))

def parse_kml(kml_data):
    agms = []
    centerline = []

    try:
        k = kml.KML()
        k.from_string(kml_data)
        doc = list(k.features())[0]
        folder = list(doc.features())[0]

        for feature in folder.features():
            if isinstance(feature, kml.Placemark):
                if hasattr(feature.geometry, 'coords'):  # Point
                    name = feature.name.strip()
                    if name.isnumeric():
                        coord = list(feature.geometry.coords)[0]
                        agms.append((name, coord))
                elif feature.geometry.geom_type == 'LineString':
                    # Check for red color
                    if '#ff0000' in feature.description.lower():
                        centerline = list(feature.geometry.coords)

    except Exception as e:
        st.error(f"Failed to parse KML: {e}")
        return [], []

    agms.sort(key=lambda x: int(x[0]))
    return centerline, agms

def calculate_distances(centerline, agms):
    distances = []
    cumulative = 0

    for i in range(len(agms)-1):
        name1, coord1 = agms[i]
        name2, coord2 = agms[i+1]

        min_seg_distance = float('inf')

        for j in range(len(centerline)-1):
            c1 = centerline[j]
            c2 = centerline[j+1]
            d1 = haversine(coord1[1], coord1[0], c1[1], c1[0])
            d2 = haversine(coord2[1], coord2[0], c2[1], c2[0])
            total_d = d1 + d2
            if total_d < min_seg_distance:
                min_seg_distance = total_d

        cumulative += min_seg_distance
        distances.append({
            "From": name1,
            "To": name2,
            "Segment Distance (ft)": round(min_seg_distance * 3.28084, 2),
            "Cumulative Distance (mi)": round(cumulative * 0.000621371, 3)
        })

    return distances

def generate_csv(data):
    output = "From,To,Segment Distance (ft),Cumulative Distance (mi)\n"
    for row in data:
        output += f"{row['From']},{row['To']},{row['Segment Distance (ft)']},{row['Cumulative Distance (mi)']}\n"
    return output

# Main logic
if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    if file_ext == ".kmz":
        kml_raw = extract_kml_from_kmz(uploaded_file)
        if not kml_raw:
            st.error("No KML found in the KMZ file.")
        else:
            centerline, agms = parse_kml(kml_raw)
    else:
        centerline, agms = parse_kml(uploaded_file.read())

    if centerline and agms:
        st.success(f"Parsed {len(agms)} AGMs and {len(centerline)} points in centerline.")
        result = calculate_distances(centerline, agms)
        st.dataframe(result)

        csv_data = generate_csv(result)
        st.download_button("📥 Download Results as CSV", csv_data, file_name="terrain_distances.csv", mime="text/csv")

        # Show map
        m = folium.Map(location=[agms[0][1][1], agms[0][1][0]], zoom_start=13)
        folium.PolyLine(locations=[(lat, lon) for lon, lat, *_ in centerline], color="red").add_to(m)
        for name, (lon, lat, *_ ) in agms:
            folium.Marker([lat, lon], tooltip=name).add_to(m)
        st_folium(m, width=700, height=500)
    else:
        st.warning("Centerline or AGMs not found. Ensure the line is red (#ff0000) and placemark names are purely numeric.")
