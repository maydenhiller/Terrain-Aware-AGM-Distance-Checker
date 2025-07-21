import streamlit as st
from fastkml import kml
from shapely.geometry import LineString, Point
import zipfile
import io
import math
import pandas as pd

st.set_page_config(page_title="Terrain-Aware AGM Distance Checker", layout="centered")
st.title("🗺️ Terrain-Aware AGM Distance Checker")

def get_elevation(lat, lon):
    return 1000  # Replace with actual elevation lookup

def haversine_3d(p1, p2):
    R = 6371000
    lat1, lon1, ele1 = map(math.radians, [p1[0], p1[1], p1[2]])
    lat2, lon2, ele2 = map(math.radians, [p2[0], p2[1], p2[2]])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    horizontal = R * c
    vertical = ele2 - ele1
    return math.sqrt(horizontal**2 + vertical**2)

def extract_kml_from_kmz(file):
    with zipfile.ZipFile(file) as z:
        for name in z.namelist():
            if name.endswith('.kml'):
                return z.read(name)
    return None

def find_folder_by_name(feature_obj, target_name):
    matches = []
    if hasattr(feature_obj, 'name') and feature_obj.name and feature_obj.name.strip().upper() == target_name.upper():
        matches.append(feature_obj)
    if hasattr(feature_obj, 'features'):
        for sub in feature_obj.features():
            matches.extend(find_folder_by_name(sub, target_name))
    return matches

def parse_kml(kml_bytes):
    k = kml.KML()
    k.from_string(kml_bytes)
    top_features = list(k.features())

    centerlines = []
    agms = []

    for f in top_features:
        centerline_folders = find_folder_by_name(f, "CENTERLINE")
        agm_folders = find_folder_by_name(f, "AGMs")

        for folder in centerline_folders:
            for placemark in folder.features():
                if hasattr(placemark, "geometry") and isinstance(placemark.geometry, LineString):
                    centerlines.append(placemark.geometry)

        for folder in agm_folders:
            for placemark in folder.features():
                if hasattr(placemark, "geometry") and isinstance(placemark.geometry, Point):
                    if placemark.name and placemark.name.strip().isdigit():
                        agms.append((int(placemark.name.strip()), placemark.geometry))

    if not centerlines or not agms:
        raise ValueError("Centerline or AGMs not found. They must be under folders named 'CENTERLINE' and 'AGMs'.")

    return centerlines[0], sorted(agms, key=lambda x: x[0])

def snap_to_line(line: LineString, point: Point, num_samples=1000):
    min_dist = float('inf')
    closest_point = None
    for i in range(num_samples + 1):
        frac = i / num_samples
        candidate = line.interpolate(frac, normalized=True)
        dist = point.distance(candidate)
        if dist < min_dist:
            min_dist = dist
            closest_point = candidate
    return closest_point

def calculate_terrain_distances(centerline, agms):
    snapped = [snap_to_line(centerline, pt) for _, pt in agms]
    with_elev = [(pt.y, pt.x, get_elevation(pt.y, pt.x)) for pt in snapped]

    distances = []
    cumulative = 0
    for i in range(1, len(with_elev)):
        d = haversine_3d(with_elev[i - 1], with_elev[i])
        cumulative += d
        distances.append({
            "From": agms[i - 1][0],
            "To": agms[i][0],
            "Segment Distance (ft)": round(d * 3.28084, 2),
            "Cumulative Distance (mi)": round(cumulative * 0.000621371, 3)
        })
    return pd.DataFrame(distances)

uploaded_file = st.file_uploader("📁 Upload a KMZ or KML file", type=["kmz", "kml"])

if uploaded_file:
    try:
        file_data = uploaded_file.read()
        if uploaded_file.name.endswith(".kmz"):
            kml_data = extract_kml_from_kmz(io.BytesIO(file_data))
        else:
            kml_data = file_data

        centerline, agms = parse_kml(kml_data)
        df = calculate_terrain_distances(centerline, agms)

        st.success("✅ Distances calculated successfully!")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Download CSV", csv, "terrain_distances.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Failed to parse KMZ/KML: {e}")
