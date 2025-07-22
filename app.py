import streamlit as st
import zipfile
import os
import tempfile
import simplekml
import xml.etree.ElementTree as ET
import pandas as pd
import math
import requests
from io import BytesIO
from shapely.geometry import LineString, Point

# === Restore built-in list if overwritten ===
if isinstance(__builtins__, dict):
    list = __builtins__["list"]
else:
    list = __builtins__.list

st.set_page_config(layout="wide")
st.title("📏 Terrain-Aware AGM Distance Checker")

# === Google Elevation API Key ===
API_KEY = "AIzaSyB9HxznAvlGb02e-K1rhld_CPeAm_wvPWU"  # Autofilled key

uploaded_kmz = st.file_uploader("Upload KML or KMZ", type=["kmz", "kml"])

# === Helper: Extract KMZ ===
def extract_kmz(kmz_file):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(kmz_file, 'r') as z:
        z.extractall(temp_dir)
    return temp_dir

# === Helper: Parse KML Placemarks and LineStrings ===
def parse_kml(kml_path):
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    placemarks = []
    lines = []

    for pm in root.findall('.//kml:Placemark', ns):
        name_elem = pm.find('kml:name', ns)
        name = name_elem.text if name_elem is not None else ""
        coords_elem = pm.find('.//kml:coordinates', ns)
        if coords_elem is None:
            continue
        coords_text = coords_elem.text.strip()
        coords = [list(map(float, c.split(","))) for c in coords_text.split()]

        if len(coords) == 1 and name.isdigit():
            placemarks.append((name, coords[0]))
        elif len(coords) > 1:
            lines.append((name, coords))

    return placemarks, lines

# === Helper: Interpolate Elevation-aware Distance ===
def get_terrain_distance(path_coords):
    total = 0
    for i in range(1, len(path_coords)):
        p1, p2 = path_coords[i-1], path_coords[i]
        seg = [p1, p2]

        # Elevation sampling (use intermediate point)
        latlngs = [(p[1], p[0]) for p in seg]
        locations = '|'.join([f"{lat},{lon}" for lat, lon in latlngs])
        url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={locations}&key={API_KEY}"
        try:
            response = requests.get(url).json()
            elevations = [r['elevation'] for r in response['results']]
        except Exception:
            elevations = [0, 0]

        dx = 111139 * (p2[1] - p1[1]) * math.cos(math.radians((p1[1] + p2[1]) / 2))
        dy = 111139 * (p2[0] - p1[0])
        dz = elevations[1] - elevations[0]
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        total += dist
    return total

# === Main Logic ===
if uploaded_kmz:
    try:
        if uploaded_kmz.name.endswith(".kmz"):
            temp_dir = extract_kmz(uploaded_kmz)
            kml_paths = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".kml")]
        else:
            temp_dir = tempfile.mkdtemp()
            kml_path = os.path.join(temp_dir, uploaded_kmz.name)
            with open(kml_path, "wb") as f:
                f.write(uploaded_kmz.read())
            kml_paths = [kml_path]

        all_placemarks = []
        red_line_coords = []

        for kml_path in kml_paths:
            placemarks, lines = parse_kml(kml_path)
            all_placemarks.extend(placemarks)
            for name, coords in lines:
                if 'red' in name.lower():
                    red_line_coords = coords

        if not all_placemarks or not red_line_coords:
            st.error("Error: CENTERLINE or AGMs folder not found.")
        else:
            all_placemarks.sort(key=lambda x: int(x[0]))
            centerline = LineString([(lon, lat) for lon, lat, *_ in red_line_coords])

            results = []
            cum_dist = 0

            for i in range(len(all_placemarks) - 1):
                agm1 = Point(all_placemarks[i][1][0], all_placemarks[i][1][1])
                agm2 = Point(all_placemarks[i+1][1][0], all_placemarks[i+1][1][1])
                proj1 = centerline.project(agm1)
                proj2 = centerline.project(agm2)

                if proj2 < proj1:
                    proj1, proj2 = proj2, proj1

                segment = centerline.interpolate(proj1), centerline.interpolate(proj2)
                segment_line = centerline.segmentize(5)
                cut_coords = [pt.coords[0] for pt in segment_line if proj1 <= centerline.project(Point(pt.coords[0])) <= proj2]
                if not cut_coords:
                    cut_coords = [segment[0].coords[0], segment[1].coords[0]]

                seg_dist = get_terrain_distance(cut_coords)
                cum_dist += seg_dist
                results.append({
                    'From': all_placemarks[i][0],
                    'To': all_placemarks[i+1][0],
                    'Segment Distance (ft)': round(seg_dist * 3.28084, 2),
                    'Segment Distance (mi)': round(seg_dist * 3.28084 / 5280, 4),
                    'Cumulative Distance (ft)': round(cum_dist * 3.28084, 2),
                    'Cumulative Distance (mi)': round(cum_dist * 3.28084 / 5280, 4),
                })

            df = pd.DataFrame(results)
            st.success("✅ Distances computed successfully!")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "terrain_aware_distances.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Failed to parse KMZ/KML: {e}")
