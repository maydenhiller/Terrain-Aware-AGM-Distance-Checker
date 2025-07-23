import streamlit as st
import zipfile
import os
import tempfile
import xml.etree.ElementTree as ET
import simplekml
import math
import pandas as pd
import requests

def get_coords_from_folder(folder, target_name):
    for sub in folder.findall(".//{http://www.opengis.net/kml/2.2}Folder"):
        name = sub.find("{http://www.opengis.net/kml/2.2}name")
        if name is not None and name.text and name.text.strip().lower() == target_name.lower():
            return sub
    return None

def extract_kml_from_kmz(kmz_file):
    with tempfile.TemporaryDirectory() as tmpdirname:
        kmz_path = os.path.join(tmpdirname, "uploaded.kmz")
        with open(kmz_path, "wb") as f:
            f.write(kmz_file.read())

        with zipfile.ZipFile(kmz_path, 'r') as z:
            z.extractall(tmpdirname)
            for file in os.listdir(tmpdirname):
                if file.endswith(".kml"):
                    return os.path.join(tmpdirname, file)
    return None

def parse_coordinates(coord_str):
    coords = []
    for line in coord_str.strip().split():
        parts = line.split(',')
        if len(parts) >= 2:
            lon = float(parts[0])
            lat = float(parts[1])
            coords.append((lat, lon))
    return coords

def get_elevation(lat, lon, api_key):
    url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat},{lon}&key={api_key}"
    r = requests.get(url)
    if r.status_code == 200:
        result = r.json()['results'][0]
        return result['elevation']
    else:
        return 0

def terrain_distance(p1, p2, api_key):
    lat1, lon1 = p1
    lat2, lon2 = p2
    elev1 = get_elevation(lat1, lon1, api_key)
    elev2 = get_elevation(lat2, lon2, api_key)

    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    planar_distance = R * c

    delta_elev = elev2 - elev1
    return math.sqrt(planar_distance**2 + delta_elev**2)

def calculate_distances(centerline_coords, agm_points, api_key):
    output = []
    agm_sorted = sorted(agm_points.items())

    cum_dist = 0
    for i in range(len(agm_sorted)-1):
        label1, p1 = agm_sorted[i]
        label2, p2 = agm_sorted[i+1]

        # Snap to centerline by finding closest segment along red path
        min_dist = float('inf')
        segment_coords = None
        for j in range(len(centerline_coords)-1):
            seg_start = centerline_coords[j]
            seg_end = centerline_coords[j+1]
            dist1 = terrain_distance(p1, seg_start, api_key) + terrain_distance(p2, seg_end, api_key)
            if dist1 < min_dist:
                min_dist = dist1
                segment_coords = (seg_start, seg_end)

        seg_dist = terrain_distance(*segment_coords, api_key)
        cum_dist += seg_dist
        output.append({
            "From": label1,
            "To": label2,
            "Segment (feet)": seg_dist * 3.28084,
            "Segment (miles)": seg_dist * 3.28084 / 5280,
            "Cumulative (miles)": cum_dist * 3.28084 / 5280
        })
    return output

# Streamlit app
st.set_page_config(layout="wide")
st.title("📏 Terrain-Aware Distance Calculator")

st.markdown("Upload a KMZ file with folders named `CENTERLINE` (red path) and `AGMs` (numbered waypoints). Folder names are case-insensitive.")

uploaded_file = st.file_uploader("Upload KMZ file", type="kmz")

if uploaded_file:
    api_key = "AIzaSyB9HxznAvlGb02e-K1rhld_CPeAm_wvPWU"  # Autofilled

    try:
        kml_path = extract_kml_from_kmz(uploaded_file)
        if not kml_path:
            st.error("❌ Error: No KML file found inside KMZ.")
        else:
            tree = ET.parse(kml_path)
            root = tree.getroot()

            center_folder = get_coords_from_folder(root, "CENTERLINE")
            agm_folder = get_coords_from_folder(root, "AGMs")

            if not center_folder or not agm_folder:
                st.error("❌ Error: 'CENTERLINE' or 'AGMs' folder not found.")
            else:
                # Centerline
                red_coords = []
                for placemark in center_folder.findall(".//{http://www.opengis.net/kml/2.2}Placemark"):
                    color = placemark.find(".//{http://www.opengis.net/kml/2.2}color")
                    coord_elem = placemark.find(".//{http://www.opengis.net/kml/2.2}coordinates")
                    if coord_elem is not None:
                        red_coords += parse_coordinates(coord_elem.text)

                # AGMs
                agm_pts = {}
                for placemark in agm_folder.findall(".//{http://www.opengis.net/kml/2.2}Placemark"):
                    name = placemark.find("{http://www.opengis.net/kml/2.2}name")
                    point = placemark.find(".//{http://www.opengis.net/kml/2.2}coordinates")
                    if name is not None and name.text.strip().isdigit() and point is not None:
                        coords = parse_coordinates(point.text)
                        if coords:
                            agm_pts[name.text.strip()] = coords[0]

                if not red_coords:
                    st.error("❌ Error: No red centerline found inside the CENTERLINE folder.")
                elif not agm_pts:
                    st.error("❌ Error: No AGMs found with numeric labels.")
                else:
                    results = calculate_distances(red_coords, agm_pts, api_key)
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download CSV", csv, "terrain_distances.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
