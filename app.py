import streamlit as st
import zipfile
import os
import tempfile
import xml.etree.ElementTree as ET
import simplekml
import math
import pandas as pd
import requests
from io import BytesIO

# Auto-inject your API key
GOOGLE_ELEVATION_API_KEY = "AIzaSyB9HxznAvlGb02e-K1rhld_CPeAm_wvPWU"

st.set_page_config(layout="wide")
st.title("📏 Terrain-Aware Distance Calculator")

# Helper: Convert lat/lon/elev to 3D distance
def distance_3d(p1, p2):
    lat1, lon1, ele1 = p1
    lat2, lon2, ele2 = p2
    R = 6371000  # Radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    horizontal = R * c
    vertical = ele2 - ele1
    return math.sqrt(horizontal ** 2 + vertical ** 2)

# Helper: Query Google Elevation API
def get_elevations(coords):
    url = (
        "https://maps.googleapis.com/maps/api/elevation/json"
        f"?locations={'|'.join([f'{lat},{lon}' for lat, lon in coords])}"
        f"&key={GOOGLE_ELEVATION_API_KEY}"
    )
    r = requests.get(url)
    results = r.json().get("results", [])
    return [pt["elevation"] for pt in results] if results else [0] * len(coords)

# Extract uploaded KMZ or KML file
def extract_kml(uploaded_file):
    content = uploaded_file.getvalue()
    with tempfile.TemporaryDirectory() as tmpdir:
        if uploaded_file.name.endswith(".kmz"):
            with zipfile.ZipFile(BytesIO(content), 'r') as kmz:
                kmz.extractall(tmpdir)
                for f in kmz.namelist():
                    if f.endswith(".kml"):
                        return os.path.join(tmpdir, f)
        else:
            path = os.path.join(tmpdir, "temp.kml")
            with open(path, "wb") as f:
                f.write(content)
            return path
    return None

# Parse KML for placemarks and line coordinates
def parse_kml(kml_path):
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    def find_folder(root, name):
        for folder in root.findall(".//kml:Folder", ns):
            fname = folder.find("kml:name", ns)
            if fname is not None and fname.text.strip().upper() == name:
                return folder
        return None

    centerline_folder = find_folder(root, "CENTERLINE")
    agms_folder = find_folder(root, "AGMs")

    if centerline_folder is None or agms_folder is None:
        raise ValueError("CENTERLINE or AGMs folder not found.")

    # Get red line path from CENTERLINE
    for pm in centerline_folder.findall(".//kml:Placemark", ns):
        coords = pm.find(".//kml:coordinates", ns)
        if coords is not None:
            points = [
                tuple(map(float, coord.strip().split(",")[:2]))
                for coord in coords.text.strip().split()
            ]
            if len(points) > 1:
                centerline = points
                break
    else:
        raise ValueError("No red centerline found inside the CENTERLINE folder.")

    # Get AGMs (numeric placemarks only)
    agms = []
    for pm in agms_folder.findall(".//kml:Placemark", ns):
        name_el = pm.find("kml:name", ns)
        coords_el = pm.find(".//kml:coordinates", ns)
        if name_el is not None and coords_el is not None:
            name = name_el.text.strip()
            if name.isnumeric():
                lon, lat = map(float, coords_el.text.strip().split(",")[:2])
                agms.append((name, lat, lon))

    agms.sort(key=lambda x: int(x[0]))
    return centerline, agms

# Project a point onto a line segment
def project_point_to_segment(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p

    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return a

    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    proj = (ax + t * dx, ay + t * dy)
    return proj

# Find projected AGM locations on centerline
def snap_agms_to_centerline(centerline, agms):
    snapped = []
    for name, lat, lon in agms:
        min_dist = float('inf')
        closest_point = None
        for i in range(len(centerline) - 1):
            a = centerline[i]
            b = centerline[i + 1]
            proj = project_point_to_segment((lon, lat), a, b)
            dist = math.hypot(proj[0] - lon, proj[1] - lat)
            if dist < min_dist:
                min_dist = dist
                closest_point = (proj[1], proj[0])  # back to (lat, lon)
        snapped.append((name, *closest_point))
    return snapped

# Compute distances
def compute_distances(centerline, snapped_agms):
    coords_with_elev = []
    for i in range(len(centerline)):
        coords_with_elev.append((centerline[i][1], centerline[i][0]))  # lat, lon
    elevations = get_elevations(coords_with_elev)
    centerline_3d = [
        (lat, lon, elev) for (lat, lon), elev in zip(coords_with_elev, elevations)
    ]

    # Interpolate distances along centerline between AGMs
    results = []
    cum_distance = 0
    for i in range(len(snapped_agms) - 1):
        name1, lat1, lon1 = snapped_agms[i]
        name2, lat2, lon2 = snapped_agms[i + 1]

        # Traverse centerline to measure 3D segment
        segment_dist = 0
        start_found = False
        for j in range(len(centerline_3d) - 1):
            lat_a, lon_a, ele_a = centerline_3d[j]
            lat_b, lon_b, ele_b = centerline_3d[j + 1]

            if not start_found:
                if math.isclose(lat_a, lat1, abs_tol=1e-5) and math.isclose(lon_a, lon1, abs_tol=1e-5):
                    start_found = True
                else:
                    continue

            segment_dist += distance_3d((lat_a, lon_a, ele_a), (lat_b, lon_b, ele_b))

            if math.isclose(lat_b, lat2, abs_tol=1e-5) and math.isclose(lon_b, lon2, abs_tol=1e-5):
                break

        segment_miles = segment_dist / 1609.344
        cum_distance += segment_dist
        cum_miles = cum_distance / 1609.344

        results.append({
            "From": name1,
            "To": name2,
            "Segment Distance (ft)": round(segment_dist * 3.28084, 2),
            "Segment Distance (mi)": round(segment_miles, 4),
            "Cumulative Distance (mi)": round(cum_miles, 4)
        })

    return pd.DataFrame(results)

# Upload UI
uploaded_file = st.file_uploader("📂 Upload KMZ or KML file", type=["kmz", "kml"])
if uploaded_file:
    try:
        kml_path = extract_kml(uploaded_file)
        centerline, agms = parse_kml(kml_path)
        snapped = snap_agms_to_centerline(centerline, agms)
        df = compute_distances(centerline, snapped)
        st.success("✅ Distances calculated successfully!")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "terrain_distances.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Error: {e}")
