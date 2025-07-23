import streamlit as st
import xml.etree.ElementTree as ET
from geopy.distance import geodesic
import zipfile, io, time, re, requests
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

# --- Page Configuration ---
st.set_page_config(page_title="Terrain-Aware Distance Calculator", layout="centered")

# --- Constants ---
KML_NAMESPACE = "{http://www.opengis.net/kml/2.2}"
KM_TO_FEET = 3280.84
KM_TO_MILES = 0.621371

# --- Helper: Parse Coordinates ---
def parse_coordinates(coords_text):
    coords = []
    if not coords_text:
        return coords
    coordinate_pairs = re.split(r'\s+', coords_text.strip())
    for pair in coordinate_pairs:
        try:
            parts = pair.split(',')
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
                alt = float(parts[2]) if len(parts) > 2 else 0.0
                coords.append((lon, lat, alt))
        except Exception as e:
            st.warning(f"Skipping invalid coordinate: {pair} - {e}")
    return coords

# --- Helper: KML Element Finder ---
def find_element_by_name(root, target_name):
    for tag in ["Folder", "Placemark"]:
        for element in root.iter(f"{KML_NAMESPACE}{tag}"):
            name = element.find(f"{KML_NAMESPACE}name")
            if name is not None and name.text.strip() == target_name:
                return element
    return None

# --- KML Parser ---
def parse_kml_content(kml_data):
    centerline, agms = [], []
    try:
        root = ET.fromstring(kml_data)
        scope = find_element_by_name(root, "1GOOGLE EARTH SEED FILE V2.0") or root

        cl_container = find_element_by_name(scope, "CENTERLINE")
        if cl_container is not None:
            for placemark in cl_container.iter(f"{KML_NAMESPACE}Placemark"):
                geom = placemark.find(f"{KML_NAMESPACE}LineString")
                if geom is not None:
                    coords = parse_coordinates(geom.find(f"{KML_NAMESPACE}coordinates").text)
                    centerline.extend(coords)

        agm_container = find_element_by_name(scope, "AGMs")
        if agm_container is not None:
            for placemark in agm_container.iter(f"{KML_NAMESPACE}Placemark"):
                geom = placemark.find(f"{KML_NAMESPACE}Point")
                if geom is not None:
                    coords = parse_coordinates(geom.find(f"{KML_NAMESPACE}coordinates").text)
                    name = placemark.find(f"{KML_NAMESPACE}name").text
                    agms.append({"name": name, "coordinates": coords[0]})
    except ET.ParseError as e:
        st.error(f"XML Parsing Error: {e}")
    return centerline, agms

# --- Elevation API ---
@st.cache_data(ttl=3600)
def get_elevations(coords, api_key):
    chunk_size = 100
    elevations = []
    for i in range(0, len(coords), chunk_size):
        chunk = coords[i:i+chunk_size]
        loc_str = "|".join([f"{lat},{lon}" for lon, lat in chunk])
        url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={loc_str}&key={api_key}"
        try:
            time.sleep(0.1)
            resp = requests.get(url)
            data = resp.json()
            if data.get("status") == "OK":
                elevations += [r["elevation"] for r in data["results"]]
            else:
                st.warning(f"Elevation API Error: {data.get('error_message', '')}")
        except:
            st.warning("Error fetching elevation data.")
    return elevations

# --- Distance Calculator ---
def calculate_distances(centerline, agms, api_key):
    if len(centerline) < 2 or len(agms) < 2:
        st.error("Need at least 2 centerline points and 2 AGMs.")
        return []

    # Elevation for centerline
    cl_2d = [(lon, lat) for lon, lat, _ in centerline]
    cl_elevs = get_elevations(cl_2d, api_key)
    centerline_3d = [(lat, lon, cl_elevs[i]) for i, (lon, lat) in enumerate(cl_2d)]

    # Cumulative distances
    cumulative = [0.0]
    for i in range(1, len(centerline_3d)):
        p1, p2 = centerline_3d[i-1], centerline_3d[i]
        d2d = geodesic((p1[0], p1[1]), (p2[0], p2[1])).meters
        d_alt = p2[2] - p1[2]
        d3d = np.sqrt(d2d**2 + d_alt**2)
        cumulative.append(cumulative[-1] + d3d / 1000.0)

    # Elevation for AGMs
    agm_2d = [(a["coordinates"][0], a["coordinates"][1]) for a in agms]
    agm_elevs = get_elevations(agm_2d, api_key)
    for i, agm in enumerate(agms):
        agm["coordinates"] = (agm["coordinates"][0], agm["coordinates"][1], agm_elevs[i])

    # Snap AGMs to centerline
    cl_geom = LineString([(lon, lat) for lon, lat, _ in centerline])
    results = []
    base = None
    distances = []
    for agm in agms:
        lon, lat, alt = agm["coordinates"]
        pt = Point(lon, lat)
        proj = nearest_points(cl_geom, pt)[0]
        frac = cl_geom.project(proj) / cl_geom.length if cl_geom.length > 0 else 0
        dist_km = frac * cumulative[-1]
        distances.append({"name": agm["name"], "dist_km": dist_km})

    # Natural sort
    distances.sort(key=lambda d: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', d["name"].lower())])

    # Build result table
    output = []
    for i in range(len(distances)-1):
        seg_km = distances[i+1]["dist_km"] - distances[i]["dist_km"]
        tot_km = distances[i+1]["dist_km"] - distances[0]["dist_km"]
        output.append({
            "From AGM": distances[i]["name"],
            "To AGM": distances[i+1]["name"],
            "Segment Distance (feet)": f"{seg_km * KM_TO_FEET:.2f}",
            "Segment Distance (miles)": f"{seg_km * KM_TO_MILES:.3f}",
            "Total Distance (feet)": f"{tot_km * KM_TO_FEET:.2f}",
            "Total Distance (miles)": f"{tot_km * KM_TO_MILES:.3f}"
        })
    return output

# --- Streamlit UI ---
st.title("🗺️ Terrain-Aware Distance Calculator")
st.markdown("Upload a KML or KMZ file with AGM points and a CENTERLINE path.")

api_key = st.text_input("Enter your Google Elevation API Key", type="password")
file = st.file_uploader("Upload .kml or .kmz", type=["kml", "kmz"])

if file and api_key:
    ext = file.name.split('.')[-1].lower()
    kml = None
    if ext == "kml":
        kml = file.read().decode("utf-8")
    elif ext == "kmz":
        with zipfile.ZipFile(io.BytesIO(file.read()), 'r') as zf:
            kml_names = [n for n in zf.namelist() if n.endswith(".kml")]
            if kml_names:
                kml = zf.read(kml_names[0]).decode("utf-8")

    if kml:
        centerline, agms = parse_kml_content(kml)
        if centerline and agms:
            results = calculate_distances(centerline, agms, api_key)
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df.set_index("From AGM"))
                st.download_button("📥 Export to CSV", data=df.to_csv(index=False), file_name="agm_distances.csv")
            else:
                st.error("No distances calculated. Check data quality.")
        else:
            st.error("Missing CENTERLINE or AGM points.")
