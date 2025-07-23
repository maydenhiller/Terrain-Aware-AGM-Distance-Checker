import streamlit as st
import xml.etree.ElementTree as ET
from geopy.distance import geodesic
import zipfile, io, time, re, requests
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

# --- Page Setup ---
st.set_page_config(page_title="🗺️ Terrain Distance Debugger", layout="centered")
st.title("🚧 Terrain-Aware Distance Debugger")
st.write("Upload a KMZ or KML file with AGMs and CENTERLINE. Folders 'MAP NOTES' and 'ACCESS' will be ignored.")

# --- Constants ---
KML_NAMESPACE = "{http://www.opengis.net/kml/2.2}"
KM_TO_FEET = 3280.84
KM_TO_MILES = 0.621371
API_KEY = "AIzaSyB9HxznAvlGb02e-K1rhld_CPeAm_wvPWU"

# --- Coordinate Parser ---
def parse_coordinates(text):
    coords = []
    for pair in re.split(r'\s+', text.strip()):
        try:
            lon, lat, alt = map(float, pair.split(','))
            coords.append((lon, lat, alt))
        except Exception as e:
            st.warning(f"Skipping malformed coordinate: {pair} — {e}")
    return coords

# --- KML Parser with Folder Filtering ---
def parse_kml(kml_data):
    centerline, agms = [], []
    try:
        root = ET.fromstring(kml_data)

        folders = root.findall(f".//{KML_NAMESPACE}Folder")
        st.write(f"🔎 Found {len(folders)} folders in KML.")

        for folder in folders:
            name_tag = folder.find(f"{KML_NAMESPACE}name")
            folder_name = name_tag.text.strip().upper() if name_tag is not None and name_tag.text else ""
            st.write(f"📂 Folder: {folder_name}")

            if folder_name in ["MAP NOTES", "ACCESS"]:
                st.write(f"⛔ Skipping folder '{folder_name}'")
                continue

            placemarks = folder.findall(f"{KML_NAMESPACE}Placemark")
            for placemark in placemarks:
                name_tag = placemark.find(f"{KML_NAMESPACE}name")
                name = name_tag.text.strip() if name_tag is not None else "Unnamed"

                point = placemark.find(f"{KML_NAMESPACE}Point")
                line = placemark.find(f"{KML_NAMESPACE}LineString")

                if point is not None:
                    coords = parse_coordinates(point.find(f"{KML_NAMESPACE}coordinates").text)
                    if coords:
                        agms.append({"name": name, "coordinates": coords[0]})
                elif line is not None:
                    coords = parse_coordinates(line.find(f"{KML_NAMESPACE}coordinates").text)
                    centerline.extend(coords)

        document = root.find(f"{KML_NAMESPACE}Document")
        if document is not None:
            placemarks = document.findall(f"{KML_NAMESPACE}Placemark")
            for placemark in placemarks:
                name_tag = placemark.find(f"{KML_NAMESPACE}name")
                name = name_tag.text.strip() if name_tag is not None else "Unnamed"

                point = placemark.find(f"{KML_NAMESPACE}Point")
                line = placemark.find(f"{KML_NAMESPACE}LineString")

                if point is not None:
                    coords = parse_coordinates(point.find(f"{KML_NAMESPACE}coordinates").text)
                    if coords:
                        agms.append({"name": name, "coordinates": coords[0]})
                elif line is not None:
                    coords = parse_coordinates(line.find(f"{KML_NAMESPACE}coordinates").text)
                    centerline.extend(coords)

    except ET.ParseError as e:
        st.error(f"KML Parse Error: {e}")
    return centerline, agms

# --- Elevation Fetcher ---
@st.cache_data(ttl=3600)
def get_elevations(coords):
    elevations = []
    for i in range(0, len(coords), 100):
        chunk = coords[i:i+100]
        loc_str = "|".join([f"{lat},{lon}" for lon, lat in chunk])
        url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={loc_str}&key={API_KEY}"
        st.write("🌐 Request URL:", url)
        try:
            time.sleep(0.2)
            resp = requests.get(url)
            data = resp.json()
            status = data.get("status", "UNKNOWN")
            st.write("🔄 API Response Status:", status)
            if status == "OK":
                elevations += [r["elevation"] for r in data["results"]]
            else:
                st.warning(f"Elevation API error: {status}")
        except Exception as e:
            st.error(f"Elevation fetch failed: {e}")
    return elevations

# --- Distance Calculation ---
def calculate_distances(centerline, agms):
    if len(centerline) < 2 or len(agms) < 2:
        st.error("Need at least 2 centerline points and 2 AGMs.")
        return []

    cl_2d = [(lon, lat) for lon, lat, _ in centerline]
    cl_elevs = get_elevations(cl_2d)
    if len(cl_elevs) != len(cl_2d):
        st.error("Elevation fetch failed for centerline.")
        return []

    cl_3d = [(lon, lat, cl_elevs[i]) for i, (lon, lat) in enumerate(cl_2d)]
    cumulative = [0.0]
    for i in range(1, len(cl_3d)):
        p1, p2 = cl_3d[i-1], cl_3d[i]
        d2d = geodesic((p1[1], p1[0]), (p2[1], p2[0])).meters
        d_alt = p2[2] - p1[2]
        d3d = np.sqrt(d2d**2 + d_alt**2)
        cumulative.append(cumulative[-1] + d3d / 1000.0)

    agm_2d = [(a["coordinates"][0], a["coordinates"][1]) for a in agms]
    agm_elevs = get_elevations(agm_2d)
    if len(agm_elevs) != len(agm_2d):
        st.error("Elevation fetch failed for AGMs.")
        return []

    for i, agm in enumerate(agms):
        agm["coordinates"] = (agm["coordinates"][0], agm["coordinates"][1], agm_elevs[i])

    cl_geom = LineString(cl_2d)
    distances = []
    for agm in agms:
        lon, lat, alt = agm["coordinates"]
        pt = Point(lon, lat)
        proj = nearest_points(cl_geom, pt)[0]
        frac = cl_geom.project(proj) / cl_geom.length if cl_geom.length > 0 else 0
        dist_km = max(0, frac * cumulative[-1])
        st.write(f"📍 AGM: {agm['name']} → projected fraction: {frac:.4f} → {dist_km:.2f} km")
        distances.append({"name": agm["name"], "dist_km": dist_km})

    distances.sort(key=lambda d: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', d["name"].lower())])
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

# --- Upload and Analyze ---
file = st.file_uploader("📤 Upload KMZ or KML", type=["kmz", "kml"])
if file:
    ext = file.name.split('.')[-1].lower()
    kml = None
    if ext == "kml":
        kml = file.read().decode("utf-8")
    elif ext == "kmz":
        with zipfile.ZipFile(io.BytesIO(file.read()), 'r') as zf:
            kml_files = [n for n in zf.namelist() if n.endswith(".kml")]
            st.write("📦 KMZ contents:", kml_files)
            if kml_files:
                kml = zf.read(kml_files[0]).
