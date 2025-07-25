import streamlit as st
import xml.etree.ElementTree as ET
from geopy.distance import geodesic
import zipfile, io, time, re, requests
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

# --- Setup ---
st.set_page_config(page_title="🗺️ Terrain Distance Debugger", layout="centered")
st.title("🚧 Terrain-Aware Distance Debugger (USGS)")
st.write("Only placemarks from the AGMS folder will be measured. MAP NOTES and ACCESS folders are ignored.")

# --- Constants ---
KML_NAMESPACE = "{http://www.opengis.net/kml/2.2}"
KM_TO_FEET = 3280.84
KM_TO_MILES = 0.621371

# --- Coordinate Parser ---
def parse_coordinates(text):
    coords = []
    for pair in re.split(r"\s+", text.strip()):
        try:
            lon, lat, alt = map(float, pair.split(","))
            coords.append((lon, lat, alt))
        except:
            pass
    return coords

# --- KML Parser ---
def parse_kml(kml_data):
    centerline, agms = [], []
    try:
        root = ET.fromstring(kml_data)
        folders = root.findall(f".//{KML_NAMESPACE}Folder")

        for folder in folders:
            name_tag = folder.find(f"{KML_NAMESPACE}name")
            folder_name = name_tag.text.strip().upper() if name_tag is not None else ""

            if folder_name in ["MAP NOTES", "ACCESS"]:
                continue

            placemarks = folder.findall(f"{KML_NAMESPACE}Placemark")
            for pm in placemarks:
                name = pm.find(f"{KML_NAMESPACE}name")
                label = name.text.strip() if name is not None else "Unnamed"
                point = pm.find(f"{KML_NAMESPACE}Point")
                line = pm.find(f"{KML_NAMESPACE}LineString")

                if folder_name == "AGMS" and point is not None:
                    coords = parse_coordinates(point.find(f"{KML_NAMESPACE}coordinates").text)
                    if coords:
                        agms.append({"name": label, "coordinates": coords[0]})
                elif line is not None:
                    coords = parse_coordinates(line.find(f"{KML_NAMESPACE}coordinates").text)
                    centerline.extend(coords)
    except Exception as e:
        st.error(f"KML Parse Error: {e}")
    return centerline, agms

# --- USGS Elevation Fetcher with Error Handling and Delay ---
def get_usgs_elevations(coords):
    elevations = []
    for lon, lat in coords:
        url = f"https://nationalmap.gov/epqs/pqs.php?x={lon}&y={lat}&units=Feet&output=json"
        try:
            resp = requests.get(url)
            if resp.status_code == 200 and resp.text.strip():
                data = resp.json()
                elev = data["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]["Elevation"]
                elevations.append(elev)
            else:
                st.warning(f"USGS returned no data for ({lat}, {lon})")
                elevations.append(0)
        except Exception as e:
            st.warning(f"USGS elevation failed for ({lat}, {lon}): {e}")
            elevations.append(0)
        time.sleep(0.25)
    return elevations

# --- Distance Calculator ---
def calculate_distances(centerline, agms):
    if len(centerline) < 2 or len(agms) < 2:
        st.error("Need at least 2 centerline points and 2 AGMs.")
        return []

    cl_2d = [(lon, lat) for lon, lat, _ in centerline]
    cl_elevs = get_usgs_elevations(cl_2d)
    if not cl_elevs or len(cl_elevs) != len(cl_2d):
        st.error("USGS elevation lookup failed.")
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
    agm_elevs = get_usgs_elevations(agm_2d)
    for i, agm in enumerate(agms):
        agm["coordinates"] = (agm["coordinates"][0], agm["coordinates"][1], agm_elevs[i])

    cl_geom = LineString(cl_2d)
    distances = []
    for agm in agms:
        name = agm["name"]
        lon, lat, alt = agm["coordinates"]
        pt = Point(lon, lat)
        proj = nearest_points(cl_geom, pt)[0]
        frac = cl_geom.project(proj) / cl_geom.length if cl_geom.length > 0 else 0
        dist_km = max(0, frac * cumulative[-1])
        dist_miles = dist_km * KM_TO_MILES
        distances.append({"name": name, "dist_miles": dist_miles})

    distances.sort(key=lambda d: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', d["name"].lower())])
    output = []
    for i in range(len(distances)-1):
        seg_miles = distances[i+1]["dist_miles"] - distances[i]["dist_miles"]
        tot_miles = distances[i+1]["dist_miles"] - distances[0]["dist_miles"]
        output.append({
            "From AGM": distances[i]["name"],
            "To AGM": distances[i+1]["name"],
            "Segment Distance (feet)": f"{seg_miles * 5280:.2f}",
            "Segment Distance (miles)": f"{seg_miles:.3f}",
            "Total Distance (feet)": f"{tot_miles * 5280:.2f}",
            "Total Distance (miles)": f"{tot_miles:.3f}"
        })
    return output

# --- Streamlit UI ---
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
                kml = zf.read(kml_files[0]).decode("utf-8")
            else:
                st.warning("❌ No .kml file found inside KMZ archive.")

    if kml:
        centerline, agms = parse_kml(kml)
        st.write(f"✅ Parsed CENTERLINE points: {len(centerline)}")
        st.write(f"✅ Parsed AGMs from 'AGMS' folder: {len(agms)}")

        if centerline and agms:
            results = calculate_distances(centerline, agms)
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df)
                st.download_button(
                    "📥 Download AGM Distances (Feet & Miles)",
                    df.to_csv(index=False),
                    file_name="agm_distances_usgs.csv"
                )
        else:
            st.warning("❌ Missing valid AGMs or CENTERLINE data.")
else:
    st.info("👀 Upload a KMZ or KML file to begin.")
