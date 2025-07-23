import streamlit as st
import xml.etree.ElementTree as ET
from geopy.distance import geodesic
import zipfile, io, time, re, requests
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

# --- Page Setup ---
st.set_page_config(page_title="Terrain-Aware Distance Calculator", layout="centered")
st.title("🗺️ Terrain-Aware Distance Calculator")
st.write("Upload a KMZ or KML file with AGM Points and CENTERLINE path. ACCESS and MAP NOTES are ignored.")

# --- Constants ---
KML_NAMESPACE = "{http://www.opengis.net/kml/2.2}"
KM_TO_FEET = 3280.84
KM_TO_MILES = 0.621371
API_KEY = "AIzaSyB9HxznAvlGb02e-K1rhld_CPeAm_wvPWU"  # <-- Replace with your actual key

# --- Coordinate Parser ---
def parse_coordinates(text):
    coords = []
    for pair in re.split(r'\s+', text.strip()):
        try:
            lon, lat, alt = map(float, pair.split(','))
            coords.append((lon, lat, alt))
        except:
            continue
    return coords

# --- KML Parser (Improved) ---
def parse_kml(kml_data):
    centerline, agms = [], []
    try:
        root = ET.fromstring(kml_data)
        placemarks = root.findall(f".//{KML_NAMESPACE}Placemark")
        for placemark in placemarks:
            name_tag = placemark.find(f"{KML_NAMESPACE}name")
            name = name_tag.text.strip() if name_tag is not None else "Unnamed"

            # AGMs: look for Point geometry
            point = placemark.find(f"{KML_NAMESPACE}Point")
            if point is not None:
                coord_text = point.find(f"{KML_NAMESPACE}coordinates")
                if coord_text is not None:
                    coords = parse_coordinates(coord_text.text)
                    if coords:
                        agms.append({"name": name, "coordinates": coords[0]})
                continue

            # CENTERLINE: look for LineString geometry
            line = placemark.find(f"{KML_NAMESPACE}LineString")
            if line is not None:
                coord_text = line.find(f"{KML_NAMESPACE}coordinates")
                if coord_text is not None:
                    coords = parse_coordinates(coord_text.text)
                    centerline.extend(coords)
    except ET.ParseError as e:
        st.error(f"KML parse error: {e}")
    return centerline, agms

# --- Elevation Fetcher ---
@st.cache_data(ttl=3600)
def get_elevations(coords):
    elevations = []
    for i in range(0, len(coords), 100):
        chunk = coords[i:i+100]
        loc_str = "|".join([f"{lat},{lon}" for lon, lat in chunk])
        url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={loc_str}&key={API_KEY}"
        try:
            time.sleep(0.1)
            resp = requests.get(url)
            data = resp.json()
            if data.get("status") == "OK":
                elevations += [r["elevation"] for r in data["results"]]
            else:
                st.warning(f"Elevation API error: {data.get('status')}")
        except:
            st.warning("Elevation request failed.")
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
        distances.append({"name": agm["name"], "dist_km": dist_km})

    # Sorting and building output
    distances.sort(key=lambda d: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', d["name"].lower())])
    output = []
    for i in range(len(distances)-1):
        seg_km = max(0, distances[i+1]["dist_km"] - distances[i]["dist_km"])
        tot_km = max(0, distances[i+1]["dist_km"] - distances[0]["dist_km"])
        output.append({
            "From AGM": distances[i]["name"],
            "To AGM": distances[i+1]["name"],
            "Segment Distance (feet)": f"{seg_km * KM_TO_FEET:.2f}",
            "Segment Distance (miles)": f"{seg_km * KM_TO_MILES:.3f}",
            "Total Distance (feet)": f"{tot_km * KM_TO_FEET:.2f}",
            "Total Distance (miles)": f"{tot_km * KM_TO_MILES:.3f}"
        })
    return output

# --- File Upload Handler ---
file = st.file_uploader("Upload KMZ or KML", type=["kmz", "kml"])
if file:
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
        centerline, agms = parse_kml(kml)
        st.write(f"✅ AGMs found: {len(agms)}, CENTERLINE points: {len(centerline)}")

        if centerline and agms:
            results = calculate_distances(centerline, agms)
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df.set_index("From AGM"))
                st.download_button("📥 Export to CSV", data=df.to_csv(index=False), file_name="agm_distances.csv")
            else:
                st.error("No distances calculated.")
        else:
            st.error("Failed to locate required data in the file.")
else:
    st.info("Awaiting KMZ or KML upload to begin analysis.")
