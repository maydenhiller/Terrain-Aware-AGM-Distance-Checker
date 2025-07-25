import streamlit as st
import xml.etree.ElementTree as ET
from geopy.distance import geodesic
import zipfile, io, time, re, requests
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

# --- Streamlit Setup ---
st.set_page_config(page_title="🗺️ Terrain Distance Debugger", layout="centered")
st.title("🚧 Terrain-Aware Distance Debugger (USGS + Fallback)")
st.write("Measuring only placemarks in the AGMS folder. MAP NOTES and ACCESS are ignored.")

# --- Constants ---
KML_NAMESPACE = "{http://www.opengis.net/kml/2.2}"
KM_TO_MILES = 0.621371
USGS_BASE = "https://nationalmap.gov/epqs/pqs.php"
OPEN_ELEV_BASE = "https://api.open-elevation.com/api/v1/lookup"

# --- Coordinate Parsing ---
def parse_coordinates(text):
    coords = []
    for trio in re.split(r"\s+", text.strip()):
        try:
            lon, lat, alt = map(float, trio.split(","))
            coords.append((lon, lat, alt))
        except:
            continue
    return coords

# --- KML Parsing ---
def parse_kml(kml_data):
    centerline, agms = [], []
    try:
        root = ET.fromstring(kml_data)
        folders = root.findall(f".//{KML_NAMESPACE}Folder")
        for folder in folders:
            name_tag = folder.find(f"{KML_NAMESPACE}name")
            fname = name_tag.text.strip().upper() if name_tag is not None else ""
            if fname in ("MAP NOTES", "ACCESS"):
                continue
            for pm in folder.findall(f"{KML_NAMESPACE}Placemark"):
                label = pm.find(f"{KML_NAMESPACE}name")
                label = label.text.strip() if label is not None else "Unnamed"
                pt = pm.find(f"{KML_NAMESPACE}Point")
                ln = pm.find(f"{KML_NAMESPACE}LineString")
                if fname == "AGMS" and pt is not None:
                    txt = pt.find(f"{KML_NAMESPACE}coordinates").text
                    c = parse_coordinates(txt)
                    if c:
                        agms.append({"name": label, "coords": c[0]})
                elif ln is not None:
                    txt = ln.find(f"{KML_NAMESPACE}coordinates").text
                    centerline.extend(parse_coordinates(txt))
    except Exception as e:
        st.error(f"KML parse error: {e}")
    return centerline, agms

# --- Elevation Fetchers ---
def fetch_usgs_elev(session, lon, lat, retries=3, backoff=0.5):
    params = {"x": lon, "y": lat, "units": "Feet", "output": "json"}
    for i in range(retries):
        try:
            resp = session.get(USGS_BASE, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                eqs = data.get("USGS_Elevation_Point_Query_Service", {})
                elev_q = eqs.get("Elevation_Query", {})
                elev = elev_q.get("Elevation")
                if elev is not None:
                    return elev
                st.warning(f"USGS JSON missing Elevation at ({lat:.6f},{lon:.6f})")
            else:
                st.warning(f"USGS HTTP {resp.status_code} at ({lat:.6f},{lon:.6f})")
        except Exception as e:
            st.warning(f"USGS error at ({lat:.6f},{lon:.6f}): {e}")
        time.sleep(backoff * (i + 1))
    return None

def fetch_open_elev(session, lon, lat):
    params = {"locations": f"{lat},{lon}"}
    try:
        resp = session.get(OPEN_ELEV_BASE, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results and "elevation" in results[0]:
                return results[0]["elevation"]
    except:
        pass
    return None

def get_elevations(coords):
    session = requests.Session()
    elevs = []
    for lon, lat in coords:
        elev = fetch_usgs_elev(session, lon, lat)
        if elev is None:
            elev = fetch_open_elev(session, lon, lat)
            if elev is not None:
                st.info(f"Fallback Open-Elev used at ({lat:.6f},{lon:.6f})")
        if elev is None:
            st.error(f"No elevation for ({lat:.6f},{lon:.6f}); defaulting to 0")
            elev = 0
        elevs.append(elev)
        time.sleep(0.25)
    return elevs

# --- Distance Calculation ---
def calculate_distances(centerline, agms):
    if len(centerline) < 2 or len(agms) < 2:
        st.error("Need at least 2 centerline points and 2 AGMs.")
        return []

    # 2D centerline coords & fetch elevations
    cl2d = [(lon, lat) for lon, lat, _ in centerline]
    cl_elevs = get_elevations(cl2d)
    cl3d = [(lon, lat, cl_elevs[i]) for i, (lon, lat) in enumerate(cl2d)]

    # cumulative 3D distances in km
    cum = [0.0]
    for i in range(1, len(cl3d)):
        p1, p2 = cl3d[i - 1], cl3d[i]
        d2d = geodesic((p1[1], p1[0]), (p2[1], p2[0])).meters
        d_alt = p2[2] - p1[2]
        cum.append(cum[-1] + np.sqrt(d2d**2 + d_alt**2) / 1000.0)

    # AGM 3D coords
    agm2d = [(a["coords"][0], a["coords"][1]) for a in agms]
    agm_elevs = get_elevations(agm2d)
    for i, a in enumerate(agms):
        lon, lat = a["coords"][0], a["coords"][1]
        a["coords"] = (lon, lat, agm_elevs[i])

    # project AGMs onto centerline
    cl_geom = LineString(cl2d)
    distances = []
    for a in agms:
        lon, lat, elev = a["coords"]
        pt = Point(lon, lat)
        proj = nearest_points(cl_geom, pt)[0]
        frac = cl_geom.project(proj) / cl_geom.length if cl_geom.length else 0
        dist_km = max(0, frac * cum[-1])
        distances.append({"name": a["name"], "dist_miles": dist_km * KM_TO_MILES})

    # sort and compute segments
    distances.sort(key=lambda d: [
        int(t) if t.isdigit() else t for t in re.split(r"(\d+)", d["name"].lower())
    ])
    results = []
    for i in range(len(distances) - 1):
        d0, d1 = distances[i], distances[i + 1]
        seg = d1["dist_miles"] - d0["dist_miles"]
        tot = d1["dist_miles"] - distances[0]["dist_miles"]
        results.append({
            "From AGM": d0["name"],
            "To AGM": d1["name"],
            "Segment Distance (miles)": f"{seg:.3f}",
            "Segment Distance (feet)": f"{seg * 5280:.2f}",
            "Total Distance (miles)": f"{tot:.3f}",
            "Total Distance (feet)": f"{tot * 5280:.2f}"
        })
    return results

# --- Streamlit UI ---
uploaded = st.file_uploader("📤 Upload KMZ or KML", type=["kmz", "kml"])
if uploaded:
    ext = uploaded.name.split(".")[-1].lower()
    raw_kml = None

    if ext == "kml":
        raw_kml = uploaded.read().decode("utf-8")
    else:  # kmz
        with zipfile.ZipFile(io.BytesIO(uploaded.read()), "r") as zf:
            kml_files = [f for f in zf.namelist() if f.endswith(".kml")]
            st.write("📦 KMZ contains:", kml_files)
            if kml_files:
                raw_kml = zf.read(kml_files[0]).decode("utf-8")
            else:
                st.warning("No .kml found inside the KMZ.")

    if raw_kml:
        centerline, agms = parse_kml(raw_kml)
        st.write(f"✅ Parsed CENTERLINE points: {len(centerline)}")
        st.write(f"✅ Parsed AGMs: {len(agms)}")

        if centerline and agms:
            table = calculate_distances(centerline, agms)
            if table:
                df = pd.DataFrame(table)
                st.dataframe(df)
                st.download_button(
                    "📥 Download AGM Distances",
                    df.to_csv(index=False),
                    file_name="agm_distances.csv"
                )
        else:
            st.warning("Missing valid centerline or AGM data.")
else:
    st.info("Upload a KMZ or KML file to start measuring.")
