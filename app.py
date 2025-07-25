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
st.title("🚧 Terrain-Aware Distance Debugger (USGS v1 + Fallback)")
st.write("Measuring only placemarks in the AGMS folder. MAP NOTES and ACCESS are ignored.")

# --- Constants ---
KML_NAMESPACE   = "{http://www.opengis.net/kml/2.2}"
KM_TO_MILES     = 0.621371
USGS_ENDPOINT   = "https://epqs.nationalmap.gov/v1/json"
OPEN_ELEV_BASE  = "https://api.open-elevation.com/api/v1/lookup"

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
        root    = ET.fromstring(kml_data)
        folders = root.findall(f".//{KML_NAMESPACE}Folder")
        for folder in folders:
            name_tag = folder.find(f"{KML_NAMESPACE}name")
            fname    = name_tag.text.strip().upper() if name_tag is not None else ""
            if fname in ("MAP NOTES", "ACCESS"):
                continue

            for pm in folder.findall(f"{KML_NAMESPACE}Placemark"):
                label = pm.find(f"{KML_NAMESPACE}name")
                label = label.text.strip() if label is not None else "Unnamed"
                pt = pm.find(f"{KML_NAMESPACE}Point")
                ln = pm.find(f"{KML_NAMESPACE}LineString")

                if fname == "AGMS" and pt is not None:
                    c = parse_coordinates(pt.find(f"{KML_NAMESPACE}coordinates").text)
                    if c:
                        agms.append({"name": label, "coords": c[0]})
                elif ln is not None:
                    centerline.extend(parse_coordinates(ln.find(f"{KML_NAMESPACE}coordinates").text))
    except Exception as e:
        st.error(f"KML parse error: {e}")
    return centerline, agms

# --- Elevation Fetchers ---
def fetch_usgs_elev(sess, lon, lat, retries=3, backoff=0.5):
    params = {"x": lon, "y": lat, "units": "Feet"}
    for i in range(retries):
        try:
            resp = sess.get(USGS_ENDPOINT, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                elev = (data
                        .get("USGS_Elevation_Point_Query_Service", {})
                        .get("Elevation_Query", {})
                        .get("Elevation"))
                if elev is not None:
                    return elev
                st.warning(f"USGS JSON missing Elevation at ({lat:.6f},{lon:.6f})")
            else:
                st.warning(f"USGS HTTP {resp.status_code} at ({lat:.6f},{lon:.6f})")
        except Exception as e:
            st.warning(f"USGS error at ({lat:.6f},{lon:.6f}): {e}")
        time.sleep(backoff * (i + 1))
    return None

def fetch_open_elev(sess, lon, lat):
    params = {"locations": f"{lat},{lon}"}
    try:
        resp = sess.get(OPEN_ELEV_BASE, params=params, timeout=5)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results and "elevation" in results[0]:
                return results[0]["elevation"]
    except:
        pass
    return None

def get_elevations(coords):
    sess, elevs = requests.Session(), []
    for lon, lat in coords:
        elev = fetch_usgs_elev(sess, lon, lat)
        if elev is None:
            elev = fetch_open_elev(sess, lon, lat)
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

    # Centerline elevations
    cl2d    = [(lon, lat) for lon, lat, _ in centerline]
    cl_elev = get_elevations(cl2d)
    cl3d    = [(lon, lat, cl_elev[i]) for i, (lon, lat) in enumerate(cl2d)]

    # Cumulative 3D distance (km)
    cum = [0.0]
    for i in range(1, len(cl3d)):
        p1, p2 = cl3d[i-1], cl3d[i]
        d2d     = geodesic((p1[1], p1[0]), (p2[1], p2[0])).meters
        d_alt   = p2[2] - p1[2]
        cum.append(cum[-1] + np.sqrt(d2d**2 + d_alt**2) / 1000.0)

    # AGM elevations & projection
    agm2d      = [(a["coords"][0], a["coords"][1]) for a in agms]
    agm_elev   = get_elevations(agm2d)
    for i, a in enumerate(agms):
        lon, lat = a["coords"][:2]
        a["coords"] = (lon, lat, agm_elev[i])

    cl_geom = LineString(cl2d)
    pts     = []
    for a in agms:
        lon, lat, _ = a["coords"]
        proj       = nearest_points(cl_geom, Point(lon, lat))[0]
        frac       = cl_geom.project(proj) / cl_geom.length if cl_geom.length else 0
        dist_km    = max(0, frac * cum[-1])
        pts.append({"name": a["name"], "miles": dist_km * KM_TO_MILES})

    # Sort & build table
    pts.sort(key=lambda x: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", x["name"].lower())])
    rows = []
    start = pts[0]["miles"]
    for i in range(len(pts)-1):
        m0, m1 = pts[i]["miles"], pts[i+1]["miles"]
        seg, tot = m1-m0, m1-start
        rows.append({
            "From AGM": pts[i]["name"],
            "To AGM": pts[i+1]["name"],
            "Segment (mi)": f"{seg:.3f}",
            "Segment (ft)": f"{seg*5280:.2f}",
            "Total (mi)":   f"{tot:.3f}",
            "Total (ft)":   f"{tot*5280:.2f}"
        })
    return rows

# --- UI Wiring ---
uploaded = st.file_uploader("📤 Upload KMZ or KML", type=["kmz","kml"])
if uploaded:
    raw_kml = None
    ext     = uploaded.name.split(".")[-1].lower()

    if ext == "kml":
        raw_kml = uploaded.read().decode("utf-8")
    else:
        with zipfile.ZipFile(io.BytesIO(uploaded.read())) as zf:
            kmls = [f for f in zf.namelist() if f.endswith(".kml")]
            st.write("📦 KMZ contents:", kmls)
            if kmls:
                raw_kml = zf.read(kmls[0]).decode("utf-8")
            else:
                st.warning("No .kml inside KMZ.")

    if raw_kml:
        cl, agms = parse_kml(raw_kml)
        st.write(f"✅ Parsed CENTERLINE points: {len(cl)}")
        st.write(f"✅ Parsed AGMs: {len(agms)}")

        if cl and agms:
            table = calculate_distances(cl, agms)
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

