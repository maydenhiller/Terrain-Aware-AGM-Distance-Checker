import streamlit as st
import xml.etree.ElementTree as ET
import zipfile, io, re, time, requests
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from functools import lru_cache

# --- Streamlit Setup ---
st.set_page_config(page_title="🗺️ AGM Distance Debugger", layout="centered")
st.title("🚧 Terrain-Aware AGM Distance Debugger (OpenTopography)")
st.write("Only placemarks in the AGMS folder will be measured. Folders MAP NOTES and ACCESS are ignored.")

# --- Constants ---
KML_NS      = "{http://www.opengis.net/kml/2.2}"
SRTM_TYPE   = "SRTMGL1"
OPTO_API    = "https://portal.opentopography.org/API/globaldem"
OPTO_KEY    = "49a90bbd39265a2efa15a52c00575150"
KM_TO_FEET  = 3280.84
KM_TO_MILES = 0.621371

# --- Coordinate Parser ---
def parse_coordinates(text: str):
    """Parse KML coordinate text into list of (lon, lat, alt)."""
    coords = []
    for trio in re.split(r"\s+", text.strip()):
        try:
            lon, lat, alt = map(float, trio.split(","))
            coords.append((lon, lat, alt))
        except:
            continue
    return coords

# --- KML Parser ---
def parse_kml(kml: str):
    """Extract centerline coordinates and AGM points from KML string."""
    centerline, agms = [], []
    root = ET.fromstring(kml)
    for folder in root.findall(f".//{KML_NS}Folder"):
        name_el = folder.find(f"{KML_NS}name")
        fname = name_el.text.strip().upper() if name_el is not None else ""
        if fname in ("MAP NOTES", "ACCESS"):
            continue

        for pm in folder.findall(f"{KML_NS}Placemark"):
            label_el = pm.find(f"{KML_NS}name")
            label = label_el.text.strip() if label_el is not None else "Unnamed"
            pt = pm.find(f"{KML_NS}Point")
            ln = pm.find(f"{KML_NS}LineString")

            if fname == "AGMS" and pt is not None:
                txt = pt.find(f"{KML_NS}coordinates").text
                pts = parse_coordinates(txt)
                if pts:
                    agms.append({"name": label, "coords": pts[0]})
            elif ln is not None:
                txt = ln.find(f"{KML_NS}coordinates").text
                centerline.extend(parse_coordinates(txt))

    return centerline, agms

# --- OpenTopography Elevation Fetch ---
@lru_cache(maxsize=None)
def fetch_opentopo_elevation(lon: float, lat: float) -> float:
    """
    Query OpenTopography GlobalDEM API for elevation at a single point.
    Returns elevation in meters.
    """
    params = {
        "demtype": SRTM_TYPE,
        "south": lat, "north": lat,
        "west": lon, "east": lon,
        "outputFormat": "JSON",
        "API_Key": OPTO_KEY
    }
    resp = requests.get(OPTO_API, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    # data["data"] → [[lon, lat, elev]]
    elev = data.get("data", [[None, None, 0]])[0][2]
    return float(elev)

def get_opentopo_elevations(coords):
    """
    Given a list of (lon, lat) tuples, return list of elevations (m).
    Uses caching to avoid repeat calls.
    """
    elevs = []
    for lon, lat in coords:
        try:
            z = fetch_opentopo_elevation(lon, lat)
        except Exception as e:
            st.warning(f"OpenTopo error at ({lat:.6f}, {lon:.6f}): {e}")
            z = 0.0
        elevs.append(z)
        time.sleep(0.2)  # throttle
    return elevs

# --- Distance Calculator ---
def calculate_distances(centerline, agms):
    if len(centerline) < 2 or len(agms) < 2:
        st.error("Need at least 2 centerline points and 2 AGMs.")
        return []

    # Prepare 2D line coords
    cl2d = [(lon, lat) for lon, lat, _ in centerline]
    # Fetch elevations for each centerline vertex
    cl_z = get_opentopo_elevations(cl2d)
    cl3d = [(lon, lat, cl_z[i]) for i, (lon, lat) in enumerate(cl2d)]

    # Compute cumulative 3D distance (in kilometers) along centerline
    cum = [0.0]
    for i in range(1, len(cl3d)):
        x0, y0, z0 = cl3d[i - 1]
        x1, y1, z1 = cl3d[i]
        d2d = geodesic((y0, x0), (y1, x1)).meters
        d3d = np.sqrt(d2d**2 + (z1 - z0)**2)
        cum.append(cum[-1] + d3d / 1000.0)

    # Elevations for each AGM
    agm2d = [(a["coords"][0], a["coords"][1]) for a in agms]
    agm_z = get_opentopo_elevations(agm2d)
    for i, a in enumerate(agms):
        lon, lat = a["coords"][:2]
        a["coords"] = (lon, lat, agm_z[i])

    # Project AGMs onto centerline and record distances
    line = LineString(cl2d)
    pts = []
    for a in agms:
        lon, lat, _ = a["coords"]
        proj_pt = nearest_points(line, Point(lon, lat))[0]
        frac = line.project(proj_pt) / line.length if line.length else 0
        dist_km = max(0, frac * cum[-1])
        pts.append({"name": a["name"], "miles": dist_km * KM_TO_MILES})

    # Sort by name (handles numeric sorting)
    pts.sort(key=lambda d: [int(t) if t.isdigit() else t
                             for t in re.split(r"(\d+)", d["name"].lower())])

    # Build segment & total distances
    rows = []
    start = pts[0]["miles"]
    for i in range(len(pts) - 1):
        m0, m1 = pts[i]["miles"], pts[i+1]["miles"]
        seg = m1 - m0
        tot = m1 - start
        rows.append({
            "From AGM": pts[i]["name"],
            "To AGM": pts[i+1]["name"],
            "Segment (mi)": f"{seg:.3f}",
            "Segment (ft)": f"{seg * KM_TO_FEET:.2f}",
            "Total (mi)":   f"{tot:.3f}",
            "Total (ft)":   f"{tot * KM_TO_FEET:.2f}"
        })
    return rows

# --- Streamlit UI ---
uploaded = st.file_uploader("📤 Upload KMZ or KML", type=["kmz", "kml"])
if uploaded:
    raw = None
    ext = uploaded.name.split(".")[-1].lower()

    if ext == "kml":
        raw = uploaded.read().decode("utf-8")
    else:
        zf = zipfile.ZipFile(io.BytesIO(uploaded.read()))
        kmls = [f for f in zf.namelist() if f.endswith(".kml")]
        st.write("📦 KMZ contents:", kmls)
        if kmls:
            raw = zf.read(kmls[0]).decode("utf-8")
        else:
            st.warning("No .kml file found inside KMZ.")

    if raw:
        cl, agms = parse_kml(raw)
        st.write(f"✅ Parsed CENTERLINE points: {len(cl)}")
        st.write(f"✅ Parsed AGMs: {len(agms)}")

        if cl and agms:
            table = calculate_distances(cl, agms)
            if table:
                df = pd.DataFrame(table)
                st.dataframe(df)
                st.download_button(
                    "📥 Download Distances (Feet & Miles)",
                    df.to_csv(index=False),
                    file_name="agm_distances_opentopo.csv"
                )
        else:
            st.warning("Missing valid CENTERLINE or AGM data.")
else:
    st.info("Upload a KMZ or KML file to begin.")
