import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import math
from io import BytesIO
import zipfile
import srtm  # local SRTM lookup

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
FT_PER_M = 3.28084
MI_PER_FT = 1 / 5280

# ── LOAD & CACHE LOCAL SRTM ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_srtm():
    return srtm.get_data()

elev_data = load_srtm()

# ── ELEVATION LOOKUP WITH FALLBACKS ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=10000, ttl=24*3600)
def get_elevation(lat: float, lon: float) -> float:
    # 1) LOCAL SRTM  
    elev = elev_data.get_elevation(lat, lon)
    if elev is not None:
        return elev

    # 2) USGS EPQS (LiDAR / best‐available DEM)
    try:
        r = requests.get(
            "https://nationalmap.gov/epqs/pqs.php",
            params={"x": lon, "y": lat, "units": "Meters", "output": "json"},
            timeout=5
        )
        r.raise_for_status()
        e = r.json()["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]["Elevation"]
        if e is not None:
            return float(e)
    except:
        pass

    # 3) OPEN‐ELEVATION fallback
    try:
        r = requests.get(
            "https://api.open-elevation.com/api/v1/lookup",
            params={"locations": f"{lat:.6f},{lon:.6f}"},
            timeout=5
        )
        r.raise_for_status()
        return float(r.json()["results"][0]["elevation"])
    except:
        return 0.0

# ── HAVERSINE IN METERS ─────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # radius in meters
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ── PARSE KMZ / KML ─────────────────────────────────────────────────────────────
def parse_kml_kmz(file) -> tuple[list[tuple[str,int,float,float]], list[tuple[float,float]]]:
    raw = file.read()
    # unzip if KMZ
    if file.name.lower().endswith(".kmz"):
        with zipfile.ZipFile(BytesIO(raw)) as z:
            for nm in z.namelist():
                if nm.lower().endswith(".kml"):
                    raw = z.read(nm)
                    break

    root = ET.fromstring(raw)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    # extract AGM points
    agms = []
    for fld in root.findall(".//kml:Folder", ns):
        nm = fld.find("kml:name", ns)
        if nm is not None and nm.text.strip().lower() == "agms":
            for pm in fld.findall(".//kml:Placemark", ns):
                label = pm.find("kml:name", ns).text.strip()
                coord = pm.find(".//kml:coordinates", ns).text.strip()
                lon, lat, *_ = coord.split(",")
                agms.append((label, float(lat), float(lon)))

    # extract centerline polyline
    center = []
    for fld in root.findall(".//kml:Folder", ns):
        nm = fld.find("kml:name", ns)
        if nm is not None and nm.text.strip().lower() == "centerline":
            for ls in fld.findall(".//kml:LineString", ns):
                coords = ls.find("kml:coordinates", ns).text.strip()
                for pair in coords.split():
                    lon, lat, *_ = pair.split(",")
                    center.append((float(lat), float(lon)))

    return agms, center

# ── SNAP AGM TO CENTERLINE INDEX ───────────────────────────────────────────────
def closest_index(path: list[tuple[float,float]], pt: tuple[float,float]) -> int:
    best_i, best_d = 0, float("inf")
    lat0, lon0 = pt
    for i, (lat, lon) in enumerate(path):
        d = haversine(lat0, lon0, lat, lon)
        if d < best_d:
            best_i, best_d = i, d
    return best_i

# ── CALCULATE PATH‐WALK DISTANCE BETWEEN TWO CENTERLINE INDICES ───────────────
def path_distance(path: list[tuple[float,float]], i1: int, i2: int) -> float:
    a, b = sorted((i1, i2))
    total = 0.0
    for i in range(a, b):
        lat1, lon1 = path[i]
        lat2, lon2 = path[i+1]
        h = haversine(lat1, lon1, lat2, lon2)
        e1 = get_elevation(lat1, lon1)
        e2 = get_elevation(lat2, lon2)
        total += math.sqrt(h*h + (e2 - e1)**2)
    return total

# ── STREAMLIT UI ───────────────────────────────────────────────────────────────
st.set_page_config("AGM Terrain Distances", layout="wide")
st.title("📏 Terrain-Aware AGM Distances Along Centerline")

up = st.file_uploader("Upload KML or KMZ (must include ‘AGMs’ & ‘Centerline’ folders)",
                     type=["kml","kmz"])
if not up:
    st.info("Please upload your KML or KMZ file.")
    st.stop()

# parse
agms, centerline = parse_kml_kmz(up)
if not agms or not centerline:
    st.error("Could not find an ‘AGMs’ folder or ‘Centerline’ in your file.")
    st.stop()

# order AGMs by station number (last three digits)
agms_sorted = sorted(
    agms,
    key=lambda x: int("".join(filter(str.isdigit, x[0])) or 0)
)

# precompute snap-indices
indices = [
    closest_index(centerline, (lat, lon))
    for _, lat, lon in agms_sorted
]

# compute segment distances
rows = []
total_ft = 0.0
n = len(indices) - 1
prog = st.progress(0)

for i in range(n):
    label1, lat1, lon1 = agms_sorted[i]
    label2, lat2, lon2 = agms_sorted[i+1]

    d_m = path_distance(centerline, indices[i], indices[i+1])
    d_ft = d_m * FT_PER_M
    d_mi = d_ft * MI_PER_FT
    total_ft += d_ft

    # zero-pad numeric part to 3 digits
    st1 = int("".join(filter(str.isdigit, label1)) or 0)
    st2 = int("".join(filter(str.isdigit, label2)) or 0)
    seg_lbl = f"Distance from {st1:03d} to {st2:03d}:"

    rows.append({
        "Segment": seg_lbl,
        "Distance (ft)": f"{d_ft:,.2f}",
        "Distance (mi)": f"{d_mi:.4f}"
    })
    prog.progress((i+1)/n)

# render results & CSV download
df = pd.DataFrame(rows)
st.subheader("AGM Segment Distances")
st.dataframe(df, use_container_width=True)

tot_mi = total_ft * MI_PER_FT
st.markdown(f"**Total:** {total_ft:,.2f} ft  |  **{tot_mi:.4f} mi**")

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV", data=csv, file_name="agm_distances.csv")

