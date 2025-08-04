import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import math
import re
import zipfile
from io import BytesIO
import srtm  # pip install srtm.py

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
FT_PER_M   = 3.28084
MI_PER_FT  = 1 / 5280

# ─── LOAD & CACHE LOCAL SRTM ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_srtm_data():
    return srtm.get_data()

elev_data = load_srtm_data()

# ─── HELPER: PARSE STATION LABELS ──────────────────────────────────────────────
def parse_station(label: str) -> tuple[int, str]:
    m = re.match(r"^(\d+)([A-Za-z]*)$", label)
    if not m:
        return 0, ""
    return int(m.group(1)), m.group(2)

# ─── ELEVATION LOOKUP w/ FALLBACKS ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=5000, ttl=24*3600)
def get_elevation(lat: float, lon: float) -> float:
    elev = elev_data.get_elevation(lat, lon)
    if elev is not None:
        return elev

    # USGS EPQS
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

    # Open-Elevation
    try:
        r2 = requests.get(
            "https://api.open-elevation.com/api/v1/lookup",
            params={"locations": f"{lat:.6f},{lon:.6f}"},
            timeout=5
        )
        r2.raise_for_status()
        return float(r2.json()["results"][0]["elevation"])
    except:
        return 0.0

# ─── HAVERSINE DISTANCE (METERS) ───────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    φ1, φ2 = map(math.radians, (lat1, lat2))
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ─── PARSE .KML / .KMZ FOR AGMs + CENTERLINE ────────────────────────────────
def parse_kml_kmz(file_bytes) -> tuple[list[tuple[str,float,float]], list[tuple[float,float]]]:
    raw = file_bytes.read()

    if file_bytes.name.lower().endswith(".kmz"):
        with zipfile.ZipFile(BytesIO(raw)) as z:
            for nm in z.namelist():
                if nm.lower().endswith(".kml"):
                    raw = z.read(nm)
                    break

    root = ET.fromstring(raw)
    ns   = {"kml": "http://www.opengis.net/kml/2.2"}

    agms = []
    for fld in root.findall(".//kml:Folder", ns):
        name = fld.find("kml:name", ns)
        if name is not None and name.text.strip().lower() == "agms":
            for pm in fld.findall(".//kml:Placemark", ns):
                label = pm.find("kml:name", ns).text.strip()
                coord = pm.find(".//kml:coordinates", ns).text.strip()
                lon, lat, *_ = coord.split(",")
                agms.append((label, float(lat), float(lon)))

    center = []
    for fld in root.findall(".//kml:Folder", ns):
        name = fld.find("kml:name", ns)
        if name is not None and name.text.strip().lower() == "centerline":
            for ls in fld.findall(".//kml:LineString", ns):
                for pair in ls.find("kml:coordinates", ns).text.strip().split():
                    lon, lat, *_ = pair.split(",")
                    center.append((float(lat), float(lon)))

    return agms, center

# ─── SNAP TO UNIQUE CENTERLINE INDEX ─────────────────────────────────────────
def snap_unique_indices(centerline, agm_points):
    used = set()
    snaps = []
    for label, lat, lon in agm_points:
        dists = [(i, haversine(lat, lon, vlat, vlon))
                 for i, (vlat, vlon) in enumerate(centerline)]
        for idx, _ in sorted(dists, key=lambda x: x[1]):
            if idx not in used:
                used.add(idx)
                snaps.append(idx)
                break
    return snaps

# ─── COMPUTE 3D PATH‐WALK DISTANCE ─────────────────────────────────────────────
def path_distance(centerline, i1, i2):
    a, b = sorted((i1, i2))
    total = 0.0
    for i in range(a, b):
        lat1, lon1 = centerline[i]
        lat2, lon2 = centerline[i + 1]
        h = haversine(lat1, lon1, lat2, lon2)
        e1 = get_elevation(lat1, lon1)
        e2 = get_elevation(lat2, lon2)
        total += math.sqrt(h*h + (e2 - e1)**2)
    return total

# ─── STREAMLIT UI ─────────────────────────────────────────────────────────────
st.set_page_config("AGM Terrain Distances", layout="wide")
st.title("📏 Terrain-Aware AGM Distances Along Centerline")

uploaded = st.file_uploader(
    "Upload a .kml or .kmz containing ‘AGMs’ & ‘Centerline’ folders",
    type=["kml","kmz"]
)
if not uploaded:
    st.info("Please upload your KML or KMZ file.")
    st.stop()

agms, centerline = parse_kml_kmz(uploaded)
if not agms or not centerline:
    st.error("Could not find ‘AGMs’ or ‘Centerline’ in the file.")
    st.stop()

# Sort AGMs by station label (numeric then alpha)
agms_sorted = sorted(agms, key=lambda x: parse_station(x[0]))

# Snap AGMs to unique centerline vertices
indices = snap_unique_indices(centerline, agms_sorted)

# ─── DEBUG MODE ───────────────────────────────────────────────────────────────
debug = st.checkbox("🔍 Enable debug mode")

if debug:
    st.subheader("1. Centerline Vertices")
    st.write(f"Total vertices parsed: {len(centerline)}")
    st.dataframe(
        pd.DataFrame(centerline[:10], columns=["lat", "lon"]),
        use_container_width=True
    )

    st.subheader("2. AGM → Centerline Snap Mapping")
    snap_map = []
    for (label, alat, alon), idx in zip(agms_sorted, indices):
        vlat, vlon = centerline[idx]
        snap_map.append({
            "AGM": label,
            "AGM lat": alat,
            "AGM lon": alon,
            "snapped_idx": idx,
            "vertex lat": vlat,
            "vertex lon": vlon
        })
    st.dataframe(pd.DataFrame(snap_map), use_container_width=True)

    st.subheader("3. Flat (2D) Haversine Between AGMs")
    straight = []
    for i in range(len(agms_sorted) - 1):
        l1, la1, lo1 = agms_sorted[i]
        l2, la2, lo2 = agms_sorted[i+1]
        d2 = haversine(la1, lo1, la2, lo2) * FT_PER_M
        straight.append({
            "Segment": f"{l1} → {l2}",
            "flat_dist_ft": f"{d2:,.2f}"
        })
    st.dataframe(pd.DataFrame(straight), use_container_width=True)

# ─── DISTANCE COMPUTATION ────────────────────────────────────────────────────
rows = []
total_ft = 0.0
n = len(indices) - 1
progress = st.progress(0)

for i in range(n):
    lab1, lat1, lon1 = agms_sorted[i]
    lab2, lat2, lon2 = agms_sorted[i+1]
    idx1, idx2 = indices[i], indices[i+1]

    d_m = path_distance(centerline, idx1, idx2)
    d_ft = d_m * FT_PER_M
    d_mi = d_ft * MI_PER_FT
    total_ft += d_ft

    rows.append({
        "Segment": f"Distance from {lab1} to {lab2}:",
        "Distance (ft)": f"{d_ft:,.2f}",
        "Distance (mi)": f"{d_mi:.4f}"
    })
    progress.progress((i+1)/n)

# ─── OUTPUT ──────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
st.subheader("AGM Segment Distances")
st.dataframe(df, use_container_width=True)

total_mi = total_ft * MI_PER_FT
st.markdown(f"**Total:** {total_ft:,.2f} ft &nbsp;|&nbsp; {total_mi:.4f} mi")

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV", csv, "agm_distances.csv", "text/csv")
