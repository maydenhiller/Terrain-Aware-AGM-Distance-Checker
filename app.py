import streamlit as st
import requests
import math
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
import srtm  # note: this comes from 'srtm.py' in requirements

# ── Load local SRTM tiles (cached) ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_srtm():
    return srtm.get_data()

elev_data = load_srtm()

# ── Helpers ────────────────────────────────────────────────────────────────────
def fetch_open_elev(lat, lon):
    url = "https://api.open-elevation.com/api/v1/lookup"
    resp = requests.get(url, params={"locations": f"{lat:.6f},{lon:.6f}"}, timeout=10)
    resp.raise_for_status()
    return float(resp.json()["results"][0]["elevation"])

def get_elevation(lat, lon):
    # try local SRTM first
    elev = elev_data.get_elevation(lat, lon)
    if elev is None:
        elev = fetch_open_elev(lat, lon)
        src = "Open-Elevation"
    else:
        src = "SRTM"
    return elev, src

def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def parse_centerline(uploaded_file):
    data = uploaded_file.read()
    if uploaded_file.name.lower().endswith(".kmz"):
        z = zipfile.ZipFile(BytesIO(data))
        for name in z.namelist():
            if name.lower().endswith(".kml"):
                data = z.read(name)
                break
    root = ET.fromstring(data)
    ns = {"kml": root.tag.split("}")[0].strip("{")}
    pts = []
    for ls in root.findall(".//kml:LineString", ns):
        text = ls.find("kml:coordinates", ns).text.strip()
        for tok in text.split():
            lon, lat, *_ = tok.split(",")
            pts.append((float(lat), float(lon)))
    return pts

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config("Fast AGM Distances", layout="wide")
st.title("🚀 Terrain-Aware AGM Distances (Local SRTM + Progress)")

upload = st.file_uploader("Upload centerline (KML or KMZ)", type=["kml", "kmz"])
if not upload:
    st.info("Please upload a KML or KMZ to begin.")
    st.stop()

# parse once
try:
    points = parse_centerline(upload)
except Exception as e:
    st.error(f"Failed to parse file: {e}")
    st.stop()

if len(points) < 2:
    st.error("Centerline must contain at least two points.")
    st.stop()

st.success(f"Parsed {len(points)} points from centerline.")

if st.button("▶️ Compute Distances"):
    total_2d = 0.0
    total_3d = 0.0
    n = len(points) - 1
    progress = st.progress(0)

    rows = []
    for i in range(n):
        lat1, lon1 = points[i]
        lat2, lon2 = points[i + 1]

        d2d = haversine(lat1, lon1, lat2, lon2)
        e1, src1 = get_elevation(lat1, lon1)
        e2, src2 = get_elevation(lat2, lon2)

        d3d = math.sqrt(d2d**2 + (e2 - e1)**2)

        total_2d += d2d
        total_3d += d3d

        rows.append({
            "Segment": i + 1,
            "2D (m)": f"{d2d:.2f}",
            "ΔElev (m)": f"{(e2 - e1):.2f}",
            "3D (m)": f"{d3d:.2f}",
            "Src Start": src1,
            "Src End": src2,
        })

        progress.progress((i + 1) / n)

    st.subheader("Segment Distances")
    st.table(rows)

    st.markdown(f"**Total 2D Distance:** {total_2d:.2f} m  ")
    st.markdown(f"**Total 3D Distance:** {total_3d:.2f} m  ")

