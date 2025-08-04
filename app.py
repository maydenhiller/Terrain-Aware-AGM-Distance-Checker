import streamlit as st
import requests
import math
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
import srtm
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
FT_PER_M = 3.28084
MI_PER_FT = 1 / 5280

# ── Load SRTM ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_srtm_data():
    return srtm.get_data()

elev_data = load_srtm_data()

# ── Elevation Helpers ──────────────────────────────────────────────────────────
def fetch_open_elev(lat, lon):
    url = "https://api.open-elevation.com/api/v1/lookup"
    r = requests.get(url, params={"locations": f"{lat:.6f},{lon:.6f}"})
    r.raise_for_status()
    return float(r.json()["results"][0]["elevation"])

def get_elevation(lat, lon):
    elev = elev_data.get_elevation(lat, lon)
    if elev is None:
        elev = fetch_open_elev(lat, lon)
    return elev

# ── Planar Distance (Haversine) ────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # earth radius in meters
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ── KML/KMZ Parser ─────────────────────────────────────────────────────────────
def parse_centerline(file_uploader):
    raw = file_uploader.read()
    if file_uploader.name.lower().endswith(".kmz"):
        z = zipfile.ZipFile(BytesIO(raw))
        for f in z.namelist():
            if f.lower().endswith(".kml"):
                raw = z.read(f)
                break
    root = ET.fromstring(raw)
    ns = {"kml": root.tag.split("}")[0].strip("{")}
    pts = []
    for ls in root.findall(".//kml:LineString", ns):
        coords = ls.find("kml:coordinates", ns).text.strip()
        for pair in coords.split():
            lon, lat, *_ = pair.split(",")
            pts.append((float(lat), float(lon)))
    return pts

# ── Streamlit Interface ───────────────────────────────────────────────────────
st.set_page_config("AGM Segment Distances", layout="wide")
st.title("📏 Terrain-Aware AGM Segment Distances")

upload = st.file_uploader("Upload KML or KMZ centerline", type=["kml", "kmz"])
if not upload:
    st.info("Please upload your KML/KMZ file.")
    st.stop()

try:
    points = parse_centerline(upload)
except Exception as e:
    st.error(f"Failed to parse file: {e}")
    st.stop()

if len(points) < 2:
    st.error("Centerline needs at least 2 points.")
    st.stop()

st.success(f"Loaded {len(points)} centerline points.")

if st.button("Compute Segment Distances"):
    n = len(points) - 1

    # cumulative planar (2D) distances in meters
    cumul2d = [0.0]
    for i in range(n):
        lat1, lon1 = points[i]
        lat2, lon2 = points[i+1]
        cumul2d.append(cumul2d[-1] + haversine(lat1, lon1, lat2, lon2))

    rows = []
    total_ft = 0.0

    prog = st.progress(0)
    for i in range(n):
        lat1, lon1 = points[i]
        lat2, lon2 = points[i+1]

        d2d = haversine(lat1, lon1, lat2, lon2)
        e1, e2 = get_elevation(lat1, lon1), get_elevation(lat2, lon2)
        d3d = math.sqrt(d2d**2 + (e2 - e1)**2)

        # convert to feet & miles
        seg_ft = d3d * FT_PER_M
        seg_mi = seg_ft * MI_PER_FT
        total_ft += seg_ft

        # station labels (zero-padded to 3 digits)
        start_ft = round(cumul2d[i] * FT_PER_M)
        end_ft   = round(cumul2d[i+1] * FT_PER_M)
        label = f"Distance from {start_ft:03d} to {end_ft:03d}:"

        rows.append({
            "Segment": label,
            "Distance (ft)": f"{seg_ft:.2f}",
            "Distance (mi)": f"{seg_mi:.4f}"
        })
        prog.progress((i+1)/n)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # show totals
    total_mi = total_ft * MI_PER_FT
    st.markdown(f"**Total Terrain-Aware Distance:** {total_ft:.2f} ft ({total_mi:.2f} mi)")

    # CSV download
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="segment_distances.csv",
        mime="text/csv"
    )
