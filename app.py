import streamlit as st
import requests
import math
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
import srtm  # installed via requirements.txt

# ── Constants ──────────────────────────────────────────────────────────────────
FT_PER_M = 3.28084

# ── Load local SRTM tiles ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_srtm():
    return srtm.get_data()

elev_data = load_srtm()

# ── Elevation Helpers ──────────────────────────────────────────────────────────
def fetch_open_elev(lat, lon):
    url = "https://api.open-elevation.com/api/v1/lookup"
    resp = requests.get(url, params={"locations": f"{lat:.6f},{lon:.6f}"}, timeout=10)
    resp.raise_for_status()
    return float(resp.json()["results"][0]["elevation"])

def get_elevation(lat, lon):
    elev = elev_data.get_elevation(lat, lon)
    if elev is None:
        elev = fetch_open_elev(lat, lon)
        src = "Open-Elevation"
    else:
        src = "SRTM"
    return elev, src

# ── Distance Functions ─────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ── KML/KMZ Parser ─────────────────────────────────────────────────────────────
def parse_centerline(upload_file):
    data = upload_file.read()
    if upload_file.name.lower().endswith(".kmz"):
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

# ── Streamlit App ──────────────────────────────────────────────────────────────
st.set_page_config("AGM Segment Distances", layout="wide")
st.title("📏 Terrain-Aware Segment Distances Along AGM Centerline (Feet)")

upload = st.file_uploader("Upload centerline (KML or KMZ)", type=["kml", "kmz"])
if not upload:
    st.info("Please upload a KML or KMZ file containing your centerline.")
    st.stop()

# parse once
try:
    points = parse_centerline(upload)
except Exception as e:
    st.error(f"Failed to parse KML/KMZ: {e}")
    st.stop()

if len(points) < 2:
    st.error("Centerline must contain at least two points.")
    st.stop()

st.success(f"Parsed {len(points)} centerline points.")

if st.button("▶️ Compute Segment Distances"):
    n = len(points) - 1
    progress = st.progress(0)

    # compute cumulative planar distances (meters)
    cumul = [0.0]
    for i in range(n):
        lat1, lon1 = points[i]
        lat2, lon2 = points[i+1]
        d2d = haversine(lat1, lon1, lat2, lon2)
        cumul.append(cumul[-1] + d2d)

    # build table rows
    rows = []
    total_ft = 0.0
    for i in range(n):
        lat1, lon1 = points[i]
        lat2, lon2 = points[i+1]

        # segment planar & elevations
        d2d_m = haversine(lat1, lon1, lat2, lon2)
        e1, src1 = get_elevation(lat1, lon1)
        e2, src2 = get_elevation(lat2, lon2)
        d3d_m = math.sqrt(d2d_m**2 + (e2 - e1)**2)

        # convert to feet
        start_ft = cumul[i] * FT_PER_M
        end_ft   = cumul[i+1] * FT_PER_M
        seg_ft   = d3d_m * FT_PER_M
        total_ft += seg_ft

        # label with zero-padded integers
        lbl = f"Distance from {int(start_ft):03d} to {int(end_ft):03d}:"

        rows.append({"Segment": lbl, "Distance (ft)": f"{seg_ft:.2f}"})
        progress.progress((i+1)/n)

    # show results
    st.subheader("Segment Distances")
    st.table(rows)

    st.markdown(f"**Total Terrain-Aware Distance:** {total_ft:.2f} ft")
```
