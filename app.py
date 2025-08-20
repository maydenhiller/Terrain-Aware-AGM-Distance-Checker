import os
import io
import math
import zipfile
import xml.etree.ElementTree as ET

import requests
import streamlit as st
import pandas as pd
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import linemerge
from pyproj import CRS, Transformer
import srtm  # pip install srtm.py

# ---------------------------- Config ----------------------------
METERS_TO_FEET = 3.28084
FEET_PER_MILE = 5280
DEFAULT_STEP_M = 5.0

EPQS_URL = "https://nationalmap.gov/epqs/pqs.php"
OPENTOPO_URL = "https://portal.opentopography.org/API/point"
OPENTOPO_KEY = os.getenv("OPENTOPO_KEY") or st.secrets.get("OPENTOPO_KEY", None)
OPENTOPO_DEMTYPES = ["USGS3DEP1m", "USGSNED10m", "SRTMGL1"]
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"

# ---------------------------- UI ----------------------------
st.set_page_config(page_title="Terrain-aware AGM distances", layout="wide")
st.title("Terrain-aware AGM distances (KML/KMZ)")

with st.expander("Options", expanded=False):
    step_m = st.number_input(
        "Densification step (meters)",
        min_value=1.0, max_value=50.0,
        value=DEFAULT_STEP_M, step=1.0
    )
    show_debug = st.checkbox("Show debug info", value=False)

uploaded = st.file_uploader(
    "Upload KML or KMZ containing AGMs and CENTERLINE", type=["kml", "kmz"]
)
if not uploaded:
    st.stop()

# ---------------------------- Helpers ----------------------------
def read_uploaded_bytes(file) -> bytes:
    file.seek(0)
    return file.read()

def extract_kml_from_kmz(data: bytes) -> bytes | None:
    with zipfile.ZipFile(io.BytesIO(data), 'r') as zf:
        for name in zf.namelist():
            if name.lower().endswith(".kml"):
                return zf.read(name)
    return None

def utm_crs_for(lats, lons) -> CRS:
    lat_mean = sum(lats) / len(lats)
    lon_mean = sum(lons) / len(lons)
    zone = int((lon_mean + 180) // 6) + 1
    epsg = 32600 + zone if lat_mean >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)

def parse_station(label: str) -> tuple[int, str]:
    import re
    m = re.match(r"^\s*(\d+)\s*([A-Za-z]*)\s*$", label or "")
    return (int(m.group(1)), m.group(2)) if m else (0, "")

def parse_kml_for_agms_and_centerline(kml_bytes: bytes):
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    root = ET.fromstring(kml_bytes)

    agms = []
    for fld in root.findall(".//kml:Folder", ns):
        nm = fld.find("kml:name", ns)
        if nm is None or (nm.text or "").strip().lower() != "agms":
            continue
        for pm in fld.findall("kml:Placemark", ns):
            name_el = pm.find("kml:name", ns)
            coord_el = pm.find(".//kml:Point/kml:coordinates", ns)
            if name_el is None or coord_el is None or not coord_el.text.strip():
                continue
            label = (name_el.text or "").strip()
            lon, lat, *_ = coord_el.text.strip().split(",")
            agms.append((label, float(lon), float(lat)))

    centerline_segments = []
    for fld in root.findall(".//kml:Folder", ns):
        nm = fld.find("kml:name", ns)
        if nm is None or (nm.text or "").strip().lower() != "centerline":
            continue
        for pm in fld.findall("kml:Placemark", ns):
            style_url = pm.find("kml:styleUrl", ns)
            if style_url is None or "#2_0" not in (style_url.text or ""):
                continue
            coords_el = pm.find(".//kml:LineString/kml:coordinates", ns)
            if coords_el is None or not coords_el.text.strip():
                continue
            seg = []
            for token in coords_el.text.strip().split():
                lon, lat, *_ = token.split(",")
                seg.append((float(lon), float(lat)))
            if len(seg) >= 2:
                centerline_segments.append(seg)

    return agms, centerline_segments

@st.cache_resource(show_spinner=False)
def get_srtm():
    return srtm.get_data()
srtm_data = get_srtm()

@st.cache_data(show_spinner=False, ttl=86400, max_entries=200000)
def get_elevation(lat: float, lon: float) -> float:
    try:
        r = requests.get(EPQS_URL, params={"x": lon, "y": lat, "units": "Meters", "output": "json"}, timeout=6)
        r.raise_for_status()
        e = r.json()["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]["Elevation"]
        if e is not None: return float(e)
    except Exception:
        pass
    if OPENTOPO_KEY:
        for dem in OPENTOPO_DEMTYPES:
            try:
                r = requests.get(OPENTOPO_URL,
                                 params={"x": lon, "y": lat, "demtype": dem,
                                         "outputFormat": "JSON", "key": OPENTOPO_KEY},
                                 timeout=6)
                r.raise_for_status()
                j = r.json()
                if "data" in j and "elevation" in j["data"]:
                    return float(j["data"]["elevation"])
            except Exception:
                continue
    try:
        e = srtm_data.get_elevation(lat, lon)
        if e is not None: return float(e)
    except Exception:
        pass
    try:
        r = requests.get(OPEN_ELEVATION_URL, params={"locations": f"{lat:.6f},{lon:.6f}"}, timeout=6)
        r.raise_for_status()
        return float(r.json()["results"][0]["elevation"])
    except Exception:
        return 0.0

def build_centerline_utm(segments_ll, to_utm: Transformer) -> LineString:
    utm_lines = []
    for seg in segments_ll:
        if len(seg) < 2: continue
        xs, ys = to_utm.transform(*zip(*seg))
        utm_lines.append(LineString(zip(xs, ys)))
    merged = linemerge(MultiLineString(utm_lines)) if len(utm_lines) > 1 else utm_lines[0]
    if isinstance(merged, LineString):
        return merged
    parts = list(merged.geoms)
    parts.sort(key=lambda g: g.length, reverse=True)
    return parts[0]

def densified_points(line_utm, s0: float, s1: float, step: float):
    a, b = (s0, s1) if s0 <= s1 else (s1, s0)
    if abs(b - a) < 1e-6: b = a + step
    n_steps = max(1, int(math.floor((b - a) / step)))
    dists = [a + i * step for i in range(n_steps)]
    if not dists or dists[-1] < b: dists.append(b)
    pts = [line_utm.interpolate(d) for d in dists]
    pts_xy = [(p.x, p.y) for p in pts]
    if s0 > s1: pts_xy.reverse()
    return pts_xy

def terrain_distance_m(pts_xy, to_wgs84: Transformer) -> float:
    total = 0.0
    xs, ys = zip(*pts_xy)
    lons, lats = to_wgs84.transform(xs, ys)
    elevs = [get_elevation(lat, lon) for lat, lon in zip(lats, lons)]
    for i in range(len(pts_xy) - 1):
        x1, y1 = pts_xy[i]
        x2, y2 = pts_xy[i + 1]
        h = math.hypot(x2 - x1, y2 - y1)
        v = elevs[i + 1] - elevs[i]
        total += math.hypot(h, v)
    return total

# ---------------------------- Main flow ----------------------------
data = read_uploaded_bytes(uploaded)
kml_bytes = extract_kml_from_kmz(data) if uploaded.name.lower().endswith(".kmz") else data

agms, centerline_segments = parse_kml_for_agms_and_centerline(kml_bytes)

if not agms:
    st.error("No AGMs found in the 'AGMs' folder.")
    st.stop()
if not centerline_segments:
    st.error("No red CENTERLINE segments found (style '#2_0').")
    st.stop()

# Build UTM CRS and transformers from centerline extents
all_lons = [lon for seg in centerline_segments for (lon, lat) in seg]
all_lats = [lat for seg in centerline_segments for (lon, lat) in seg]
crs_utm = utm_crs_for(all_lats, all_lons)
to_utm = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
to_wgs84 = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)

# Centerline in UTM
line_utm = build_centerline_utm(centerline_segments, to_utm)
line_len_m = float(line_utm.length)

# Project AGMs to line and order along the centerline
records = []
for label, lon, lat in agms:
    s_num, suf = parse_station(label)
    x, y = to_utm.transform(lon, lat)
    s_on = float(line_utm.project(Point(x, y)))
    records.append(
        {
            "label": label,
            "num": s_num,
            "suffix": suf,
            "lon": lon,
            "lat": lat,
            "x": x,
            "y": y,
            "s_on": s_on,
        }
    )

# Sort AGMs by their curvilinear position on the centerline
recs_sorted = sorted(records, key=lambda r: r["s_on"])

# Start from AGM 000 at 0 feet if present; otherwise start from first AGM along line
start_idx = next((i for i, r in enumerate(recs_sorted) if r["num"] == 0), None)
if start_idx is None:
    st.warning("AGM '000' not found. Starting from the first AGM along the centerline.")
    start_idx = 0

# Build segment distances between consecutive AGMs (000→010, 010→020, ...)
rows = []
cum_ft = 0.0
for i in range(start_idx, len(recs_sorted) - 1):
    r0 = recs_sorted[i]
    r1 = recs_sorted[i + 1]

    s0 = max(0.0, min(r0["s_on"], line_len_m))
    s1 = max(0.0, min(r1["s_on"], line_len_m))

    # Densify the path between AGMs along the centerline and compute terrain-aware 3D distance
    pts_xy = densified_points(line_utm, s0, s1, float(step_m))
    seg_m = terrain_distance_m(pts_xy, to_wgs84)

    seg_ft = seg_m * METERS_TO_FEET
    seg_mi = seg_ft / FEET_PER_MILE
    cum_ft += seg_ft

    lab0 = f"{r0['num']:03d}"
    lab1 = f"{r1['num']:03d}"

    rows.append(
        {
            "Segment": f"{lab0} to {lab1}",
            "Distance (ft)": seg_ft,
            "Distance (mi)": seg_mi,
            "Total Distance So Far (ft)": cum_ft,
        }
    )

df = pd.DataFrame(rows)

if df.empty:
    st.error("No AGM segments to measure. Need at least two AGMs aligned to the centerline.")
    st.stop()

# Display table
st.subheader("Segment distances")
st.dataframe(
    df.style.format(
        {
            "Distance (ft)": "{:,.2f}",
            "Distance (mi)": "{:,.4f}",
            "Total Distance So Far (ft)": "{:,.2f}",
        }
    ),
    use_container_width=True,
)

# CSV download
csv_data = df.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name="agm_segment_distances.csv",
    mime="text/csv",
)

# Optional debug info
if show_debug:
    st.write(
        {
            "utm_crs": str(crs_utm),
            "line_length_m": round(line_len_m, 3),
            "num_agms_found": len(agms),
            "num_centerline_segments": len(centerline_segments),
            "densify_step_m": float(step_m),
            "start_index": start_idx,
        }
    )
