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
DEFAULT_STEP_M = 5.0  # densification step along centerline in meters

# Elevation sources
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
        min_value=1.0,
        max_value=50.0,
        value=DEFAULT_STEP_M,
        step=1.0,
        help="Smaller steps increase accuracy but require more elevation queries."
    )
    show_debug = st.checkbox("Show debug info", value=False)

uploaded = st.file_uploader("Upload KML or KMZ containing AGMs and CENTERLINE", type=["kml", "kmz"])
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

# ---------------------------- KML parsing ----------------------------
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
            if name_el is None or coord_el is None or not (coord_el.text or "").strip():
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
            if coords_el is None or not (coords_el.text or "").strip():
                continue
            seg = []
            for token in coords_el.text.strip().split():
                lon, lat, *_ = token.split(",")
                seg.append((float(lon), float(lat)))
            if len(seg) >= 2:
                centerline_segments.append(seg)

    return agms, centerline_segments

# ---------------------------- Elevation stack ----------------------------
@st.cache_resource(show_spinner=False)
def get_srtm():
    return srtm.get_data()

srtm_data = get_srtm()

@st.cache_data(show_spinner=False, ttl=24 * 3600, max_entries=200000)
def get_elevation(lat: float, lon: float) -> float:
    try:
        r = requests.get(
            EPQS_URL,
            params={"x": lon, "y": lat, "units": "Meters", "output": "json"},
            timeout=6,
        )
        r.raise_for_status()
        e = r.json()["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]["Elevation"]
        if e is not None:
            return float(e)
    except Exception:
        pass

    if OPENTOPO_KEY:
        for dem in OPENTOPO_DEMTYPES:
            try:
                r = requests.get(
                    OPENTOPO_URL,
                    params={"x": lon, "y": lat, "demtype": dem, "outputFormat": "JSON", "key": OPENTOPO_KEY},
                    timeout=6,
                )
                r.raise_for_status()
                j = r.json()
                if "data" in j and "elevation" in j["data"]:
                    return float(j["data"]["elevation"])
            except Exception:
                continue

    try:
        e = srtm_data.get_elevation(lat, lon)
        if e is not None:
            return float(e)
    except Exception:
        pass

    try:
        r = requests.get(OPEN_ELEVATION_URL, params={"locations": f"{lat:.6f},{lon:.6f}"}, timeout=6)
        r.raise_for_status()
        return float(r.json()["results"][0]["elevation"])
    except Exception:
        return 0.0

# ---------------------------- Geometry helpers ----------------------------
def build_centerline_utm(segments_ll, to_utm: Transformer) -> LineString:
    utm_lines = []
    for seg in segments_ll:
        if not seg or len(seg) < 2:
            continue
        xs, ys = to_utm.transform(*zip(*seg))
        if len(xs) >= 2:
            utm_lines.append(LineString(list(zip(xs, ys))))
    if not utm_lines:
        raise ValueError("No valid Centerline geometry.")
    try:
        merged = linemerge(MultiLineString(utm_lines)) if len(utm_lines) > 1 else utm_lines[0]
        if isinstance(merged, LineString):
            return merged
        parts = list(merged.geoms)
        parts.sort(key=lambda g: g.length, reverse=True)
        return parts[0]
    except Exception:
        coords = []
        for ln in utm_lines:
            coords.extend(list(ln.coords))
        return LineString(coords)

def densified_points(line_utm, s0: float, s1: float, step: float):
    a, b = (s0, s1) if s0 <= s1 else (s1, s0)
    if abs(b - a) < 1e-6:
        b = a + step
    n_steps = max(1, int(math.floor((b - a) / step)))
    dists = [a + i * step for i in range(n_steps)]
    if not dists or dists[-1] < b:
        dists.append(b)
    pts = [line_utm.interpolate(d) for d in dists]
    pts_xy = [(p.x, p.y) for p in pts]
    if s0 > s1:
        pts_xy.reverse()
    return pts_xy

def terrain_distance_m(pts_xy, to_wgs84: Transformer) -> float:
    total = 0.0
    xs, ys = zip(*pts_xy)
    lons, lats = to_wgs84.transform(xs, ys)
    elevs = [get_elevation(lat, lon) for lat, lon in zip(lats, lons)]
    for i in range(len(pts_xy) - 1):
        h = math.hypot(pts_xy[i+1
