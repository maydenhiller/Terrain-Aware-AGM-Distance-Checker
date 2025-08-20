import os
import math
import io
import re
import zipfile
import requests
import pandas as pd
import streamlit as st
import xml.etree.ElementTree as ET
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
from pyproj import CRS, Transformer
import srtm

FT_PER_M = 3.28084
MI_PER_FT = 1/5280
DEFAULT_STEP_M = 5.0

EPQS_URL = "https://nationalmap.gov/epqs/pqs.php"
OPENTOPO_URL = "https://portal.opentopography.org/API/point"
OPENTOPO_KEY = os.getenv("OPENTOPO_KEY")
OPENTOPO_DEMTYPES = ["USGS3DEP1m", "USGSNED10m", "SRTMGL1"]
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"

st.set_page_config(page_title="Terrain-aware AGM distances", layout="wide")
st.title("📏 Terrain-aware AGM distances (KML/KMZ)")

with st.expander("Advanced options", expanded=False):
    step_m = st.number_input("Densification step (m)", 1.0, 50.0, DEFAULT_STEP_M, step=1.0)
    debug = st.checkbox("Enable debug", value=False)

uploaded = st.file_uploader("Upload KML/KMZ with AGMs + red Centerline", type=["kml", "kmz"])

def extract_kml_from_kmz(data: bytes) -> bytes | None:
    with zipfile.ZipFile(io.BytesIO(data), 'r') as zf:
        for name in zf.namelist():
            if name.lower().endswith(".kml"):
                return zf.read(name)
    return None

def parse_station(label: str) -> tuple[int, str]:
    m = re.match(r"^(\d+)([A-Za-z]*)$", label.strip())
    return (int(m.group(1)), m.group(2)) if m else (0, "")

def parse_kml_kmz(file):
    raw = file.read()
    if file.name.lower().endswith(".kmz"):
        raw = extract_kml_from_kmz(raw) or b""
    if not raw:
        return [], []

    root = ET.fromstring(raw)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    # --- AGMs folder ---
    agms = []
    for fld in root.findall(".//kml:Folder", ns):
        nm = fld.find("kml:name", ns)
        if nm is not None and nm.text.strip().lower() == "agms":
            for pm in fld.findall(".//kml:Placemark", ns):
                name_el = pm.find("kml:name", ns)
                coord_el = pm.find(".//kml:Point/kml:coordinates", ns)
                if not (name_el is not None and coord_el is not None):
                    continue
                label = name_el.text.strip()
                lon, lat, *_ = coord_el.text.strip().split(",")
                agms.append((label, float(lat), float(lon)))

    # --- Centerline folder, only red ---
    centerline_segments = []
    for fld in root.findall(".//kml:Folder", ns):
        nm = fld.find("kml:name", ns)
        if nm is not None and nm.text.strip().lower() == "centerline":
            for pm in fld.findall(".//kml:Placemark", ns):
                style_url = pm.find("kml:styleUrl", ns)
                if style_url is None or "red" not in style_url.text.lower():
                    continue
                for ls in pm.findall(".//kml:LineString", ns):
                    coords_el = ls.find("kml:coordinates", ns)
                    if coords_el is None or not coords_el.text.strip():
                        continue
                    coords = []
                    for pair in coords_el.text.strip().split():
                        lon, lat, *_ = pair.split(",")
                        coords.append((float(lon), float(lat)))
                    if len(coords) >= 2:
                        centerline_segments.append(coords)

    return agms, centerline_segments

def utm_crs_for(lats, lons):
    lat_mean = sum(lats)/len(lats)
    lon_mean = sum(lons)/len(lons)
    zone = int((lon_mean+180)//6) + 1
    epsg = 32600+zone if lat_mean >= 0 else 32700+zone
    return CRS.from_epsg(epsg)

@st.cache_resource
def get_srtm():
    return srtm.get_data()

srtm_data = get_srtm()

@st.cache_data(ttl=24*3600, max_entries=200000)
def get_elevation(lat, lon):
    # 1) EPQS
    try:
        r = requests.get(EPQS_URL, params={"x": lon, "y": lat, "units": "Meters", "output": "json"}, timeout=6)
        r.raise_for_status()
        e = r.json()["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]["Elevation"]
        if e is not None:
            return float(e)
    except: pass
    # 2) OpenTopo
    if OPENTOPO_KEY:
        for dem in OPENTOPO_DEMTYPES:
            try:
                r = requests.get(OPENTOPO_URL, params={"x": lon, "y": lat, "demtype": dem,
                                                       "outputFormat": "JSON", "key": OPENTOPO_KEY}, timeout=6)
                r.raise_for_status()
                j = r.json()
                if "data" in j and "elevation" in j["data"]:
                    return float(j["data"]["elevation"])
            except: continue
    # 3) Local SRTM
    try:
        e = srtm_data.get_elevation(lat, lon)
        if e is not None:
            return float(e)
    except: pass
    # 4) Open-Elevation
    try:
        r = requests.get(OPEN_ELEVATION_URL, params={"locations": f"{lat},{lon}"}, timeout=6)
        r.raise_for_status()
        return float(r.json()["results"][0]["elevation"])
    except: return 0.0

def build_centerline(lines_wgs84, to_utm):
    utm_lines = []
    for line in lines_wgs84:
        if not line or len(line) < 2: continue
        xs, ys = to_utm.transform(*zip(*line))
        if len(xs) >= 2:
            utm_lines.append(LineString(zip(xs, ys)))
    if not utm_lines:
        raise ValueError("No valid Centerline geometry.")
    return linemerge(utm_lines if len(utm_lines) > 1 else utm_lines[0])

def densified_points(line_utm, s0, s1, step):
    a, b = (s0, s1) if s0 <= s1 else (s1, s0)
    if b - a < 1e-6:
        b = a + step
    dists = [a + i*step for i in range(int(math.floor((b-a)/step))+1)]
    if dists[-1] < b: dists.append(b)
    pts = [line_utm.interpolate(s) for s in dists]
    return [(p.x, p.y) for p in pts]

def terrain_distance_m(pts_xy, to_wgs84):
    total = 0.0
    xs, ys = zip(*pts_xy)
    lons, lats = to_wgs84.transform(xs, ys)
    elevs = [get_elevation(lat, lon) for lat, lon in zip(lats, lons)]
    for i in range(len(pts_xy)-1):
        h = math.hypot(pts_xy[i+1][0]-pts_xy[i][0],
                       pts_xy[i+1][1]-pts_xy[i][1])
        v = elevs[i+1] - elevs[i]
        total += math.hypot(h, v)
    return total

if not uploaded:
    st.info("Please upload a KML/KMZ file.")
    st.stop()

agms, centerline_ll = parse_kml_kmz(uploaded)
if not agms:
    st.error("No AGMs found in AGMs folder.")
    st.stop()
if not centerline_ll:
    st.error("No red Centerline found.")
    st.stop()

all_lons = [lon for seg in centerline_ll for (lon, _) in seg]
all_lats = [lat for seg in centerline_ll for (_, lat) in seg]
crs_utm = utm_crs_for(all_lats, all_lons)
to_utm = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
to_wgs84 = Transformer.from_crs
