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

# ---------------------------- Elevation Sources ----------------------------
EPQS_URL = "https://nationalmap.gov/epqs/pqs.php"
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"

@st.cache_resource
def get_srtm():
    return srtm.get_data()
srtm_data = get_srtm()

@st.cache_data(ttl=86400)
def get_elevation(lat: float, lon: float) -> float:
    try:
        r = requests.get(EPQS_URL, params={"x": lon, "y": lat, "units": "Meters", "output": "json"}, timeout=6)
        r.raise_for_status()
        return float(r.json()["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]["Elevation"])
    except Exception:
        pass
    try:
        e = srtm_data.get_elevation(lat, lon)
        if e is not None:
            return float(e)
    except Exception:
        pass
    try:
        r = requests.get(OPEN_ELEVATION_URL, params={"locations": f"{lat},{lon}"}, timeout=6)
        r.raise_for_status()
        return float(r.json()["results"][0]["elevation"])
    except Exception:
        return 0.0

# ---------------------------- Geometry Helpers ----------------------------
def utm_crs_for(lats, lons):
    lat_mean = sum(lats) / len(lats)
    lon_mean = sum(lons) / len(lons)
    zone = int((lon_mean + 180) // 6) + 1
    epsg = 32600 + zone if lat_mean >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)

def build_centerline_utm(segments_ll, to_utm):
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

def densified_points(line_utm, s0, s1, step):
    a, b = (s0, s1) if s0 <= s1 else (s1, s0)
    if abs(b - a) < 1e-6: b = a + step
    dists = [a + i * step for i in range(int((b - a) // step) + 1)]
    if dists[-1] < b: dists.append(b)
    pts = [line_utm.interpolate(d) for d in dists]
    pts_xy = [(p.x, p.y) for p in pts]
    return pts_xy if s0 <= s1 else list(reversed(pts_xy))

def terrain_distance_m(pts_xy, to_wgs84):
    xs, ys = zip(*pts_xy)
    lons, lats = to_wgs84.transform(xs, ys)
    elevs = [get_elevation(lat, lon) for lat, lon in zip(lats, lons)]
    total = 0.0
    for i in range(len(pts_xy) - 1):
        x1, y1 = pts_xy[i]
        x2, y2 = pts_xy[i + 1]
        h = math.hypot(x2 - x1, y2 - y1)
        v = elevs[i + 1] - elevs[i]
        total += math.hypot(h, v)
    return total

def extract_station_number(label):
    import re
    match = re.match(r"^(\d{3})([A-Za-z]*)$", label.strip())
    return (int(match.group(1)), match.group(2)) if match else (999, "")

# ---------------------------- KML Parsing ----------------------------
def parse_kml_for_agms_and_centerline(kml_bytes):
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    root = ET.fromstring(kml_bytes)

    agms = []
    for fld in root.findall(".//kml:Folder", ns):
        name = fld.find("kml:name", ns)
        if name is not None and name.text.strip().lower() == "agms":
            for pm in fld.findall("kml:Placemark", ns):
                label = pm.find("kml:name", ns).text.strip()
                coords = pm.find(".//kml:Point/kml:coordinates", ns).text.strip()
                lon, lat, *_ = coords.split(",")
                agms.append((label, float(lon), float(lat)))

    centerline = []
    for fld in root.findall(".//kml:Folder", ns):
        name = fld.find("kml:name", ns)
        if name is not None and name.text.strip().lower() == "centerline":
            for pm in fld.findall("kml:Placemark", ns):
                style = pm.find("kml:styleUrl", ns)
                if style is None or "#2_0" not in style.text: continue
                coords = pm.find(".//kml:LineString/kml:coordinates", ns).text.strip()
                seg = []
                for token in coords.split():
                    lon, lat, *_ = token.split(",")
                    seg.append((float(lon), float(lat)))
                if len(seg) >= 2:
                    centerline.append(seg)

    return agms, centerline

# ---------------------------- Streamlit UI ----------------------------
st.set_page_config(page_title="Terrain-aware AGM distances", layout="wide")
st.title("Terrain-aware AGM Segment Distances")

step_m = st.number_input("Densification step (meters)", min_value=1.0, max_value=50.0, value=DEFAULT_STEP_M)
uploaded = st.file_uploader("Upload KML or KMZ", type=["kml", "kmz"])
if not uploaded: st.stop()

raw = uploaded.read()
kml_bytes = zipfile.ZipFile(io.BytesIO(raw)).read("doc.kml") if uploaded.name.lower().endswith(".kmz") else raw
if isinstance(kml_bytes, str): kml_bytes = kml_bytes.encode("utf-8")

agms, centerline_ll = parse_kml_for_agms_and_centerline(kml_bytes)
if not agms or not centerline_ll:
    st.error("AGMs or red centerline not found.")
    st.stop()

# CRS setup
all_lons = [lon for seg in centerline_ll for lon, _ in seg]
all_lats = [lat for seg in centerline_ll for _, lat in seg]
crs_utm = utm_crs_for(all_lats, all_lons)
to_utm = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
to_wgs84 = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)

line_utm = build_centerline_utm(centerline_ll, to_utm)

# Project AGMs to centerline
agm_chain = []
for label, lon, lat in agms:
    x, y = to_utm.transform(lon, lat)
    s = line_utm.project(Point(x, y))
    agm_chain.append((label, lon, lat, s))

# Sort by station number
agm_chain.sort(key=lambda x: extract_station_number(x[0]))

# Rebase to AGM 000
offset = next((s for lab, _, _, s in agm_chain if extract_station_number(lab)[0] == 0), agm_chain[0][3])
agm_chain = [(lab, lon, lat, s - offset) for lab, lon, lat, s in agm_chain]

# Compute segment distances
rows = []
total_ft = 0.0
for i in range(len(agm_chain) - 1):
    lab1, _, _, s0 = agm_chain[i]
    lab2, _, _, s1 = agm_chain[i + 1]
    pts_xy = densified_points(line_utm, s0 + offset, s1 + offset, step_m)
    dist_m = terrain_distance_m(pts_xy, to_wgs84)
    dist_ft = dist_m * METERS_TO_FEET
    dist_mi = dist
