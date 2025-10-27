# app.py
# Terrain-Aware AGM Distance Calculator — Geodesic-only snap/slice + true 3D integration
# - No UTM used for length; only for nothing (we removed projection-based slicing entirely)
# - Centerline parts are densified by geodesic arclength (default 5 m)
# - AGMs snap to nearest densified vertex on a single best-matching part
# - Slice between snapped indices; resample by DEM-aware spacing
# - 3D distance = sum sqrt( geodesic_dxy^2 + dz^2 )

import math
import io
import zipfile
import requests
import xml.etree.ElementTree as ET

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Point, LineString
from pyproj import Geod

# =========================
# CONFIG / UI
# =========================

st.set_page_config(page_title="Terrain-Aware AGM Distance Calculator", layout="wide")
st.title("Terrain-Aware AGM Distance Calculator (Geodesic Snap/Slice + True 3D)")

MAPBOX_TOKEN = st.secrets["mapbox"]["token"]
TERRAIN_TILE_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"
GEOD = Geod(ellps="WGS84")

with st.sidebar:
    st.header("Settings")
    mapbox_zoom = st.slider("Terrain tile zoom", 15, 17, 17)
    densify_step_m = st.slider("Centerline densify step (m)", 2.0, 20.0, 5.0, 1.0)
    requested_spacing_m = st.slider("Requested sampling spacing (m)", 0.5, 10.0, 1.0, 0.5)
    snap_max_offset_m = st.slider("Max AGM snap offset (m)", 5.0, 150.0, 60.0, 5.0)
    supersample = st.checkbox("Supersample elevation per sample (center + 4 offsets)", value=True)
    median_win = st.slider("Elevation median window (samples)", 1, 25, 9, 2)
    mean_win = st.slider("Elevation moving-average window (samples)", 1, 25, 7, 2)
    st.caption("Best results: densify=5 m, zoom=17, spacing≈auto, median 9–15, mean 5–11, supersample ON.")

FT_PER_M = 3.28084
MI_PER_FT = 1.0 / 5280.0
INITIAL_RES_M_PER_PX = 156543.03392  # Web Mercator @ equator, z=0

# =========================
# KML / KMZ PARSER
# =========================

def agm_sort_key(name_geom):
    name = name_geom[0]
    base_digits = ''.join(filter(str.isdigit, name))
    base = int(base_digits) if base_digits else -1
    suffix = ''.join(filter(str.isalpha, name)).upper()
    return (base, suffix)

def parse_kml_kmz(uploaded_file):
    """Returns: AGMs[(name, Point lon/lat)], CenterlineParts[List[LineString lon/lat]]"""
    if uploaded_file.name.endswith(".kmz"):
        with zipfile.ZipFile(uploaded_file) as zf:
            kml_file = next((f for f in zf.namelist() if f.endswith(".kml")), None)
            if not kml_file:
                return [], []
            with zf.open(kml_file) as f:
                kml_data = f.read()
    else:
        kml_data = uploaded_file.read()

    root = ET.fromstring(kml_data)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    agms = []
    parts = []

    for folder in root.findall(".//kml:Folder", ns):
        name_el = folder.find("kml:name", ns)
        if name_el is None or not name_el.text:
            continue
        folder_name = name_el.text.strip().lower()

        if folder_name == "agms":
            for placemark in folder.findall("kml:Placemark", ns):
                pname = placemark.find("kml:name", ns)
                coords = placemark.find(".//kml:coordinates", ns)
                if pname is None or coords is None:
                    continue
                try:
                    lon, lat, *_ = map(float, coords.text.strip().split(","))
                    agms.append((pname.text.strip(), Point(lon, lat)))
                except Exception:
                    continue

        elif folder_name == "centerline":
            for placemark in folder.findall("kml:Placemark", ns):
                coords = placemark.find(".//kml:coordinates", ns)
                if coords is None:
                    continue
                pts = []
                for pair in coords.text.strip().split():
                    lon, lat, *_ = map(float, pair.split(","))
                    pts.append((lon, lat))
                if len(pts) >= 2:
                    parts.append(LineString(pts))

    agms.sort(key=agm_sort_key)
    return agms, parts

# =========================
# GEODESIC HELPERS
# =========================

def geodesic_segment_m(p0, p1):
    lon1, lat1 = p0
    lon2, lat2 = p1
    _, _, d = GEOD.inv(lon1, lat1, lon2, lat2)
    return float(d)

def meters_per_pixel(lat_deg, z):
    # Web Mercator ground resolution
    return (INITIAL_RES_M_PER_PX * math.cos(math.radians(lat_deg))) / (2 ** z)

def densify_polyline_geodesic(coords_ll, step_m):
    """Return a densified lon/lat polyline at ~step_m geodesic spacing (includes endpoints)."""
    if len(coords_ll) < 2:
        return coords_ll
    out = [coords_ll[0]]
    for i in range(len(coords_ll) - 1):
        a = coords_ll[i]
        b = coords_ll[i + 1]
        seg_len = geodesic_segment_m(a, b)
        if seg_len <= 0:
            continue
        n = max(int(seg_len // step_m), 1)
        # linear in lon/lat is fine for tiny steps; GEOD.npts could also be used
        for k in range(1, n):
            t = k / n
            lon = a[0] + t * (b[0] - a[0])
            lat = a[1] + t * (b[1] - a[1])
            out.append((lon, lat))
        out.append(b)
    # dedupe possible duplicates
    dedup = [out[0]]
    for p in out[1:]:
        if p != dedup[-1]:
            dedup.append(p)
    return dedup

def geodesic_snap_to_polyline(agm_lonlat, densified_coords):
    """Snap a point to the nearest densified vertex; returns (index, distance_m)."""
    ax, ay = agm_lonlat
    best_idx = None
    best_d = float("inf")
    for idx, (lon, lat) in enumerate(densified_coords):
        _, _, d = GEOD.inv(ax, ay, lon, lat)
        if d < best_d:
            best_d = d
            best_idx = idx
    return best_idx, best_d

# =========================
# MAPBOX TERRAIN-RGB (bilinear + optional supersample)
# =========================

def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return int(x), int(y), x, y

def decode_terrain_rgb(r, g, b):
    return -10000.0 + (r * 256.0 * 256.0 + g * 256.0 + b) * 0.1

class TerrainTileCache:
    def __init__(self, token, zoom=17):
        self.token = token
        self.zoom = zoom
        self.cache = {}

    def get_tile_array(self, z, x, y):
        key = (z, x, y)
        arr = self.cache.get(key)
        if arr is not None:
            return arr
        url = TERRAIN_TILE_URL.format(z=z, x=x, y=y)
        resp = requests.get(url, params={"access_token": self.token}, timeout=20)
        if resp.status_code != 200:
            return None
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        self.cache[key] = arr
        return arr

    def elevation_at_bilinear(self, lon, lat):
        z = self.zoom
        x_tile, y_tile, x_f, y_f = lonlat_to_tile(lon, lat, z)
        x_pix_f = (x_f - x_tile) * 256.0
        y_pix_f = (y_f - y_tile) * 256.0
        x0, y0 = int(math.floor(x_pix_f)), int(math.floor(y_pix_f))
        dx, dy = x_pix_f - x0, y_pix_f - y0
        x0 = max(0, min(255, x0))
        y0 = max(0, min(255, y0))
        x1 = min(x0 + 1, 255)
        y1 = min(y0 + 1, 255)
        arr = self.get_tile_array(z, x_tile, y_tile)
        if arr is None:
            return None
        p00 = decode_terrain_rgb(*arr[y0, x0])
        p10 = decode_terrain_rgb(*arr[y0, x1])
        p01 = decode_terrain_rgb(*arr[y1, x0])
        p11 = decode_terrain_rgb(*arr[y1, x1])
        elev = (
            p00 * (1 - dx) * (1 - dy)
            + p10 * dx * (1 - dy)
            + p01 * (1 - dx) * dy
            + p11 * dx * dy
        )
        return float(elev)

def elevation_supersampled(lon, lat, cache: TerrainTileCache, lat_hint):
    # supersample radius ≈ 0.6 px, based on ground resolution at this latitude
    mpp = meters_per_pixel(lat_hint, cache.zoom)
    r_m = 0.6 * mpp
    # approximate lon/lat offsets (very small, ok for tiny r)
    dlat = (r_m / 111320.0)  # meters per degree latitude
    dlon = dlat / max(math.cos(math.radians(lat)), 1e-6)
    samples = [(lon, lat),
               (lon + dlon, lat),
               (lon - dlon, lat),
               (lon, lat + dlat),
               (lon, lat - dlat)]
    vals = []
    for lo, la in samples:
        e = cache.elevation_at_bilinear(lo, la)
        if e is not None and np.isfinite(e):
            vals.append(e)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))

def get_elevations_ll(points_ll, cache: TerrainTileCache, supersample_flag, lat_hint):
    elevs = []
    for lon, lat in points_ll:
        if supersample_flag:
            e = elevation_supersampled(lon, lat, cache, lat_hint)
        else:
            e = cache.elevation_at_bilinear(lon, lat)
            e = 0.0 if (e is None or not np.isfinite(e)) else float(e)
        elevs.append(e)
    return elevs

def smooth_two_stage(elevs, median_win, mean_win):
    arr = np.asarray(elevs, dtype=float)
    if median_win > 1:
        s = pd.Series(arr)
        arr = s.rolling(window=median_win, center=True, min_periods=1).median().to_numpy()
    if mean_win > 1:
        kernel = np.ones(mean_win) / float(mean_win)
        arr = np.convolve(arr, kernel, mode="same")
    return arr.tolist()

# =========================
# MAIN APP
# =========================

uploaded_file = st.file_uploader("Upload KML or KMZ file", type=["kml", "kmz"])

if uploaded_file:
    agms, parts_ll = parse_kml_kmz(uploaded_file)

    st.subheader("📌 AGM summary")
    st.text(f"Total AGMs found: {len(agms)}")
    st.subheader("📈 CENTERLINE parts")
    st.text(f"Found {len(parts_ll)} centerline part(s)")

    if len(parts_ll) == 0 or len(agms) < 2:
        st.warning("Missing CENTERLINE parts or insufficient AGM points.")
    else:
        # Center latitude for DEM/res clamp
        all_lats = [c[1] for p in parts_ll for c in p.coords]
        lat_hint = float(np.mean(all_lats)) if all_lats else 35.0

        # Densify each part geodesically at ~densify_step_m
        densified_parts = []
        for ls in parts_ll:
            coords = list(ls.coords)
            dens = densify_polyline_geodesic(coords, densify_step_m)
            if len(dens) >= 2:
                densified_parts.append(dens)

        tile_cache = TerrainTileCache(MAPBOX_TOKEN, zoom=mapbox_zoom)
        mpp = meters_per_pixel(lat_hint, mapbox_zoom)
        min_spacing = max(2.0, 1.5 * mpp)           # DEM-aware minimum
        spacing_used = max(float(requested_spacing_m), min_spacing)

        rows = []
        skipped = 0
        cumulative_miles = 0.0

        for i in range(len(agms) - 1):
            name1, pt1 = agms[i]
            name2, pt2 = agms[i + 1]
            p1 = (pt1.x, pt1.y)
            p2 = (pt2.x, pt2.y)

            # Pick best single part by minimal total snap distance (both within threshold)
            best = None
            for idx, dens in enumerate(densified_parts):
                i1, d1 = geodesic_snap_to_polyline(p1, dens)
                i2, d2 = geodesic_snap_to_polyline(p2, dens)
                if d1 <= snap_max_offset_m and d2 <= snap_max_offset_m:
                    tot = d1 + d2
                    if (best is None) or (tot < best[0]):
                        lo, hi = sorted((i1, i2))
                        seg = dens[lo:hi + 1]
                        best = (tot, idx, lo, hi, seg)
            if best is None:
                skipped += 1
                continue

            _, part_idx, lo, hi, seg_ll = best
            if len(seg_ll) < 2:
                skipped += 1
                continue

            # Resample along this lon/lat segment by geodesic arclength
            # (build
