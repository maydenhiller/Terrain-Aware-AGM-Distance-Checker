# app.py
# Terrain-Aware AGM Distance Calculator (FAST + Parser Unchanged + Streamlit-Safe)

import io
import math
import time
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from shapely.geometry import LineString, Point
from pyproj import Geod, Transformer

# ---------------- CONFIG ----------------
st.set_page_config("Terrain AGM Distance — FAST", layout="wide")
st.title("📏 Terrain-Aware AGM Distance Calculator — FAST")

MAPBOX_TOKEN = st.secrets.get("MAPBOX_TOKEN", "")
MAPBOX_ZOOM = 14
RESAMPLE_M = 10
SMOOTH_WINDOW = 40
DZ_THRESH = 0.5
FT_PER_M = 3.28084
GEOD = Geod(ellps="WGS84")
MAX_SNAP_M = 80


# ---------------- TERRAIN CACHE ----------------
class TerrainCache:
    """Mapbox Terrain-RGB tile cache with vectorized bilinear interpolation"""
    def __init__(self, token: str, zoom: int):
        self.token = token
        self.zoom = int(zoom)
        self.tiles = {}

    @staticmethod
    def decode_rgb_arr(rgb_arr: np.ndarray) -> np.ndarray:
        r = rgb_arr[..., 0].astype(np.int64)
        g = rgb_arr[..., 1].astype(np.int64)
        b = rgb_arr[..., 2].astype(np.int64)
        return -10000.0 + (r * 256 * 256 + g * 256 + b) * 0.1

    def fetch_tile(self, z: int, x: int, y: int):
        key = (z, x, y)
        if key in self.tiles:
            return self.tiles[key]
        url = f"https://api.mapbox.com/v1/mapbox/terrain-rgb/{z}/{x}/{y}.pngraw"
        try:
            r = requests.get(url, params={"access_token": self.token}, timeout=10)
        except Exception as e:
            st.warning(f"⚠ Network issue fetching tile {z}/{x}/{y}: {e}")
            return None
        if r.status_code == 401:
            st.error("❌ Mapbox 401 Unauthorized — check your MAPBOX_TOKEN.")
            st.stop()
        if r.status_code == 403:
            st.error("❌ Mapbox 403 Forbidden — invalid or restricted token.")
            st.stop()
        if r.status_code == 429:
            time.sleep(2)
            return None
        if r.status_code != 200:
            print(f"[Mapbox {r.status_code}] {z}/{x}/{y}")
            return None
        arr = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=np.uint8)
        self.tiles[key] = arr
        return arr

    def elevations_bulk(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        """Vectorized elevation lookup using a single fetch per tile."""
        z = self.zoom
        n = float(2 ** z)
        xt = (lons + 180.0) / 360.0 * n
        lat_r = np.radians(lats)
        yt = (1.0 - np.log(np.tan(lat_r) + 1.0 / np.cos(lat_r)) / math.pi) / 2.0 * n

        x_tile = np.floor(xt).astype(np.int64)
        y_tile = np.floor(yt).astype(np.int64)
        xp = (xt - x_tile) * 255.0
        yp = (yt - y_tile) * 255.0

        out = np.zeros_like(lons, dtype=np.float64)
        tile_groups = {}
        for i, (xx, yy) in enumerate(zip(x_tile, y_tile)):
            tile_groups.setdefault((z, int(xx), int(yy)), []).append(i)

        for key, idxs in tile_groups.items():
            arr = self.fetch_tile(*key)
            if arr is None:
                continue
            idxs = np.array(idxs)
            x0 = np.clip(xp[idxs].astype(int), 0, 255)
            y0 = np.clip(yp[idxs].astype(int), 0, 255)
            x1 = np.clip(x0 + 1, 0, 255)
            y1 = np.clip(y0 + 1, 0, 255)
            dx = xp[idxs] - x0
            dy = yp[idxs] - y0

            e00 = self.decode_rgb_arr(arr[y0, x0])
            e10 = self.decode_rgb_arr(arr[y0, x1])
            e01 = self.decode_rgb_arr(arr[y1, x0])
            e11 = self.decode_rgb_arr(arr[y1, x1])

            out[idxs] = (e00 * (1 - dx) * (1 - dy)
                         + e10 * dx * (1 - dy)
                         + e01 * (1 - dx) * dy
                         + e11 * dx * dy)
        return out


# ---------------- HELPERS ----------------
def smooth(a, window_m, spacing_m):
    if len(a) < 3:
        return a
    n = max(3, int(round(window_m / max(spacing_m, 1e-6))))
    if n % 2 == 0:
        n += 1
    kernel = np.ones(n) / n
    return np.convolve(a, kernel, mode="same")


def strip_ns(e):
    e.tag = e.tag.split("}", 1)[-1]
    for k in list(e.attrib):
        nk = k.split("}", 1)[-1]
        if nk != k:
            e.attrib[nk] = e.attrib.pop(k)
    for c in e:
        strip_ns(c)


# ---------------- ORIGINAL PARSER (UNCHANGED) ----------------
def parse_kml_kmz(uploaded_file):
    if uploaded_file.name.lower().endswith(".kmz"):
        with zipfile.ZipFile(uploaded_file) as zf:
            kml_name = next((n for n in zf.namelist() if n.lower().endswith(".kml")), None)
            if not kml_name:
                return [], []
            data = zf.read(kml_name)
    else:
        data = uploaded_file.read()
    root = ET.fromstring(data)
    strip_ns(root)

    agms = []
    centerlines = []
    for folder in root.findall(".//Folder"):
        n = folder.find("name")
        if n is None or not n.text:
            continue
        fname = n.text.strip()
        if fname == "AGMs":
            for pm in folder.findall(".//Placemark"):
                nm_el = pm.find("name")
                if nm_el is None or not nm_el.text:
                    continue
                nm = nm_el.text.strip()
                if nm.upper().startswith("SP"):
                    continue
                coords_el = pm.find(".//Point/coordinates")
                if coords_el is None or not coords_el.text:
                    continue
                lon, lat, *_ = map(float, coords_el.text.strip().split(","))
                agms.append((nm, Point(lon, lat)))
        if fname == "CENTERLINE":
            for pm in folder.findall(".//Placemark"):
                coords_el = pm.find(".//LineString/coordinates")
                if coords_el is None or not coords_el.text:
                    continue
                pts = [tuple(map(float, c.split(",")[:2])) for c in coords_el.text.strip().split()]
                if len(pts) >= 2:
                    centerlines.append(LineString(pts))
    agms.sort(key=lambda p: int(''.join(filter(str.isdigit, p[0]))) if any(ch.isdigit() for ch in p[0]) else -1)
    return agms, centerlines


def build_local_meter_crs(line_ll):
    xs, ys = zip(*line_ll.coords)
    lon0, lat0 = np.mean(xs), np.mean(ys)
    proj = f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m"
    to_m = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    to_ll = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)
    return to_m, to_ll


def snap_offset_m(line_m, pt_m):
    s = line_m.project(pt_m)
    sp = line_m.interpolate(s)
    return math.hypot(pt_m.x - sp.x, pt_m.y - sp.y)


# ---------------- MAIN ----------------
u = st.file_uploader("Upload KML or KMZ", type=["kml", "kmz"])
if not u:
    st.stop()

agms, lines = parse_kml_kmz(u)
st.text(f"{len(agms)} AGMs | {len(lines)} centerline part(s)")
if not agms or not lines:
    st.warning("Need both AGMs and CENTERLINE.")
    st.stop()

line_ll = lines[0]
to_m, to_ll = build_local_meter_crs(line_ll)
X_m, Y_m = to_m.transform(*zip(*line_ll.coords))
line_m = LineString(zip(X_m, Y_m))

if "cache" not in st.session_state:
    st.session_state.cache = TerrainCache(MAPBOX_TOKEN, MAPBOX_ZOOM)
cache = st.session_state.cache

rows = []
cum_mi = 0.0
bar = st.progress(0)
status = st.empty()
total = len(agms) - 1

for i in range(total):
    n1, a1 = agms[i]
    n2, a2 = agms[i + 1]
    status.text(f"⏱ Calculating {n1} → {n2} ({i + 1}/{total}) …")
    bar.progress((i + 1) / total)

    p1_m = Point(*to_m.transform(a1.x, a1.y))
    p2_m = Point(*to_m.transform(a2.x, a2.y))
    if snap_offset_m(line_m, p1_m) > MAX_SNAP_M or snap_offset_m(line_m, p2_m) > MAX_SNAP_M:
        continue

    s1, s2 = sorted((line_m.project(p1_m), line_m.project(p2_m)))
    si = np.arange(s1, s2, RESAMPLE_M)
    if si[-1] < s2:
        si = np.append(si, s2)
    pts_m = [line_m.interpolate(s) for s in si]
    pts_ll = np.array([to_ll.transform(p.x, p.y) for p in pts_m])

    elev = cache.elevations_bulk(pts_ll[:, 0], pts_ll[:, 1])
    elev = smooth(elev, SMOOTH_WINDOW, RESAMPLE_M)

    dist_m = 0.0
    for j in range(len(pts_ll) - 1):
        lon1, lat1 = pts_ll[j]
        lon2, lat2 = pts_ll[j + 1]
        _, _, dxy = GEOD.inv(lon1, lat1, lon2, lat2)
        dz = elev[j + 1] - elev[j]
        if abs(dz) < DZ_THRESH:
            dz = 0
        dist_m += math.hypot(dxy, dz)

    feet = dist_m * FT_PER_M
    miles = feet / 5280
    cum_mi += miles
    rows.append({
        "From AGM": n1,
        "To AGM": n2,
        "Distance (feet)": round(feet, 2),
        "Distance (miles)": round(miles, 6),
        "Cumulative (miles)": round(cum_mi, 6)
    })

status.text("✅ Complete.")
bar.progress(1.0)
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)
st.download_button("Download CSV", df.to_csv(index=False).encode(), "terrain_distances.csv", "text/csv")
