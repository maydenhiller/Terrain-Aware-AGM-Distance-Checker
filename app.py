# app.py
# Terrain-Aware AGM Distance Calculator
# - Full app with progress bar
# - Accurate 3D distance using geodesic XY + Terrain-RGB elevation
# - Centerline slicing done in local meter CRS (no degree-scale error)
# - AGM names starting with "SP" are ignored
# - Snaps each AGM to nearest point on centerline (max offset configurable)
# - Uses Mapbox v1 Terrain-RGB with clear 401/403/429 handling
#
# Secrets:
#   In Streamlit Cloud (or .streamlit/secrets.toml) set:
#     MAPBOX_TOKEN = "pk.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

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
from pyproj import Geod, Transformer
from shapely.geometry import LineString, Point

# ---------------------------- CONFIG ----------------------------
st.set_page_config("Terrain AGM Distance", layout="wide")
st.title("📏 Terrain-Aware AGM Distance Calculator")

MAPBOX_TOKEN = st.secrets.get("MAPBOX_TOKEN", "")
MAPBOX_ZOOM = 14  # 14–17 are reasonable. Higher zoom = more tiles.
RESAMPLE_M = 10   # step along centerline when creating sampling points (meters)
SMOOTH_WINDOW = 40  # moving-average window (meters) for smoothing XY/elevation
DZ_THRESH = 0.5   # ignore sub-50 cm vertical jitter
FT_PER_M = 3.28084
GEOD = Geod(ellps="WGS84")
MAX_SNAP_M = 80   # maximum AGM snap distance to the centerline (meters)

# ---------------------------- TERRAIN ----------------------------
class TerrainCache:
    """Mapbox Terrain-RGB tile cache (v1 endpoint) with basic error handling."""
    def __init__(self, token: str, zoom: int):
        self.token = token
        self.zoom = int(zoom)
        self.cache = {}

    @staticmethod
    def _decode_rgb(r, g, b) -> float:
        # E(m) = -10000 + (R*256^2 + G*256 + B) * 0.1
        return -10000.0 + (int(r) * 256 * 256 + int(g) * 256 + int(b)) * 0.1

    def _fetch(self, z: int, x: int, y: int):
        key = (z, x, y)
        if key in self.cache:
            return self.cache[key]

        url = f"https://api.mapbox.com/v1/mapbox/terrain-rgb/{z}/{x}/{y}.pngraw"
        try:
            r = requests.get(url, params={"access_token": self.token}, timeout=12)
        except Exception as e:
            print(f"[Mapbox fetch error] {e}")
            return None

        if r.status_code == 401:
            st.error("❌ Mapbox 401 Unauthorized — check MAPBOX_TOKEN and Tilesets scope.")
            st.stop()
        if r.status_code == 403:
            st.error("❌ Mapbox 403 Forbidden — token lacks tileset access or has domain restrictions.")
            st.stop()
        if r.status_code == 429:
            # Simple backoff; caller will try again later if needed
            print("⚠ Mapbox 429 rate limit — backing off 2s")
            time.sleep(2)
            return None
        if r.status_code != 200:
            print(f"[Mapbox HTTP {r.status_code}] for tile {z}/{x}/{y}")
            return None

        arr = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=np.uint8)
        self.cache[key] = arr
        return arr

    def elev(self, lon: float, lat: float) -> float:
        # Web Mercator tile math
        n = 2 ** self.zoom
        xt = (lon + 180.0) / 360.0 * n
        lat_r = math.radians(lat)
        yt = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
        x, y = int(xt), int(yt)

        arr = self._fetch(self.zoom, x, y)
        if arr is None:
            return 0.0

        # Bilinear sample within 256×256 tile
        xp = (xt - x) * 255.0
        yp = (yt - y) * 255.0
        x0 = max(0, min(255, int(xp)))
        y0 = max(0, min(255, int(yp)))
        x1 = min(255, x0 + 1)
        y1 = min(255, y0 + 1)
        dx = xp - x0
        dy = yp - y0

        r00, g00, b00 = arr[y0, x0]
        r10, g10, b10 = arr[y0, x1]
        r01, g01, b01 = arr[y1, x0]
        r11, g11, b11 = arr[y1, x1]

        e00 = self._decode_rgb(r00, g00, b00)
        e10 = self._decode_rgb(r10, g10, b10)
        e01 = self._decode_rgb(r01, g01, b01)
        e11 = self._decode_rgb(r11, g11, b11)
        return float(
            e00 * (1 - dx) * (1 - dy) +
            e10 * dx * (1 - dy) +
            e01 * (1 - dx) * dy +
            e11 * dx * dy
        )

# ---------------------------- HELPERS ----------------------------
def smooth_moving_average(a: np.ndarray, window_m: float, spacing_m: float) -> np.ndarray:
    if len(a) < 3:
        return a
    n = max(3, int(round(window_m / max(spacing_m, 1e-6))))
    if n % 2 == 0:
        n += 1
    k = np.ones(n, dtype=float) / n
    return np.convolve(a, k, mode="same")

def strip_ns(e: ET.Element):
    """Remove XML namespaces recursively so we can use simple tag names if needed."""
    e.tag = e.tag.split("}", 1)[-1]
    for k in list(e.attrib):
        nk = k.split("}", 1)[-1]
        if nk != k:
            e.attrib[nk] = e.attrib.pop(k)
    for c in e:
        strip_ns(c)

def parse_kml_kmz(uploaded_file):
    """Parse AGMs and CENTERLINE from KML/KMZ with folder names exactly 'AGMs' and 'CENTERLINE'."""
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

    # AGMs folder
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
                    continue  # ignore SP-prefixed
                coords_el = pm.find(".//Point/coordinates")
                if coords_el is None or not coords_el.text:
                    continue
                try:
                    lon, lat, *_ = map(float, coords_el.text.strip().split(","))
                    agms.append((nm, Point(lon, lat)))
                except Exception:
                    pass

        if fname == "CENTERLINE":
            for pm in folder.findall(".//Placemark"):
                coords_el = pm.find(".//LineString/coordinates")
                if coords_el is None or not coords_el.text:
                    continue
                pts = []
                for token in coords_el.text.strip().split():
                    try:
                        lon, lat, *_ = map(float, token.split(","))
                        pts.append((lon, lat))
                    except Exception:
                        pass
                if len(pts) >= 2:
                    centerlines.append(LineString(pts))

    agms.sort(key=lambda p: (int(''.join(filter(str.isdigit, p[0]))) if any(ch.isdigit() for ch in p[0]) else -1,
                             ''.join(filter(str.isalpha, p[0]))))
    return agms, centerlines

def build_local_meter_crs(line_ll: LineString):
    """Build a local Transverse Mercator centered on the line for true-meter s-values."""
    xs = [c[0] for c in line_ll.coords]
    ys = [c[1] for c in line_ll.coords]
    lon0 = float(np.mean(xs))
    lat0 = float(np.mean(ys))
    # Local TMerc centered on (lon0,lat0)
    proj_str = f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    to_m = Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)
    to_ll = Transformer.from_crs(proj_str, "EPSG:4326", always_xy=True)
    return to_m, to_ll

def snap_offset_m(line_m: LineString, pt_m: Point) -> float:
    s = line_m.project(pt_m)
    sp = line_m.interpolate(s)
    return float(math.hypot(pt_m.x - sp.x, pt_m.y - sp.y))

# ---------------------------- UI + COMPUTE ----------------------------
u = st.file_uploader("Upload KML or KMZ", type=["kml", "kmz"])
if not u:
    st.stop()

agms, lines = parse_kml_kmz(u)
st.text(f"{len(agms)} AGMs | {len(lines)} centerline part(s)")
if not agms or not lines:
    st.warning("Need both AGMs and CENTERLINE.")
    st.stop()

# Use the first centerline (or merge logic if you need multiple)
line_ll = lines[0]

# Local meter CRS for correct project()/interpolate() distances
to_m, to_ll = build_local_meter_crs(line_ll)

# Transform the centerline to meters
X_m, Y_m = to_m.transform(*zip(*line_ll.coords))
line_m = LineString(list(zip(X_m, Y_m)))

cache = TerrainCache(MAPBOX_TOKEN, MAPBOX_ZOOM)

rows = []
cum_mi = 0.0
bar = st.progress(0)
status = st.empty()
total = max(0, len(agms) - 1)

for i in range(total):
    n1, a1 = agms[i]
    n2, a2 = agms[i + 1]
    status.text(f"⏱ Calculating {n1} → {n2} ({i + 1}/{total}) …")
    bar.progress((i + 1) / max(1, total))

    # Snap AGMs to centerline (in meters) and enforce max offset
    p1_m = Point(*to_m.transform(a1.x, a1.y))
    p2_m = Point(*to_m.transform(a2.x, a2.y))

    off1 = snap_offset_m(line_m, p1_m)
    off2 = snap_offset_m(line_m, p2_m)
    if off1 > MAX_SNAP_M or off2 > MAX_SNAP_M:
        # Too far from centerline — skip
        print(f"Skip {n1}->{n2}: off1={off1:.2f}m off2={off2:.2f}m")
        continue

    s1 = line_m.project(p1_m)
    s2 = line_m.project(p2_m)
    s_lo, s_hi = sorted((s1, s2))
    if s_hi - s_lo <= 0:
        continue

    # Sample along line in meters
    si = np.arange(s_lo, s_hi, RESAMPLE_M)
    if si.size == 0 or si[-1] < s_hi:
        si = np.append(si, s_hi)

    pts_m = [line_m.interpolate(s) for s in si]
    # Convert back to lon/lat for geodesic + elevation
    pts_ll = [to_ll.transform(p.x, p.y) for p in pts_m]

    # Elevation per point (smoothed)
    elev = [cache.elev(lon, lat) for lon, lat in pts_ll]
    elev = smooth_moving_average(np.asarray(elev, dtype=float), SMOOTH_WINDOW, RESAMPLE_M)

    # 3D distance accumulation: GEOD.inv for horizontal + dz
    dist_m = 0.0
    for j in range(len(pts_ll) - 1):
        lon1, lat1 = pts_ll[j]
        lon2, lat2 = pts_ll[j + 1]
        _, _, dxy = GEOD.inv(lon1, lat1, lon2, lat2)  # meters
        dz = elev[j + 1] - elev[j]
        if abs(dz) < DZ_THRESH:
            dz = 0.0
        dist_m += math.hypot(dxy, dz)

    feet = dist_m * FT_PER_M
    miles = feet / 5280.0
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
st.subheader("📊 Distance table")
st.dataframe(df, use_container_width=True)
st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                   "terrain_distances.csv", "text/csv")
