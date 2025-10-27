# Terrain-Aware AGM Distance Calculator (Pure NumPy version)
# Streamlit app with progress bar + AGM snap-to-centerline + SP-filtering
# Requires: streamlit, pandas, numpy, shapely, pyproj, requests, zipfile36, Pillow

import streamlit as st
import pandas as pd
import numpy as np
import math, io, zipfile, xml.etree.ElementTree as ET, requests
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from pyproj import Transformer, Geod
from PIL import Image

# ---------------- Config ----------------
MAPBOX_TOKEN = st.secrets.get("MAPBOX_TOKEN", "")
MAPBOX_ZOOM = 14
RESAMPLE_BASE_M = 20
SPACING_M = 20
XY_SMOOTH_WINDOW_M = 40
ELEV_SMOOTH_WINDOW_M = 40
DZ_THRESHOLD_M = 0.5
FT_PER_M = 3.28084
MAX_SNAP_M = 80
GEOD = Geod(ellps="WGS84")
TERRAIN_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"

# ---------------- Terrain Cache ----------------
class TerrainCache:
    def __init__(self, token, zoom):
        self.t = token
        self.z = zoom
        self.c = {}

    def get(self, z, x, y):
        k = (z, x, y)
        if k in self.c:
            return self.c[k]
        r = requests.get(
            TERRAIN_URL.format(z=z, x=x, y=y),
            params={"access_token": self.t},
            timeout=15
        )
        if r.status_code != 200:
            print(f"[Mapbox error] {r.status_code} for tile {z}/{x}/{y}")
            return None
        arr = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), np.uint8)
        self.c[k] = arr
        return arr

    def elev(self, lon, lat):
        n = 2 ** self.z
        xt = (lon + 180.0) / 360.0 * n
        lat_rad = math.radians(lat)
        yt = (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n
        x, y = int(xt), int(yt)
        arr = self.get(self.z, x, y)
        if arr is None:
            return 0
        rx = int((xt - x) * 255)
        ry = int((yt - y) * 255)
        r, g, b = arr[min(ry, 255), min(rx, 255)]
        return -10000 + (r * 256 * 256 + g * 256 + b) * 0.1

# ---------------- Helpers ----------------
def smooth_np(a, win, dx):
    if len(a) < 3:
        return a
    n = max(3, int(round(win / dx)))
    n = n + 1 if n % 2 == 0 else n
    c = np.ones(n) / n
    return np.convolve(a, c, mode="same")

def tf_ll_to(crs):
    return Transformer.from_crs("epsg:4326", crs, always_xy=True)

def tf_to_ll(crs):
    return Transformer.from_crs(crs, "epsg:4326", always_xy=True)

def build_arrays(line):
    x, y = line.xy
    x, y = np.asarray(x), np.asarray(y)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    return x, y, s

def interp_xy(x, y, s, si):
    xi = np.interp(si, s, x)
    yi = np.interp(si, s, y)
    return xi, yi

def choose_part(parts, a1, a2, xf, max_snap_m):
    p1 = Point(xf.transform(a1.x, a1.y))
    p2 = Point(xf.transform(a2.x, a2.y))
    best = None
    best_d = 1e9
    for i, p in enumerate(parts):
        d1 = p.distance(p1)
        d2 = p.distance(p2)
        if d1 < max_snap_m and d2 < max_snap_m:
            d = d1 + d2
            if d < best_d:
                best_d = d
                best = (i, p.project(p1), p.project(p2))
    return best

def get_crs(parts):
    return "epsg:3857"

# ---------------- KML Parsing ----------------
def parse(file):
    name = file.name.lower()
    if name.endswith(".kmz"):
        with zipfile.ZipFile(file) as z:
            kml_name = [n for n in z.namelist() if n.endswith(".kml")][0]
            xml = z.read(kml_name)
    else:
        xml = file.read()
    root = ET.fromstring(xml)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    agms, parts = [], []

    # AGMs
    for pm in root.findall(".//kml:Folder[kml:name='AGMs']/kml:Placemark", ns):
        n = pm.findtext("kml:name", default="", namespaces=ns).strip()
        if n.startswith("SP"):
            continue
        coords = pm.find(".//kml:coordinates", ns)
        if coords is not None:
            vals = coords.text.strip().split(",")
            if len(vals) >= 2:
                agms.append((n, Point(float(vals[0]), float(vals[1]))))

    # Centerline
    for line in root.findall(".//kml:Folder[kml:name='CENTERLINE']//kml:LineString", ns):
        coords = line.find("kml:coordinates", ns)
        if coords is None:
            continue
        pts = []
        for c in coords.text.strip().split():
            parts_str = c.split(",")
            if len(parts_str) >= 2:
                pts.append((float(parts_str[0]), float(parts_str[1])))
        if pts:
            parts.append(LineString(pts))
    return agms, parts

# ---------------- Streamlit App ----------------
st.set_page_config("Terrain-Aware AGM Distance Checker", layout="wide")
st.title("📏 Terrain-Aware AGM Distance Checker")
u = st.file_uploader("Upload KML or KMZ", type=["kml", "kmz"])

if u:
    agms, parts = parse(u)
    st.text(f"{len(agms)} AGMs | {len(parts)} centerline part(s)")
    if not agms or not parts:
        st.warning("Need both AGMs + centerline.")
        st.stop()

    crs = get_crs(parts)
    xf_fwd, xf_inv = tf_ll_to(crs), tf_to_ll(crs)
    parts_m = [LineString(xf_fwd.transform(*p.xy)) for p in parts if p.length > 0]
    cache = TerrainCache(MAPBOX_TOKEN, MAPBOX_ZOOM)
    rows, cum_mi, skipped = [], 0.0, 0

    progress = st.progress(0)
    status = st.empty()
    total = len(agms) - 1

    for i in range(total):
        n1, a1 = agms[i]; n2, a2 = agms[i + 1]
        status.text(f"Calculating {n1} → {n2} ({i + 1}/{total})...")
        progress.progress((i + 1) / total)

        res = choose_part(parts_m, a1, a2, xf_fwd, MAX_SNAP_M)
        if res is None:
            skipped += 1
            continue
        idx, s1, s2 = res
        part = parts_m[idx]
        x, y, s = build_arrays(part)
        s_lo, s_hi = sorted((float(s1), float(s2)))
        if s_hi - s_lo <= 0:
            skipped += 1
            continue

        si = np.arange(s_lo, s_hi, RESAMPLE_BASE_M)
        if si.size == 0 or si[-1] < s_hi:
            si = np.append(si, s_hi)
        xi, yi = interp_xy(x, y, s, si)
        xi_s = smooth_np(xi, XY_SMOOTH_WINDOW_M, RESAMPLE_BASE_M)
        yi_s = smooth_np(yi, XY_SMOOTH_WINDOW_M, RESAMPLE_BASE_M)
        d = np.hypot(np.diff(xi_s), np.diff(yi_s))
        s2 = np.concatenate([[0.0], np.cumsum(d)])
        L2 = s2[-1]
        if L2 <= 0:
            skipped += 1
            continue

        sp = np.arange(0.0, L2, SPACING_M)
        if sp.size == 0 or sp[-1] < L2:
            sp = np.append(sp, L2)
        X, Y = np.interp(sp, s2, xi_s), np.interp(sp, s2, yi_s)
        lons, lats = xf_inv.transform(X.tolist(), Y.tolist())
        pts = list(zip(lons, lats))

        elev = []
        for lo, la in pts:
            try:
                elev.append(cache.elev(lo, la))
            except Exception:
                elev.append(0)
        elev = smooth_np(np.asarray(elev, float), ELEV_SMOOTH_WINDOW_M, SPACING_M)

        dist_m = 0.0
        for j in range(len(pts) - 1):
            lon1, lat1 = pts[j]
            lon2, lat2 = pts[j + 1]
            _, _, dxy = GEOD.inv(lon1, lat1, lon2, lat2)
            dz = elev[j + 1] - elev[j]
            if abs(dz) < DZ_THRESHOLD_M:
                dz = 0.0
            dist_m += math.hypot(dxy, dz)

        ft = dist_m * FT_PER_M
        mi = ft / 5280.0
        cum_mi += mi
        rows.append({
            "From AGM": n1,
            "To AGM": n2,
            "Distance (feet)": round(ft, 2),
            "Distance (miles)": round(mi, 6),
            "Cumulative (miles)": round(cum_mi, 6)
        })

    df = pd.DataFrame(rows)
    status.text("✅ Complete!")
    progress.progress(1.0)
    st.subheader("📊 Distance table")
    st.dataframe(df, use_container_width=True)
    st.text(f"Skipped segments: {skipped}")
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                       "terrain_distances.csv", "text/csv")
