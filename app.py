# app.py
# Terrain-Aware AGM Distance Calculator — Geodesic 3D (tailored to your KML: Folders "AGMs" + "CENTERLINE")
# Requirements: NO CHANGE to requirements.txt

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
from pyproj import CRS, Transformer, Geod

# ───────────────────────────────── CONFIG
st.set_page_config(page_title="Terrain-Aware AGM Distance Calculator", layout="wide")
st.title("Terrain-Aware AGM Distance Calculator (Geodesic 3D)")

MAPBOX_TOKEN = st.secrets["mapbox"]["token"]
TERRAIN_TILE_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"
GEOD = Geod(ellps="WGS84")  # true ellipsoid distances (matches Google Earth)

FT_PER_M = 3.28084
MI_PER_FT = 1.0 / 5280.0

with st.sidebar:
    st.header("Settings")
    mapbox_zoom = st.slider("Terrain tile zoom", 15, 17, 17)
    spacing_m = st.slider("Sample spacing along path (m)", 0.5, 10.0, 1.0, 0.5)
    smooth_window = st.slider("Elevation smoothing window (samples)", 1, 25, 7, 2)
    simplify_tolerance_m = st.slider("Simplify centerline (m)", 0.0, 5.0, 0.0, 0.5)
    max_snap_offset_m = st.slider("Max AGM snap offset to centerline (m)", 5.0, 150.0, 80.0, 5.0)
    st.caption("Use zoom 17, spacing 1 m, smoothing ≈ 7 for best accuracy.")

# ───────────────────────────────── PARSER (tailored: Folders named AGMs and CENTERLINE)
def agm_sort_key(name_geom):
    name = name_geom[0]
    digits = ''.join(filter(str.isdigit, name))
    base = int(digits) if digits else -1
    suffix = ''.join(filter(str.isalpha, name)).upper()
    return (base, suffix)

def parse_kml_kmz(uploaded):
    """Parse AGMs from Folder 'AGMs' (Points) and a polyline from Folder 'CENTERLINE' (LineString)."""
    if uploaded.name.endswith(".kmz"):
        with zipfile.ZipFile(uploaded) as zf:
            kml_file = next((f for f in zf.namelist() if f.endswith(".kml")), None)
            if not kml_file:
                return [], []
            with zf.open(kml_file) as f:
                kml_data = f.read()
    else:
        kml_data = uploaded.read()

    root = ET.fromstring(kml_data)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    agms, parts = [], []

    for folder in root.findall(".//kml:Folder", ns):
        name_el = folder.find("kml:name", ns)
        if not name_el or not name_el.text:
            continue
        fname = name_el.text.strip().upper()

        if fname == "AGMS":
            for pm in folder.findall("kml:Placemark", ns):
                nm = pm.find("kml:name", ns)
                cr = pm.find(".//kml:coordinates", ns)
                if not nm or not cr or not cr.text:
                    continue
                try:
                    lon, lat, *_ = map(float, cr.text.strip().split(","))
                    agms.append((nm.text.strip(), Point(lon, lat)))
                except:
                    pass

        elif fname == "CENTERLINE":
            for pm in folder.findall("kml:Placemark", ns):
                cr = pm.find(".//kml:coordinates", ns)
                if not cr or not cr.text:
                    continue
                pts = []
                for pair in cr.text.strip().split():
                    try:
                        lon, lat, *_ = map(float, pair.split(","))
                        pts.append((lon, lat))
                    except:
                        pass
                if len(pts) >= 2:
                    parts.append(LineString(pts))

    agms.sort(key=agm_sort_key)
    return agms, parts

# ───────────────────────────────── CRS helpers
def get_local_utm_crs(lines_ll):
    xs, ys = [], []
    for ls in lines_ll:
        xs += [c[0] for c in ls.coords]
        ys += [c[1] for c in ls.coords]
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    zone = int((cx + 180.0) / 6.0) + 1
    epsg = 32600 + zone if cy >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)

def xf_ll_to(crs): return Transformer.from_crs("EPSG:4326", crs, always_xy=True)
def xf_to_ll(crs): return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
def tf_linestring(ls, xf):
    xs, ys = zip(*ls.coords); X, Y = xf.transform(xs, ys)
    return LineString(zip(X, Y))
def tf_point(pt, xf):
    x, y = xf.transform(pt.x, pt.y)
    return Point(x, y)

# ───────────────────────────────── Terrain-RGB (overflow-safe bilinear)
def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return int(x), int(y), x, y

def decode_terrain_rgb(r, g, b):
    # Cast to Python int to avoid uint8 overflow
    R, G, B = int(r), int(g), int(b)
    return -10000.0 + ((R * 256.0 * 256.0) + (G * 256.0) + B) * 0.1

class TerrainCache:
    def __init__(self, token, zoom=17):
        self.t = token
        self.z = int(zoom)
        self.c = {}

    def get(self, z, x, y):
        key = (int(z), int(x), int(y))
        if key in self.c:
            return self.c[key]
        url = TERRAIN_TILE_URL.format(z=int(z), x=int(x), y=int(y))
        r = requests.get(url, params={"access_token": self.t}, timeout=20)
        if r.status_code != 200:
            return None
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        self.c[key] = arr
        return arr

    def elev(self, lon, lat):
        # Clamp to Web Mercator safe bounds & sample
        lon = max(-179.999, min(179.999, float(lon)))
        lat = max(-85.0, min(85.0, float(lat)))
        z = self.z
        xt, yt, xf, yf = lonlat_to_tile(lon, lat, z)
        xp, yp = (xf - xt) * 256.0, (yf - yt) * 256.0
        x0, y0 = int(math.floor(xp)), int(math.floor(yp))
        dx, dy = float(xp - x0), float(yp - y0)
        x0, y0 = max(0, min(255, x0)), max(0, min(255, y0))
        x1, y1 = min(x0 + 1, 255), min(y0 + 1, 255)
        arr = self.get(z, xt, yt)
        if arr is None:
            return 0.0
        p00 = decode_terrain_rgb(arr[y0, x0, 0], arr[y0, x0, 1], arr[y0, x0, 2])
        p10 = decode_terrain_rgb(arr[y0, x1, 0], arr[y0, x1, 1], arr[y0, x1, 2])
        p01 = decode_terrain_rgb(arr[y1, x0, 0], arr[y1, x0, 1], arr[y1, x0, 2])
        p11 = decode_terrain_rgb(arr[y1, x1, 0], arr[y1, x1, 1], arr[y1, x1, 2])
        return float(
            p00 * (1 - dx) * (1 - dy)
            + p10 * dx * (1 - dy)
            + p01 * (1 - dx) * dy
            + p11 * dx * dy
        )

def smooth(vals, k):
    if k <= 1:
        return [float(v) for v in vals]
    arr = np.asarray(vals, dtype=float)
    kernel = np.ones(int(k)) / float(int(k))
    return np.convolve(arr, kernel, "same").tolist()

# ───────────────────────────────── Linear-referencing helpers (metric)
def build_arrays(ls_m: LineString):
    c = list(ls_m.coords)
    xs = np.array([p[0] for p in c], dtype=float)
    ys = np.array([p[1] for p in c], dtype=float)
    d = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate([[0.0], np.cumsum(d)])
    return xs, ys, cum

def interp_on_poly(xs, ys, cum, s):
    if s <= 0:
        return xs[0], ys[0]
    if s >= cum[-1]:
        return xs[-1], ys[-1]
    i = int(np.searchsorted(cum, s) - 1)
    i = max(0, min(i, len(xs) - 2))
    seg = cum[i + 1] - cum[i]
    t = (s - cum[i]) / seg if seg > 0 else 0.0
    return xs[i] + t * (xs[i + 1] - xs[i]), ys[i] + t * (ys[i + 1] - ys[i])

def sample_between(ls_m, s1, s2, spacing):
    xs, ys, cum = build_arrays(ls_m)
    s_lo, s_hi = sorted((float(s1), float(s2)))
    L = abs(s_hi - s_lo)
    if L <= 0:
        return np.array([]), np.array([])
    steps = np.arange(0.0, L, float(spacing))
    if steps.size == 0 or steps[-1] < L:
        steps = np.append(steps, L)
    out = [interp_on_poly(xs, ys, cum, s_lo + ds) for ds in steps]
    X, Y = zip(*out)
    return np.array(X, dtype=float), np.array(Y, dtype=float)

# ───────────────────────────────── AGM pair → pick one correct centerline part
def choose_part(parts_m, pt1_ll, pt2_ll, xf_ll_to_m, max_off):
    p1m, p2m = tf_point(pt1_ll, xf_ll_to_m), tf_point(pt2_ll, xf_ll_to_m)
    best = None
    for i, p in enumerate(parts_m):
        s1, s2 = p.project(p1m), p.project(p2m)
        sp1, sp2 = p.interpolate(s1), p.interpolate(s2)
        o1 = ((p1m.x - sp1.x) ** 2 + (p1m.y - sp1.y) ** 2) ** 0.5
        o2 = ((p2m.x - sp2.x) ** 2 + (p2m.y - sp2.y) ** 2) ** 0.5
        if o1 <= max_off and o2 <= max_off:
            tot = o1 + o2
            if best is None or tot < best[0]:
                best = (tot, i, s1, s2)
    return (None, None, None) if best is None else (best[1], best[2], best[3])

# ───────────────────────────────── MAIN
uploaded = st.file_uploader("Upload KML or KMZ file", type=["kml", "kmz"])
if uploaded:
    agms, parts_ll = parse_kml_kmz(uploaded)
    st.text(f"{len(agms)} AGMs | {len(parts_ll)} centerline part(s)")
    if not parts_ll or len(agms) < 2:
        st.warning("Need both AGMs + centerline.")
        st.stop()

    # metric CRS for snapping + linear reference
    try:
        crs = get_local_utm_crs(parts_ll)
    except Exception:
        crs = CRS.from_epsg(5070)  # fallback (CONUS Albers)
    xf_fwd, xf_inv = xf_ll_to(crs), xf_to_ll(crs)

    # transform/simplify parts -> metric
    parts_m = []
    for p in parts_ll:
        pm = tf_linestring(p, xf_fwd)
        if simplify_tolerance_m > 0.0:
            pm = pm.simplify(float(simplify_tolerance_m), preserve_topology=False)
        if pm.length > 0 and len(pm.coords) >= 2:
            parts_m.append(pm)

    if not parts_m:
        st.warning("All centerline parts were degenerate after transform/simplify.")
        st.stop()

    cache = TerrainCache(MAPBOX_TOKEN, zoom=mapbox_zoom)

    rows = []
    cumulative_miles = 0.0
    skipped = 0

    # compute distances between consecutive AGMs (sorted by name)
    for i in range(len(agms) - 1):
        n1, a1 = agms[i]
        n2, a2 = agms[i + 1]

        part_idx, s1, s2 = choose_part(parts_m, a1, a2, xf_fwd, max_snap_offset_m)
        if part_idx is None:
            skipped += 1
            continue

        part = parts_m[part_idx]
        Xs, Ys = sample_between(part, s1, s2, spacing_m)
        if Xs.size < 2:
            skipped += 1
            continue

        # back to lon/lat for geodesic + elevation
        lons, lats = xf_inv.transform(Xs.tolist(), Ys.tolist())
        pts = list(zip(lons, lats))

        elev = [cache.elev(lo, la) for lo, la in pts]
        elev = smooth(elev, int(smooth_window))

        # 3D accumulation: sqrt( geodesic_dxy^2 + dz^2 )
        dist_m = 0.0
        for j in range(len(pts) - 1):
            lon1, lat1 = pts[j]
            lon2, lat2 = pts[j + 1]
            _, _, dxy = GEOD.inv(lon1, lat1, lon2, lat2)
            dz = float(elev[j + 1] - elev[j])
            dist_m += math.sqrt(dxy * dxy + dz * dz)

        dist_ft = dist_m * FT_PER_M
        dist_mi = dist_ft * MI_PER_FT
        cumulative_miles += dist_mi

        rows.append({
            "From AGM": n1,
            "To AGM": n2,
            "Distance (feet)": round(dist_ft, 2),
            "Distance (miles)": round(dist_mi, 6),
            "Cumulative (miles)": round(cumulative_miles, 6),
            "Centerline part #": int(part_idx)
        })

    st.subheader("📊 Distance table")
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    st.text(f"Skipped segments: {skipped}")

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "terrain_distances.csv",
        "text/csv"
    )
