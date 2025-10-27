 # Terrain-Aware AGM Distance Calculator — Stable Final Build
# Hard-targets <Folder><name>AGMs</name> and <Folder><name>CENTERLINE</name>
# 3D geodesic distances using Mapbox Terrain-RGB

import math, io, zipfile, requests, xml.etree.ElementTree as ET
import streamlit as st, pandas as pd, numpy as np
from PIL import Image
from shapely.geometry import Point, LineString
from pyproj import CRS, Transformer, Geod

st.set_page_config(page_title="Terrain-Aware AGM Distance Calculator", layout="wide")
st.title("Terrain-Aware AGM Distance Calculator (Final Geodesic 3D)")

MAPBOX_TOKEN = st.secrets["mapbox"]["token"]
TERRAIN_TILE_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"
GEOD = Geod(ellps="WGS84")
FT_PER_M, MI_PER_FT = 3.28084, 1 / 5280.0

with st.sidebar:
    st.header("Settings")
    mapbox_zoom = st.slider("Terrain tile zoom", 15, 17, 17)
    spacing_m = st.slider("Sample spacing (m)", 0.5, 10.0, 1.0, 0.5)
    smooth_window = st.slider("Elevation smoothing window", 1, 25, 7, 2)
    simplify_tolerance_m = st.slider("Simplify centerline (m)", 0.0, 5.0, 0.0, 0.5)
    max_snap_offset_m = st.slider("Max AGM snap offset (m)", 5.0, 150.0, 80.0, 5.0)

# ────────────────────────── Parser (fixed folder names)
def agm_sort_key(name_geom):
    name = name_geom[0]
    digits = ''.join(filter(str.isdigit, name))
    base = int(digits) if digits else -1
    suffix = ''.join(filter(str.isalpha, name))
    return (base, suffix)

def parse_kml_kmz(uploaded_file):
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
    nsfree = lambda tag: tag.split("}", 1)[-1]

    agms, centerlines = [], []

    for folder in root.findall(".//{*}Folder"):
        name_el = folder.find("{*}name")
        if name_el is None or not name_el.text:
            continue
        fname = name_el.text.strip().upper()

        # ---- AGMs ----
        if fname == "AGMS":
            for pm in folder.findall(".//{*}Placemark"):
                nm = pm.find("{*}name")
                cr = pm.find(".//{*}coordinates")
                if nm is None or cr is None:
                    continue
                try:
                    lon, lat, *_ = map(float, cr.text.strip().split(","))
                    agms.append((nm.text.strip(), Point(lon, lat)))
                except:
                    pass

        # ---- CENTERLINE ----
        elif fname == "CENTERLINE":
            for pm in folder.findall(".//{*}Placemark"):
                coords_el = pm.find(".//{*}coordinates")
                if coords_el is None or not coords_el.text:
                    continue
                pts = []
                for pair in coords_el.text.strip().split():
                    try:
                        lon, lat, *_ = map(float, pair.split(","))
                        pts.append((lon, lat))
                    except:
                        pass
                if len(pts) >= 2:
                    centerlines.append(LineString(pts))

    agms.sort(key=agm_sort_key)
    return agms, centerlines

# ────────────────────────── CRS + transforms
def get_local_utm_crs(lines):
    xs, ys = [], []
    for ls in lines:
        xs += [c[0] for c in ls.coords]
        ys += [c[1] for c in ls.coords]
    cx, cy = np.mean(xs), np.mean(ys)
    zone = int((cx + 180) / 6) + 1
    epsg = 32600 + zone if cy >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)

def xf_ll_to(crs): return Transformer.from_crs("EPSG:4326", crs, always_xy=True)
def xf_to_ll(crs): return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
def tf_line(ls, xf): xs, ys = zip(*ls.coords); X, Y = xf.transform(xs, ys); return LineString(zip(X, Y))
def tf_pt(pt, xf): x, y = xf.transform(pt.x, pt.y); return Point(x, y)

# ────────────────────────── TerrainRGB utilities
def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return int(x), int(y), x, y

def decode_terrain_rgb(r, g, b):
    R, G, B = int(r), int(g), int(b)
    return -10000.0 + ((R * 256 * 256) + (G * 256) + B) * 0.1

class TerrainCache:
    def __init__(self, token, zoom=17):
        self.t = token
        self.z = int(zoom)
        self.c = {}
    def get(self, z, x, y):
        key = (z, x, y)
        if key in self.c:
            return self.c[key]
        url = TERRAIN_TILE_URL.format(z=z, x=x, y=y)
        r = requests.get(url, params={"access_token": self.t}, timeout=20)
        if r.status_code != 200:
            return None
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        self.c[key] = arr
        return arr
    def elev(self, lon, lat):
        lon = max(-179.999, min(179.999, lon))
        lat = max(-85.0, min(85.0, lat))
        z = self.z
        xt, yt, xf, yf = lonlat_to_tile(lon, lat, z)
        xp, yp = (xf - xt) * 256, (yf - yt) * 256
        x0, y0 = int(xp), int(yp)
        dx, dy = xp - x0, yp - y0
        x0, y0 = max(0, min(255, x0)), max(0, min(255, y0))
        x1, y1 = min(x0 + 1, 255), min(y0 + 1, 255)
        arr = self.get(z, xt, yt)
        if arr is None:
            return 0.0
        p00 = decode_terrain_rgb(*arr[y0, x0])
        p10 = decode_terrain_rgb(*arr[y0, x1])
        p01 = decode_terrain_rgb(*arr[y1, x0])
        p11 = decode_terrain_rgb(*arr[y1, x1])
        return float(p00*(1-dx)*(1-dy) + p10*dx*(1-dy) + p01*(1-dx)*dy + p11*dx*dy)

def smooth(vals, k):
    if k <= 1: return vals
    arr = np.asarray(vals, dtype=float)
    kernel = np.ones(int(k)) / float(k)
    return np.convolve(arr, kernel, "same").tolist()

# ────────────────────────── Linear-referencing + snapping
def build_arrays(ls):
    c = list(ls.coords)
    xs, ys = np.array([p[0] for p in c]), np.array([p[1] for p in c])
    d = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate([[0], np.cumsum(d)])
    return xs, ys, cum

def interp(xs, ys, cum, s):
    if s <= 0: return xs[0], ys[0]
    if s >= cum[-1]: return xs[-1], ys[-1]
    i = int(np.searchsorted(cum, s) - 1)
    seg = cum[i + 1] - cum[i]
    t = (s - cum[i]) / seg if seg > 0 else 0
    return xs[i] + t*(xs[i+1]-xs[i]), ys[i] + t*(ys[i+1]-ys[i])

def sample(ls, s1, s2, spacing):
    xs, ys, cum = build_arrays(ls)
    s_lo, s_hi = sorted((s1, s2))
    L = abs(s_hi - s_lo)
    steps = np.arange(0, L, spacing)
    if steps.size == 0 or steps[-1] < L:
        steps = np.append(steps, L)
    pts = [interp(xs, ys, cum, s_lo + d) for d in steps]
    X, Y = zip(*pts)
    return np.array(X), np.array(Y)

def choose_part(parts_m, p1_ll, p2_ll, xf, max_off):
    p1m, p2m = tf_pt(p1_ll, xf), tf_pt(p2_ll, xf)
    best = None
    for i, p in enumerate(parts_m):
        s1, s2 = p.project(p1m), p.project(p2m)
        sp1, sp2 = p.interpolate(s1), p.interpolate(s2)
        o1 = ((p1m.x-sp1.x)**2 + (p1m.y-sp1.y)**2)**0.5
        o2 = ((p2m.x-sp2.x)**2 + (p2m.y-sp2.y)**2)**0.5
        if o1 <= max_off and o2 <= max_off:
            tot = o1 + o2
            if best is None or tot < best[0]:
                best = (tot, i, s1, s2)
    return (None, None, None) if best is None else (best[1], best[2], best[3])

# ────────────────────────── MAIN
uploaded = st.file_uploader("Upload KML or KMZ", type=["kml", "kmz"])
if uploaded:
    agms, parts = parse_kml_kmz(uploaded)
    st.text(f"{len(agms)} AGMs | {len(parts)} centerline part(s)")
    if not agms or not parts:
        st.warning("Need both AGMs + centerline.")
        st.stop()

    crs = get_local_utm_crs(parts)
    xf_fwd, xf_inv = xf_ll_to(crs), xf_to_ll(crs)
    parts_m = [tf_line(p, xf_fwd).simplify(simplify_tolerance_m) for p in parts]

    cache = TerrainCache(MAPBOX_TOKEN, zoom=mapbox_zoom)
    rows, cum_mi, skipped = [], 0.0, 0

    for i in range(len(agms) - 1):
        n1, a1 = agms[i]; n2, a2 = agms[i + 1]
        part_idx, s1, s2 = choose_part(parts_m, a1, a2, xf_fwd, max_snap_offset_m)
        if part_idx is None:
            skipped += 1
            continue
        Xs, Ys = sample(parts_m[part_idx], s1, s2, spacing_m)
        lons, lats = xf_inv.transform(Xs.tolist(), Ys.tolist())
        pts = list(zip(lons, lats))
        elev = smooth([cache.elev(lo, la) for lo, la in pts], smooth_window)

        dist_m = 0.0
        for j in range(len(pts) - 1):
            lon1, lat1 = pts[j]; lon2, lat2 = pts[j + 1]
            _, _, dxy = GEOD.inv(lon1, lat1, lon2, lat2)
            dz = elev[j + 1] - elev[j]
            dist_m += math.sqrt(dxy * dxy + dz * dz)

        ft = dist_m * FT_PER_M; mi = ft * MI_PER_FT; cum_mi += mi
        rows.append({"From AGM": n1, "To AGM": n2, "Feet": round(ft, 2), "Miles": round(mi, 6), "Cumulative": round(cum_mi, 6)})

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    st.text(f"Skipped segments: {skipped}")
    st.download_button("Download CSV", df.to_csv(index=False).encode(), "terrain_distances.csv", "text/csv")
