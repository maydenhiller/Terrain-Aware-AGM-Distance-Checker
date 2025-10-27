# Terrain-Aware AGM Distance Calculator — Final Geodesic 3D Build (Auto-Detect Parser)
# Requirements: same as before (no new libraries)

import math, io, zipfile, requests, xml.etree.ElementTree as ET
import streamlit as st, pandas as pd, numpy as np
from PIL import Image
from shapely.geometry import Point, LineString
from pyproj import CRS, Transformer, Geod

# --- CONFIG
st.set_page_config(page_title="Terrain-Aware AGM Distance Calculator", layout="wide")
st.title("Terrain-Aware AGM Distance Calculator (Geodesic 3D, Auto-Detect)")

MAPBOX_TOKEN = st.secrets["mapbox"]["token"]
TERRAIN_TILE_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"
GEOD = Geod(ellps="WGS84")

FT_PER_M = 3.28084
MI_PER_FT = 1 / 5280.0

with st.sidebar:
    st.header("Settings")
    mapbox_zoom = st.slider("Terrain tile zoom", 15, 17, 17)
    spacing_m = st.slider("Sample spacing along path (m)", 0.5, 10.0, 1.0, 0.5)
    smooth_window = st.slider("Elevation smoothing window (samples)", 1, 25, 7, 2)
    simplify_tolerance_m = st.slider("Simplify centerline parts (m)", 0.0, 5.0, 0.0, 0.5)
    st.caption("zoom 17, spacing 1 m, smoothing ≈ 7 gives best accuracy")

# --- PARSE KML/KMZ (Auto Detect)
def agm_sort_key(n_g):
    n = n_g[0]
    d = ''.join(filter(str.isdigit, n))
    b = int(d) if d else -1
    s = ''.join(filter(str.isalpha, n)).upper()
    return (b, s)

def parse_kml_kmz(uploaded):
    """Auto-detect AGMs and Centerline geometries from KML/KMZ"""
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

    agms = []
    parts = []

    # --- Find all Placemarks anywhere in the document
    for placemark in root.findall(".//kml:Placemark", ns):
        name_el = placemark.find("kml:name", ns)
        coords_el = placemark.find(".//kml:coordinates", ns)
        linestr = placemark.find(".//kml:LineString", ns)
        point = placemark.find(".//kml:Point", ns)
        if coords_el is None:
            continue

        coords_text = coords_el.text.strip()
        if not coords_text:
            continue

        # If it’s a single coordinate → AGM point
        coord_pairs = coords_text.split()
        if len(coord_pairs) == 1 or point is not None:
            try:
                lon, lat, *_ = map(float, coord_pairs[0].split(","))
                nm = name_el.text.strip() if name_el is not None else f"AGM_{len(agms)}"
                agms.append((nm, Point(lon, lat)))
            except Exception:
                continue
        # If it’s a LineString with many coordinates → centerline
        elif linestr is not None or len(coord_pairs) > 2:
            pts = []
            for pair in coord_pairs:
                try:
                    lon, lat, *_ = map(float, pair.split(","))
                    pts.append((lon, lat))
                except:
                    pass
            if len(pts) >= 2:
                parts.append(LineString(pts))

    agms.sort(key=agm_sort_key)
    return agms, parts

# --- CRS Helpers
def get_local_utm_crs(lines_ll):
    xs, ys = [], []
    for ls in lines_ll:
        xs += [c[0] for c in ls.coords]; ys += [c[1] for c in ls.coords]
    cx, cy = np.mean(xs), np.mean(ys)
    zone = int((cx + 180) / 6) + 1
    epsg = 32600 + zone if cy >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)

def xf_ll_to(crs): return Transformer.from_crs("EPSG:4326", crs, always_xy=True)
def xf_to_ll(crs): return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
def tf_linestring(ls, xf): xs, ys = zip(*ls.coords); X, Y = xf.transform(xs, ys); return LineString(zip(X, Y))
def tf_point(pt, xf): x, y = xf.transform(pt.x, pt.y); return Point(x, y)

# --- Terrain RGB
def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return int(x), int(y), x, y

def decode_terrain_rgb(r, g, b):
    return -10000 + (r * 256 * 256 + g * 256 + b) * 0.1

class TerrainCache:
    def __init__(self, token, zoom=17): self.t, self.z, self.c = token, zoom, {}
    def get(self, z, x, y):
        k = (z, x, y)
        if k in self.c: return self.c[k]
        url = TERRAIN_TILE_URL.format(z=z, x=x, y=y)
        r = requests.get(url, params={"access_token": self.t}, timeout=20)
        if r.status_code != 200: return None
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        arr = np.asarray(img, np.uint8); self.c[k] = arr; return arr
    def elev(self, lon, lat):
        z = self.z
        xt, yt, xf, yf = lonlat_to_tile(lon, lat, z)
        xp, yp = (xf - xt) * 256, (yf - yt) * 256
        x0, y0 = int(xp), int(yp); dx, dy = xp - x0, yp - y0
        x0, y0 = np.clip(x0, 0, 255), np.clip(y0, 0, 255)
        x1, y1 = min(x0+1,255), min(y0+1,255)
        arr = self.get(z, xt, yt)
        if arr is None: return 0
        p00 = decode_terrain_rgb(*arr[y0, x0]); p10 = decode_terrain_rgb(*arr[y0, x1])
        p01 = decode_terrain_rgb(*arr[y1, x0]); p11 = decode_terrain_rgb(*arr[y1, x1])
        return float(
            p00*(1-dx)*(1-dy) + p10*dx*(1-dy) + p01*(1-dx)*dy + p11*dx*dy
        )

def smooth(vals, k):
    if k <= 1: return vals
    kernel = np.ones(k) / k
    return np.convolve(vals, kernel, "same").tolist()

# --- Linear Reference utilities
def build_arrays(ls_m):
    c = list(ls_m.coords)
    xs, ys = np.array([p[0] for p in c]), np.array([p[1] for p in c])
    d = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate([[0], np.cumsum(d)])
    return xs, ys, cum

def interp_on_poly(xs, ys, cum, s):
    if s <= 0: return xs[0], ys[0]
    if s >= cum[-1]: return xs[-1], ys[-1]
    i = int(np.searchsorted(cum, s) - 1)
    seg = cum[i+1] - cum[i]
    t = (s - cum[i]) / seg if seg else 0
    return xs[i] + t*(xs[i+1]-xs[i]), ys[i] + t*(ys[i+1]-ys[i])

def sample_between(ls_m, s1, s2, spacing):
    xs, ys, cum = build_arrays(ls_m)
    s0, s1 = sorted((s1, s2))
    L = abs(s2 - s1)
    steps = np.arange(0, L, spacing)
    if steps.size == 0 or steps[-1] < L: steps = np.append(steps, L)
    outX, outY = [], []
    for s in steps + s0:
        x, y = interp_on_poly(xs, ys, cum, s)
        outX.append(x); outY.append(y)
    return np.array(outX), np.array(outY)

# --- Snap each AGM pair to one correct part
def choose_part(parts_m, pt1_ll, pt2_ll, xf_ll_to_m, max_off):
    p1m, p2m = tf_point(pt1_ll, xf_ll_to_m), tf_point(pt2_ll, xf_ll_to_m)
    best = None
    for i, p in enumerate(parts_m):
        s1, s2 = p.project(p1m), p.project(p2m)
        sp1, sp2 = p.interpolate(s1), p.interpolate(s2)
        o1 = ((p1m.x-sp1.x)**2+(p1m.y-sp1.y)**2)**0.5
        o2 = ((p2m.x-sp2.x)**2+(p2m.y-sp2.y)**2)**0.5
        if o1<=max_off and o2<=max_off:
            tot = o1+o2
            if best is None or tot<best[0]: best=(tot,i,s1,s2)
    return (None,None,None) if best is None else best[1:]

# --- MAIN
up = st.file_uploader("Upload KML/KMZ file", type=["kml","kmz"])
if up:
    agms, parts_ll = parse_kml_kmz(up)
    st.text(f"{len(agms)} AGMs | {len(parts_ll)} centerline part(s)")
    if len(parts_ll)==0 or len(agms)<2:
        st.warning("Need both AGMs + centerline."); st.stop()

    crs = get_local_utm_crs(parts_ll)
    xf_fwd, xf_inv = xf_ll_to(crs), xf_to_ll(crs)
    parts_m = []
    for p in parts_ll:
        pm = tf_linestring(p, xf_fwd)
        if simplify_tolerance_m>0:
            pm = pm.simplify(simplify_tolerance_m, preserve_topology=False)
        if pm.length>0: parts_m.append(pm)

    cache = TerrainCache(MAPBOX_TOKEN, zoom=mapbox_zoom)
    rows, cum_mi, skip = [], 0.0, 0

    for i in range(len(agms)-1):
        n1, a1 = agms[i]; n2, a2 = agms[i+1]
        part_idx, s1, s2 = choose_part(parts_m, a1, a2, xf_fwd, 80)
        if part_idx is None: skip+=1; continue
        part = parts_m[part_idx]
        Xs, Ys = sample_between(part, s1, s2, spacing_m)
        if Xs.size<2: skip+=1; continue

        # Convert samples back to lon/lat for true geodesic distances
        lons, lats = xf_inv.transform(Xs.tolist(), Ys.tolist())
        pts = list(zip(lons, lats))

        elev = [cache.elev(lo,la) for lo,la in pts]
        elev = smooth(elev, smooth_window)

        dist_m = 0.0
        for j in range(len(pts)-1):
            lon1, lat1 = pts[j]; lon2, lat2 = pts[j+1]
            _, _, dxy = GEOD.inv(lon1, lat1, lon2, lat2)
            dz = elev[j+1]-elev[j]
            dist_m += math.sqrt(dxy*dxy + dz*dz)

        ft, mi = dist_m*FT_PER_M, dist_m*FT_PER_M*MI_PER_FT
        cum_mi += mi
        rows.append({
            "From AGM":n1,"To AGM":n2,
            "Feet":round(ft,2),"Miles":round(mi,6),
            "Cum Miles":round(cum_mi,6)
        })

    df=pd.DataFrame(rows)
    st.dataframe(df,use_container_width=True)
    st.text(f"Skipped segments: {skip}")
    st.download_button("Download CSV",df.to_csv(index=False).encode(),
                       "terrain_distances.csv","text/csv")
