# app.py
# Terrain-Aware AGM Distance Calculator — Pure NumPy smoothing, no SciPy
import io, math, zipfile, requests, xml.etree.ElementTree as ET
import streamlit as st, pandas as pd, numpy as np
from PIL import Image
from shapely.geometry import Point, LineString
from pyproj import CRS, Transformer, Geod

st.set_page_config(page_title="Terrain-Aware AGM Distance Calculator", layout="wide")
st.title("Terrain-Aware AGM Distance Calculator (Pure NumPy smoothing)")

MAPBOX_TOKEN = st.secrets["mapbox"]["token"]
TERRAIN_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"
GEOD = Geod(ellps="WGS84")
FT_PER_M, MI_PER_FT = 3.28084, 1/5280.0
MAPBOX_ZOOM = 17

# --- tuning constants ---
MAX_SNAP_M = 80.0
RESAMPLE_BASE_M = 5.0
XY_SMOOTH_WINDOW_M = 35.0
ELEV_SMOOTH_WINDOW_M = 85.0
SPACING_M = 15.0
DZ_THRESHOLD_M = 0.30

# ---- pure NumPy smoothing ----
def smooth_np(arr, window_m, spacing_m):
    if len(arr) < 5:
        return arr
    win_pts = int(round(window_m / spacing_m))
    if win_pts % 2 == 0:
        win_pts += 1
    win_pts = max(5, min(win_pts, len(arr) - 1 if len(arr) % 2 == 1 else len(arr) - 2))
    if len(arr) < win_pts or win_pts < 5:
        return arr
    x = np.arange(win_pts) - win_pts // 2
    A = np.vander(x, 3)
    coeff, *_ = np.linalg.lstsq(A, np.eye(win_pts), rcond=None)
    filt = coeff[:, 0]
    return np.convolve(arr, filt / filt.sum(), mode="same")

# ---- XML utilities ----
def strip_ns(e):
    e.tag = e.tag.split("}", 1)[-1]
    for k in list(e.attrib):
        nk = k.split("}", 1)[-1]
        if nk != k:
            e.attrib[nk] = e.attrib.pop(k)
    for c in e:
        strip_ns(c)

# ---- KML/KMZ parsing ----
def agm_sort_key(p):
    n = p[0]
    d = "".join(filter(str.isdigit, n))
    s = "".join(filter(str.isalpha, n))
    return (int(d) if d else -1, s)

def parse(upload):
    if upload.name.lower().endswith(".kmz"):
        with zipfile.ZipFile(upload) as z:
            kml = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
            if not kml:
                return [], []
            data = z.read(kml)
    else:
        data = upload.read()
    root = ET.fromstring(data)
    strip_ns(root)
    agms, cls = [], []
    for folder in root.findall(".//Folder"):
        n = folder.find("name")
        if n is None or not n.text:
            continue
        name = n.text.strip().upper()
        if name == "AGMS":
            for pm in folder.findall(".//Placemark"):
                nm = pm.find("name")
                if nm is None or not nm.text:
                    continue
                agm_name = nm.text.strip()
                if agm_name.upper().startswith("SP"):
                    continue
                c = pm.find(".//Point/coordinates")
                if c is None or not c.text:
                    continue
                try:
                    lon, lat, *_ = map(float, c.text.strip().split(","))
                    agms.append((agm_name, Point(lon, lat)))
                except:
                    pass
        elif name == "CENTERLINE":
            for pm in folder.findall(".//Placemark"):
                c = pm.find(".//LineString/coordinates")
                if c is None or not c.text:
                    continue
                pts = []
                for token in c.text.strip().split():
                    try:
                        lon, lat, *_ = map(float, token.split(","))
                        pts.append((lon, lat))
                    except:
                        pass
                if len(pts) >= 2:
                    cls.append(LineString(pts))
    agms.sort(key=agm_sort_key)
    return agms, cls

# ---- projection helpers ----
def get_crs(lines):
    xs = [x for l in lines for x, _ in l.coords]
    ys = [y for l in lines for _, y in l.coords]
    cx, cy = np.mean(xs), np.mean(ys)
    zone = int((cx + 180) / 6) + 1
    return CRS.from_epsg(32600 + zone if cy >= 0 else 32700 + zone)
def xf_ll_to(crs): return Transformer.from_crs("EPSG:4326", crs, always_xy=True)
def xf_to_ll(crs): return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
def tf_line(l, xf): x, y = zip(*l.coords); X, Y = xf.transform(x, y); return LineString(zip(X, Y))
def tf_pt(p, xf): x, y = xf.transform(p.x, p.y); return Point(x, y)

# ---- terrain RGB ----
def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return int(x), int(y), x, y
def decode_rgb(r, g, b):
    R, G, B = int(r), int(g), int(b)
    return -10000 + ((R * 256 * 256) + (G * 256) + B) * 0.1
class TerrainCache:
    def __init__(self, tok, z): self.t, self.z, self.c = tok, int(z), {}
    def get(self, z, x, y):
        k = (z, x, y)
        if k in self.c: return self.c[k]
        r = requests.get(TERRAIN_URL.format(z=z, x=x, y=y), params={"access_token": self.t}, timeout=20)
        if r.status_code != 200: return None
        arr = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), np.uint8)
        self.c[k] = arr; return arr
    def elev(self, lon, lat):
        lon = max(-179.9, min(179.9, lon)); lat = max(-85, min(85, lat))
        z = self.z; xt, yt, xf, yf = lonlat_to_tile(lon, lat, z)
        xp, yp = (xf - xt) * 256, (yf - yt) * 256
        x0, y0 = int(xp), int(yp); dx, dy = xp - x0, yp - y0
        x0, y0 = max(0, min(255, x0)), max(0, min(255, y0))
        x1, y1 = min(x0 + 1, 255), min(y0 + 1, 255)
        arr = self.get(z, xt, yt)
        if arr is None: return 0.0
        p00 = decode_rgb(*arr[y0, x0]); p10 = decode_rgb(*arr[y0, x1])
        p01 = decode_rgb(*arr[y1, x0]); p11 = decode_rgb(*arr[y1, x1])
        return float(p00*(1-dx)*(1-dy)+p10*dx*(1-dy)+p01*(1-dx)*dy+p11*dx*dy)

# ---- geometry helpers ----
def build_arrays(ls):
    c = list(ls.coords)
    x = np.array([p[0] for p in c], float)
    y = np.array([p[1] for p in c], float)
    d = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(d)])
    return x, y, s
def interp_xy(x, y, s, si): return np.interp(si, s, x), np.interp(si, s, y)
def choose_part(parts, p1, p2, xf, max_off):
    p1m, p2m = tf_pt(p1, xf), tf_pt(p2, xf)
    best = None
    for i, p in enumerate(parts):
        s1, s2 = p.project(p1m), p.project(p2m)
        sp1, sp2 = p.interpolate(s1), p.interpolate(s2)
        o1 = ((p1m.x - sp1.x)**2 + (p1m.y - sp1.y)**2)**0.5
        o2 = ((p2m.x - sp2.x)**2 + (p2m.y - sp2.y)**2)**0.5
        if o1 <= max_off and o2 <= max_off:
            tot = o1 + o2
            if best is None or tot < best[0]:
                best = (tot, i, s1, s2)
    return (None, None, None) if best is None else (best[1], best[2], best[3])

# ---- main Streamlit UI ----
u = st.file_uploader("Upload KML or KMZ", type=["kml", "kmz"])
if u:
    agms, parts = parse(u)
    st.text(f"{len(agms)} AGMs | {len(parts)} centerline part(s)")
    if not agms or not parts:
        st.warning("Need both AGMs + centerline.")
        st.stop()

    crs = get_crs(parts)
    xf_fwd, xf_inv = xf_ll_to(crs), xf_to_ll(crs)
    parts_m = [tf_line(p, xf_fwd) for p in parts if p.length > 0]

    cache = TerrainCache(MAPBOX_TOKEN, MAPBOX_ZOOM)
    rows, cum_mi, skipped = [], 0.0, 0

    for i in range(len(agms) - 1):
        n1, a1 = agms[i]; n2, a2 = agms[i + 1]
        idx, s1, s2 = choose_part(parts_m, a1, a2, xf_fwd, MAX_SNAP_M)
        if idx is None:
            skipped += 1; continue
        part = parts_m[idx]
        x, y, s = build_arrays(part)
        s_lo, s_hi = sorted((float(s1), float(s2)))
        L = s[-1]
        s_lo = max(0.0, min(L, s_lo))
        s_hi = max(0.0, min(L, s_hi))
        if s_hi - s_lo <= 0: skipped += 1; continue

        si = np.arange(s_lo, s_hi, RESAMPLE_BASE_M)
        if si.size == 0 or si[-1] < s_hi: si = np.append(si, s_hi)
        xi, yi = interp_xy(x, y, s, si)
        xi_s = smooth_np(np.asarray(xi, float), XY_SMOOTH_WINDOW_M, RESAMPLE_BASE_M)
        yi_s = smooth_np(np.asarray(yi, float), XY_SMOOTH_WINDOW_M, RESAMPLE_BASE_M)
        d = np.hypot(np.diff(xi_s), np.diff(yi_s))
        s2 = np.concatenate([[0.0], np.cumsum(d)])
        L2 = s2[-1]
        if L2 <= 0: skipped += 1; continue
        sp = np.arange(0.0, L2, SPACING_M)
        if sp.size == 0 or sp[-1] < L2: sp = np.append(sp, L2)
        X, Y = interp_xy(xi_s, yi_s, s2, sp)
        lons, lats = xf_inv.transform(X.tolist(), Y.tolist())
        pts = list(zip(lons, lats))
        elev = [cache.elev(lo, la) for lo, la in pts]
        elev = smooth_np(np.asarray(elev, float), ELEV_SMOOTH_WINDOW_M, SPACING_M)
        dist_m = 0.0
        for j in range(len(pts) - 1):
            lon1, lat1 = pts[j]; lon2, lat2 = pts[j + 1]
            _, _, dxy = GEOD.inv(lon1, lat1, lon2, lat2)
            dz = elev[j + 1] - elev[j]
            if abs(dz) < DZ_THRESHOLD_M: dz = 0.0
            dist_m += math.hypot(dxy, dz)
        ft = dist_m * FT_PER_M
        mi = ft * MI_PER_FT
        cum_mi += mi
        rows.append({"From AGM": n1, "To AGM": n2,
                     "Distance (feet)": round(ft, 2),
                     "Distance (miles)": round(mi, 6),
                     "Cumulative (miles)": round(cum_mi, 6)})

    df = pd.DataFrame(rows)
    st.subheader("📊 Distance table")
    st.dataframe(df, use_container_width=True)
    st.text(f"Skipped segments: {skipped}")
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                       "terrain_distances.csv", "text/csv")
