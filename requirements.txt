# app.py — Terrain-Aware AGM Distances (Geodesic + Elevation, 25 m spacing, hard-coded Mapbox token)

import io, math, zipfile, xml.etree.ElementTree as ET
import numpy as np, pandas as pd, requests, streamlit as st
from PIL import Image
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import linemerge
from pyproj import Geod, Transformer

# ---------------- UI & CONFIG ----------------
st.set_page_config("Terrain AGM Distance — Geodesic", layout="wide")
st.title("📏 Terrain-Aware AGM Distance Calculator — Geodesic + Elevation")

# --- HARDCODED MAPBOX TOKEN ---
MAPBOX_TOKEN = "pk.eyJ1IjoibWF5ZGVuaGlsbGVyIiwiYSI6ImNtZ2ljMnN5ejA3amwyam9tNWZnYnZibWwifQ.GXoTyHdvCYtr7GvKIW9LPA"

# Tunables
RESAMPLE_M = 25
SMOOTH_WINDOW_M = 50
ELEV_DZ_THRESHOLD = 0.25
MAPBOX_ZOOM = 14
FT_PER_M = 3.28084
GEOD = Geod(ellps="WGS84")
MAX_SNAP_M = 120
TERRAIN_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"

# ---------------- TERRAIN CACHE ----------------
class TerrainRGBCache:
    def __init__(self, token: str, zoom: int):
        self.token = token
        self.z = int(zoom)
        self.cache = {}

    @staticmethod
    def lonlat_to_tile(lon, lat, z):
        n = 2 ** z
        xt = (lon + 180.0) / 360.0 * n
        lat_r = np.radians(lat)
        yt = (1.0 - np.log(np.tan(lat_r) + 1.0 / np.cos(lat_r)) / math.pi) / 2.0 * n
        return xt, yt

    def fetch_tile(self, x_tile: int, y_tile: int):
        key = (self.z, x_tile, y_tile)
        if key in self.cache:
            return self.cache[key]
        url = TERRAIN_URL.format(z=self.z, x=x_tile, y=y_tile)
        r = requests.get(url, params={"access_token": self.token}, timeout=12)
        if r.status_code != 200:
            self.cache[key] = None
            st.warning(f"[Mapbox] HTTP {r.status_code} for tile {self.z}/{x_tile}/{y_tile}")
            return None
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        self.cache[key] = arr
        return arr

    def elevations_bulk(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        z = self.z
        xt, yt = self.lonlat_to_tile(lons, lats, z)
        tx = np.floor(xt).astype(np.int32)
        ty = np.floor(yt).astype(np.int32)
        xp = (xt - tx) * 255.0
        yp = (yt - ty) * 255.0

        out = np.zeros_like(lons, dtype=np.float64)
        keys, inv = np.unique(np.stack([tx, ty], axis=1), axis=0, return_inverse=True)

        for tile_idx, (x_tile, y_tile) in enumerate(keys):
            arr = self.fetch_tile(int(x_tile), int(y_tile))
            if arr is None:
                continue

            mask = (inv == tile_idx)
            xps = xp[mask]
            yps = yp[mask]

            x0 = np.clip(xps.astype(np.int32), 0, 254)
            y0 = np.clip(yps.astype(np.int32), 0, 254)
            x1 = x0 + 1
            y1 = y0 + 1
            dx = xps - x0
            dy = yps - y0

            p00 = arr[y0, x0]
            p10 = arr[y0, x1]
            p01 = arr[y1, x0]
            p11 = arr[y1, x1]

            e00 = -10000.0 + (p00[:, 0]*65536 + p00[:, 1]*256 + p00[:, 2]) * 0.1
            e10 = -10000.0 + (p10[:, 0]*65536 + p10[:, 1]*256 + p10[:, 2]) * 0.1
            e01 = -10000.0 + (p01[:, 0]*65536 + p01[:, 1]*256 + p01[:, 2]) * 0.1
            e11 = -10000.0 + (p11[:, 0]*65536 + p11[:, 1]*256 + p11[:, 2]) * 0.1

            vals = e00*(1-dx)*(1-dy) + e10*dx*(1-dy) + e01*(1-dx)*dy + e11*dx*dy
            out[mask] = vals

        return out

# ---------------- HELPERS ----------------
def moving_average(values: np.ndarray, window_m: float, spacing_m: float) -> np.ndarray:
    if len(values) < 3:
        return values
    k = max(3, int(round(window_m / max(spacing_m, 1e-6))))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k) / k
    return np.convolve(values, kernel, mode="same")

def strip_namespaces(elem: ET.Element):
    elem.tag = elem.tag.split("}", 1)[-1]
    elem.attrib = {k.split("}", 1)[-1]: v for k, v in elem.attrib.items()}
    for c in list(elem):
        strip_namespaces(c)

def parse_kml_kmz(uploaded):
    if uploaded.name.lower().endswith(".kmz"):
        with zipfile.ZipFile(uploaded) as zf:
            kml_name = next((n for n in zf.namelist() if n.lower().endswith(".kml")), None)
            if not kml_name:
                return [], []
            raw = zf.read(kml_name)
    else:
        raw = uploaded.read()

    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        s = raw[raw.find(b"<"):] if isinstance(raw, bytes) else raw[raw.find("<"):]
        root = ET.fromstring(s)

    strip_namespaces(root)

    agms, centerlines = [], []
    for folder in root.findall(".//Folder"):
        nm = folder.find("name")
        if nm is None or not nm.text:
            continue
        fname = nm.text.strip().upper()

        if fname == "AGMS":
            for pm in folder.findall(".//Placemark"):
                name_el = pm.find("name")
                if name_el is None or not name_el.text:
                    continue
                label = name_el.text.strip()
                if label.upper().startswith("SP"):
                    continue
                coord_el = pm.find(".//Point/coordinates")
                if coord_el is None or not coord_el.text:
                    continue
                tok = [t for t in coord_el.text.strip().replace("\n", " ").split(",") if t]
                try:
                    lon, lat = float(tok[0]), float(tok[1])
                    agms.append((label, Point(lon, lat)))
                except:
                    continue

        elif fname == "CENTERLINE":
            for coords_el in folder.findall(".//LineString/coordinates"):
                pts = []
                for pair in coords_el.text.strip().replace("\n", " ").split():
                    parts = pair.split(",")
                    if len(parts) >= 2:
                        try:
                            lon, lat = float(parts[0]), float(parts[1])
                            pts.append((lon, lat))
                        except:
                            pass
                if len(pts) >= 2:
                    centerlines.append(LineString(pts))

    def agm_key(item):
        digits = "".join(ch for ch in item[0] if ch.isdigit())
        return int(digits) if digits else -1
    agms.sort(key=agm_key)

    if len(centerlines) > 1:
        merged = linemerge(MultiLineString(centerlines))
        if isinstance(merged, LineString):
            centerlines = [merged]
        elif isinstance(merged, MultiLineString):
            centerlines = list(merged.geoms)

    return agms, centerlines

def build_local_meter_crs(line_ll):
    xs, ys = zip(*line_ll.coords)
    lon0, lat0 = np.mean(xs), np.mean(ys)
    proj = f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m"
    to_m = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    to_ll = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)
    return to_m, to_ll

def choose_best_centerline(lines, p1, p2):
    if not lines:
        return None
    best, best_d = None, 1e12
    for ln in lines:
        to_m, _ = build_local_meter_crs(ln)
        ln_m = LineString(zip(*to_m.transform(*zip(*ln.coords))))
        p1_m = Point(*to_m.transform(p1.x, p1.y))
        p2_m = Point(*to_m.transform(p2.x, p2.y))
        d = p1_m.distance(ln_m) + p2_m.distance(ln_m)
        if d < best_d:
            best_d, best = d, ln
    return best

# ---------------- MAIN ----------------
uploaded = st.file_uploader("Upload KML/KMZ", type=["kml", "kmz"])
if not uploaded:
    st.stop()

agms, centerlines = parse_kml_kmz(uploaded)
st.write(f"{len(agms)} AGMs | {len(centerlines)} centerline part(s)")
if len(agms) < 2 or not centerlines:
    st.warning("Need both AGMs and CENTERLINE.")
    st.stop()

cache = TerrainRGBCache(MAPBOX_TOKEN, MAPBOX_ZOOM)
rows, cum_mi = [], 0.0
bar, status = st.progress(0.0), st.empty()

for i in range(len(agms) - 1):
    n1, a1 = agms[i]
    n2, a2 = agms[i + 1]
    status.text(f"⏱ Calculating {n1} → {n2} ({i+1}/{len(agms)-1}) …")

    line_ll = choose_best_centerline(centerlines, a1, a2)
    if not line_ll:
        continue

    to_m, to_ll = build_local_meter_crs(line_ll)
    line_m = LineString(zip(*to_m.transform(*zip(*line_ll.coords))))
    p1_m, p2_m = Point(*to_m.transform(a1.x, a1.y)), Point(*to_m.transform(a2.x, a2.y))
    if p1_m.distance(line_m) > MAX_SNAP_M or p2_m.distance(line_m) > MAX_SNAP_M:
        continue

    s1, s2 = line_m.project(p1_m), line_m.project(p2_m)
    s_start, s_end = (s1, s2) if s1 < s2 else (s2, s1)
    count = max(2, int(round((s_end - s_start) / RESAMPLE_M)) + 1)
    stations = np.linspace(s_start, s_end, count)
    pts_m = [line_m.interpolate(s) for s in stations]
    pts_ll = np.array([to_ll.transform(p.x, p.y) for p in pts_m])

    elev = cache.elevations_bulk(pts_ll[:, 0], pts_ll[:, 1])
    elev = moving_average(elev, SMOOTH_WINDOW_M, RESAMPLE_M)

    dist_m = 0.0
    for j in range(len(pts_ll) - 1):
        lon1, lat1 = pts_ll[j]
        lon2, lat2 = pts_ll[j + 1]
        _, _, dxy = GEOD.inv(lon1, lat1, lon2, lat2)
        dz = elev[j + 1] - elev[j]
        if abs(dz) < ELEV_DZ_THRESHOLD:
            dz = 0
        dist_m += math.hypot(dxy, dz)

    feet, miles = dist_m * FT_PER_M, dist_m * FT_PER_M / 5280
    cum_mi += miles
    rows.append({
        "From AGM": n1,
        "To AGM": n2,
        "Distance (feet)": round(feet, 2),
        "Distance (miles)": round(miles, 6),
        "Cumulative (miles)": round(cum_mi, 6)
    })
    bar.progress((i + 1) / (len(agms) - 1))

status.success("✅ Complete.")
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)
st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                   file_name="terrain_distances.csv", mime="text/csv")
