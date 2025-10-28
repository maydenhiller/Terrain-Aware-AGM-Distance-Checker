# app.py — Terrain-Aware AGM Distance Calculator (Final v5: Document-based CENTERLINE support)

import io, math, re, zipfile, xml.etree.ElementTree as ET
import numpy as np, pandas as pd, requests, streamlit as st
from PIL import Image
from shapely.geometry import LineString, Point
from pyproj import Geod, Transformer
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config("Terrain AGM Distance", layout="wide")
st.title("📏 Terrain-Aware AGM Distance Calculator — Final v5")

# ---------------- CONFIG ----------------
MAPBOX_TOKEN = st.secrets.get("MAPBOX_TOKEN", "")
MAPBOX_ZOOM = 14
RESAMPLE_M = 10
SMOOTH_WINDOW = 40
DZ_THRESH = 0.5
FT_PER_M = 3.28084
GEOD = Geod(ellps="WGS84")
MAX_WORKERS = 8

# ---------------- TERRAIN CACHE ----------------
class TerrainCache:
    def __init__(self, token, zoom):
        self.token = token
        self.zoom = int(zoom)
        self.tiles = {}

    @staticmethod
    def decode_rgb_arr(rgb_arr):
        r, g, b = rgb_arr[..., 0], rgb_arr[..., 1], rgb_arr[..., 2]
        return -10000.0 + (r.astype(np.int64)*256*256 + g.astype(np.int64)*256 + b.astype(np.int64)) * 0.1

    def fetch_tile(self, z, x, y):
        key = (z, x, y)
        if key in self.tiles:
            return self.tiles[key]
        url = f"https://api.mapbox.com/v1/mapbox/terrain-rgb/{z}/{x}/{y}.pngraw"
        try:
            r = requests.get(url, params={"access_token": self.token}, timeout=10)
            if r.status_code != 200:
                print(f"[Mapbox error] {r.status_code} for {z}/{x}/{y}")
                return None
            arr = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=np.uint8)
            self.tiles[key] = arr
            return arr
        except Exception as e:
            print(f"[Mapbox fetch error] {e}")
            return None

    def elevations_bulk(self, lons, lats):
        z = self.zoom
        n = float(2 ** z)
        xt = (lons + 180.0) / 360.0 * n
        yt = (1.0 - np.log(np.tan(np.radians(lats)) + 1 / np.cos(np.radians(lats))) / math.pi) / 2.0 * n
        x_tile, y_tile = np.floor(xt).astype(np.int64), np.floor(yt).astype(np.int64)
        xp, yp = (xt - x_tile) * 255.0, (yt - y_tile) * 255.0
        out = np.zeros_like(lons)
        for i in range(len(lons)):
            arr = self.fetch_tile(z, int(x_tile[i]), int(y_tile[i]))
            if arr is None:
                continue
            x0, y0 = int(np.clip(xp[i], 0, 254)), int(np.clip(yp[i], 0, 254))
            x1, y1 = x0 + 1, y0 + 1
            dx, dy = xp[i] - x0, yp[i] - y0
            e00 = self.decode_rgb_arr(arr[y0, x0])
            e10 = self.decode_rgb_arr(arr[y0, x1])
            e01 = self.decode_rgb_arr(arr[y1, x0])
            e11 = self.decode_rgb_arr(arr[y1, x1])
            out[i] = e00*(1-dx)*(1-dy) + e10*dx*(1-dy) + e01*(1-dx)*dy + e11*dx*dy
        return out

    def prefetch_tiles(self, lon_lat_pairs):
        z = self.zoom
        n = float(2 ** z)
        tiles = set()
        for lon, lat in lon_lat_pairs:
            xt = (lon + 180.0) / 360.0 * n
            yt = (1.0 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
            tiles.add((z, int(xt), int(yt)))
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(self.fetch_tile, *t) for t in tiles]
            for _ in as_completed(futures):
                pass
        print(f"Prefetched {len(tiles)} tiles.")

# ---------------- HELPERS ----------------
def smooth(a, window_m, spacing_m):
    if len(a) < 3:
        return a
    n = max(3, int(round(window_m / max(spacing_m, 1e-6))))
    if n % 2 == 0:
        n += 1
    kernel = np.ones(n) / n
    return np.convolve(a, kernel, mode="same")

# ---------------- PARSER ----------------
def parse_kml_kmz(uploaded_file):
    if uploaded_file.name.lower().endswith(".kmz"):
        with zipfile.ZipFile(uploaded_file) as zf:
            kml_name = next((n for n in zf.namelist() if n.lower().endswith(".kml")), None)
            if not kml_name:
                return [], []
            data = zf.read(kml_name).decode("utf-8", errors="ignore")
    else:
        data = uploaded_file.read().decode("utf-8", errors="ignore")

    agms, centerlines = [], []

    # ---- AGM extraction (same as before)
    placemarks = re.findall(r"<Placemark>(.*?)</Placemark>", data, re.S | re.I)
    for pm in placemarks:
        if "<Point" not in pm:
            continue
        name_match = re.search(r"<name>(.*?)</name>", pm, re.S | re.I)
        if not name_match:
            continue
        name = name_match.group(1).strip()
        if name.upper().startswith("SP"):
            continue
        lon = lat = None
        coord_match = re.search(r"<coordinates>([-\d\.,\s]+)</coordinates>", pm, re.S | re.I)
        if coord_match:
            first_pair = coord_match.group(1).strip().split()[0]
            parts = first_pair.split(",")
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
        else:
            lat_match = re.search(r"<latitude>([-\d\.]+)</latitude>", pm)
            lon_match = re.search(r"<longitude>([-\d\.]+)</longitude>", pm)
            if lat_match and lon_match:
                lat = float(lat_match.group(1))
                lon = float(lon_match.group(1))
        if lon and lat:
            agms.append((name, Point(lon, lat)))

    agms.sort(key=lambda p: int(''.join(filter(str.isdigit, p[0]))) if any(ch.isdigit() for ch in p[0]) else -1)

    # ---- CENTERLINE extraction (supports <Document><name>CENTERLINE</name>)
    center_doc = re.search(r"<Document[^>]*>\s*<name>\s*CENTERLINE\s*</name>(.*?)</Document>", data, re.S | re.I)
    center_folder = re.search(r"<Folder[^>]*>\s*<name>\s*CENTERLINE\s*</name>(.*?)</Folder>", data, re.S | re.I)

    center_src = None
    if center_doc:
        center_src = center_doc.group(1)
    elif center_folder:
        center_src = center_folder.group(1)
    else:
        # fallback to entire file if needed
        center_src = data

    coords = re.findall(r"<coordinates>(.*?)</coordinates>", center_src, re.S | re.I)
    for block in coords:
        pts = []
        for seg in block.strip().split():
            parts = seg.split(",")
            if len(parts) >= 2:
                pts.append((float(parts[0]), float(parts[1])))
        if len(pts) >= 2:
            centerlines.append(LineString(pts))

    return agms, centerlines

# ---------------- MAIN ----------------
u = st.file_uploader("Upload KML/KMZ", type=["kml", "kmz"])
if not u:
    st.stop()

agms, lines = parse_kml_kmz(u)
st.text(f"{len(agms)} AGMs | {len(lines)} centerline part(s)")
if not agms or not lines:
    st.warning("Need both AGMs and CENTERLINE.")
    st.stop()

line_ll = lines[0]
to_m = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
X_m, Y_m = to_m.transform(*zip(*line_ll.coords))
line_m = LineString(zip(X_m, Y_m))

cache = TerrainCache(MAPBOX_TOKEN, MAPBOX_ZOOM)
st.info("🚀 Prefetching terrain tiles in parallel...")
cache.prefetch_tiles(list(line_ll.coords))
st.success("✅ Tile prefetch complete — starting calculations")

rows, cum_mi = [], 0.0
bar, status = st.progress(0), st.empty()
total = len(agms) - 1

for i in range(total):
    n1, a1 = agms[i]
    n2, a2 = agms[i + 1]
    status.text(f"⏱ Calculating {n1} → {n2} ({i + 1}/{total}) …")
    bar.progress((i + 1) / total)
    p1_m = Point(*to_m.transform(a1.x, a1.y))
    p2_m = Point(*to_m.transform(a2.x, a2.y))
    s1, s2 = sorted((line_m.project(p1_m), line_m.project(p2_m)))
    if abs(s2 - s1) < 1:
        continue
    si = np.arange(s1, s2, RESAMPLE_M)
    if len(si) == 0 or si[-1] < s2:
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
