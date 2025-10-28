# app.py — Terrain-Aware AGM Distance Calculator (robust parser + fast tile prefetch)

import io, re, math, zipfile, time, xml.etree.ElementTree as ET
from itertools import chain
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from shapely.geometry import LineString, Point
from pyproj import Geod, Transformer

st.set_page_config("Terrain AGM Distance — Robust", layout="wide")
st.title("📏 Terrain-Aware AGM Distance Calculator — Robust Parser + Prefetch")

# ============ CONFIG ============
# Put your token in .streamlit/secrets.toml as:
# [mapbox]
# token = "pk.XXXX..."
MAPBOX_TOKEN = st.secrets["mapbox"]["token"]
TERRAIN_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"  # v4 endpoint
MAPBOX_ZOOM = 14                          # 14 is a good balance for speed/precision
RESAMPLE_M = 10                           # spacing along the line used for sampling
SMOOTH_WINDOW_M = 40                      # moving-average window applied to elevations
DZ_THRESH_M = 0.5                         # ignore tiny vertical jitter
MAX_SNAP_M = 100                          # max allowed AGM offset to centerline (in meters)
FT_PER_M = 3.28084
GEOD = Geod(ellps="WGS84")

# ============ UTILITIES ============
def moving_avg(values: np.ndarray, window_m: float, spacing_m: float) -> np.ndarray:
    if len(values) < 3:
        return values
    k = max(3, int(round(window_m / max(spacing_m, 1e-6))))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(values, kernel, mode="same")

def agm_sort_key(name: str):
    base = ''.join(ch for ch in name if ch.isdigit())
    base_i = int(base) if base else -1
    suffix = ''.join(ch for ch in name if ch.isalpha())
    return (base_i, suffix)

def build_local_meter_crs(line_ll: LineString):
    xs, ys = zip(*line_ll.coords)
    lon0, lat0 = float(np.mean(xs)), float(np.mean(ys))
    proj = f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m"
    to_m = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    to_ll = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)
    return to_m, to_ll

def sanitize_kml_xml(xml_bytes: bytes) -> bytes:
    """
    Makes stubborn KMLs parseable by ElementTree:
    - removes xmlns declarations (we don't need them)
    - strips any XML prefixes like gx:, kml:, atom:, xsi:, etc. from tag/attr names
    Safe because we only care about tag *local* names and coordinates content.
    """
    data = xml_bytes
    # Drop xmlns declarations
    data = re.sub(br'\sxmlns(:\w+)?="[^"]+"', b'', data)
    # Strip prefixes from start/end tags, e.g. <gx:foo> -> <foo>, </kml:bar> -> </bar>
    data = re.sub(br'<(/?)([A-Za-z0-9_]+):', br'<\1', data)
    # Strip prefixes from attributes, e.g. kml:attr="x" -> attr="x"
    data = re.sub(br'\s([A-Za-z0-9_]+):([A-Za-z0-9_\-]+)=', br' \2=', data)
    return data

def parse_kml_kmz(file) -> tuple[list[tuple[str, Point]], list[LineString]]:
    """
    Robustly parse:
      - AGMs: inside a Folder named 'AGMs' (preferred), else any Placemark with a purely numeric name.
               Skip any name that starts with 'SP' (case-insensitive).
      - CENTERLINE: anywhere under an element whose <name> text == 'CENTERLINE', collecting all
                    descendant LineString coordinates (including inside MultiGeometry). We then
                    merge them in document order into one or more LineStrings.
    """
    # Load inner KML
    if file.name.lower().endswith(".kmz"):
        with zipfile.ZipFile(file) as zf:
            kml_name = next((n for n in zf.namelist() if n.lower().endswith(".kml")), None)
            if not kml_name:
                return [], []
            xml_bytes = zf.read(kml_name)
    else:
        xml_bytes = file.read()

    xml_bytes = sanitize_kml_xml(xml_bytes)
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        st.error(f"KML parse error: {e}")
        return [], []

    # Simple local-name queries (prefixes stripped by sanitize_kml_xml)
    def iter_placemarks(node):
        for el in node.iter():
            if el.tag.endswith("Placemark"):
                yield el

    def text_of(el, tag):
        t = el.find(tag)
        return t.text.strip() if (t is not None and t.text) else None

    # Find all Folders named AGMs
    agm_folders = []
    for folder in root.iter():
        if folder.tag.endswith("Folder"):
            nm = text_of(folder, "name")
            if nm and nm.strip().upper() == "AGMS":
                agm_folders.append(folder)

    agms: list[tuple[str, Point]] = []

    def collect_agm_from_container(container):
        nonlocal agms
        for pm in container.findall(".//Placemark"):
            nm = text_of(pm, "name")
            if not nm:
                continue
            nm_stripped = nm.strip()
            if nm_stripped.upper().startswith("SP"):
                continue
            # must be "numeric-ish" AGM label like 000, 010, 045, 100A, etc.
            # Keep numeric prefix; ignore obvious non-AGM text like 'Wire Gate'
            has_digit = any(ch.isdigit() for ch in nm_stripped)
            if not has_digit:
                continue
            # Coordinates: prefer <Point><coordinates>; fallback to LookAt lon/lat
            lon = lat = None
            coords_el = pm.find(".//Point/coordinates")
            if coords_el is not None and coords_el.text:
                first = coords_el.text.strip().split()
                if first:
                    pair = first[0].split(",")
                    if len(pair) >= 2:
                        lon, lat = float(pair[0]), float(pair[1])
            if lon is None or lat is None:
                lon_el = pm.find(".//longitude")
                lat_el = pm.find(".//latitude")
                if lon_el is not None and lat_el is not None and lon_el.text and lat_el.text:
                    lon, lat = float(lon_el.text.strip()), float(lat_el.text.strip())
            if lon is not None and lat is not None:
                agms.append((nm_stripped, Point(lon, lat)))

    if agm_folders:
        for f in agm_folders:
            collect_agm_from_container(f)
    else:
        # Fallback: accept any Placemark with numeric-ish name (still skips SP*)
        for pm in iter_placemarks(root):
            dummy_container = ET.Element("dummy")
            dummy_container.append(pm)
            collect_agm_from_container(dummy_container)

    # Sort AGMs by numeric name (000, 010, 020, ...)
    agms.sort(key=lambda x: agm_sort_key(x[0]))

    # CENTERLINE extraction
    # Find any element whose direct <name> == 'CENTERLINE', then gather all desc LineString coords
    cl_coord_sets: list[list[tuple[float, float]]] = []
    for node in root.iter():
        nm = text_of(node, "name")
        if nm and nm.strip().upper() == "CENTERLINE":
            for c in node.findall(".//LineString/coordinates"):
                if c.text:
                    pts = []
                    for tok in c.text.strip().split():
                        parts = tok.split(",")
                        if len(parts) >= 2:
                            pts.append((float(parts[0]), float(parts[1])))
                    if len(pts) >= 2:
                        cl_coord_sets.append(pts)

    # If nothing under 'CENTERLINE', try all LineStrings in the doc as a last resort
    if not cl_coord_sets:
        for c in root.findall(".//LineString/coordinates"):
            if c.text:
                pts = []
                for tok in c.text.strip().split():
                    parts = tok.split(",")
                    if len(parts) >= 2:
                        pts.append((float(parts[0]), float(parts[1])))
                if len(pts) >= 2:
                    cl_coord_sets.append(pts)

    # Merge consecutive pieces in document order into a single long path
    centerlines: list[LineString] = []
    if cl_coord_sets:
        merged: list[tuple[float, float]] = []
        for seg in cl_coord_sets:
            if not merged:
                merged.extend(seg)
            else:
                if merged[-1] == seg[0]:
                    merged.extend(seg[1:])
                else:
                    merged.extend(seg)
        if len(merged) >= 2:
            centerlines.append(LineString(merged))

    return agms, centerlines

# ============ MAPBOX TERRAIN CACHE ============
class TerrainCache:
    def __init__(self, token: str, zoom: int):
        self.token = token
        self.z = int(zoom)
        self._tiles: dict[tuple[int, int, int], np.ndarray] = {}

    @staticmethod
    def _decode_rgb_triplet(rgb: np.ndarray) -> float:
        # rgb is a length-3 uint8 array [R,G,B]
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        return -10000.0 + (r * 256 * 256 + g * 256 + b) * 0.1

    @staticmethod
    def _mercator_xy(lon: float, lat: float, z: int):
        n = 2.0 ** z
        xt = (lon + 180.0) / 360.0 * n
        lat_r = math.radians(lat)
        yt = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
        return xt, yt

    def _fetch_tile(self, z: int, x: int, y: int) -> np.ndarray | None:
        key = (z, x, y)
        if key in self._tiles:
            return self._tiles[key]
        url = TERRAIN_URL.format(z=z, x=x, y=y)
        try:
            r = requests.get(url, params={"access_token": self.token}, timeout=10)
        except Exception:
            return None
        if r.status_code != 200:
            return None
        arr = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=np.uint8)
        self._tiles[key] = arr
        return arr

    def prefetch_along(self, lonlat_coords: list[tuple[float, float]]):
        # Prefetch unique tiles hit by these coordinates to warm the cache
        tiles = set()
        for lon, lat in lonlat_coords:
            xt, yt = self._mercator_xy(lon, lat, self.z)
            tiles.add((self.z, int(xt), int(yt)))
        for (z, x, y) in tiles:
            self._fetch_tile(z, x, y)

    def elevations_bulk(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        """
        Bilinear interpolation inside a single 256x256 tile.
        """
        z = self.z
        n = float(2 ** z)
        xt = (lons + 180.0) / 360.0 * n
        lat_r = np.radians(lats)
        yt = (1.0 - np.log(np.tan(lat_r) + 1.0 / np.cos(lat_r)) / math.pi) / 2.0 * n

        x_tile = np.floor(xt).astype(np.int64)
        y_tile = np.floor(yt).astype(np.int64)
        xp = (xt - x_tile) * 255.0
        yp = (yt - y_tile) * 255.0

        out = np.zeros_like(lons, dtype=np.float64)
        for i in range(lons.shape[0]):
            arr = self._fetch_tile(z, int(x_tile[i]), int(y_tile[i]))
            if arr is None:
                out[i] = 0.0
                continue
            x0 = int(np.clip(xp[i], 0, 254))
            y0 = int(np.clip(yp[i], 0, 254))
            x1, y1 = x0 + 1, y0 + 1
            dx = float(xp[i] - x0)
            dy = float(yp[i] - y0)
            e00 = self._decode_rgb_triplet(arr[y0, x0])
            e10 = self._decode_rgb_triplet(arr[y0, x1])
            e01 = self._decode_rgb_triplet(arr[y1, x0])
            e11 = self._decode_rgb_triplet(arr[y1, x1])
            out[i] = e00 * (1 - dx) * (1 - dy) + e10 * dx * (1 - dy) + e01 * (1 - dx) * dy + e11 * dx * dy
        return out

# ============ UI ============
u = st.file_uploader("Upload KML or KMZ", type=["kml", "kmz"])
if not u:
    st.stop()

agms, cls = parse_kml_kmz(u)
st.write(f"**{len(agms)} AGMs | {len(cls)} centerline part(s)**")
if not agms or not cls:
    st.warning("Need both AGMs and CENTERLINE.")
    st.stop()

centerline_ll = cls[0]
to_m, to_ll = build_local_meter_crs(centerline_ll)
# Build metric centerline for projection/snapping
X, Y = to_m.transform(*zip(*centerline_ll.coords))
line_m = LineString(list(zip(X, Y)))

# Prefetch tiles once (speed-up)
st.info("🚀 Prefetching terrain tiles along centerline…")
cache = TerrainCache(MAPBOX_TOKEN, MAPBOX_ZOOM)
cache.prefetch_along(list(centerline_ll.coords))
st.success("✅ Prefetch complete.")

rows = []
cum_miles = 0.0
progress = st.progress(0.0)
status = st.empty()
total = max(0, len(agms) - 1)

for i in range(total):
    n1, p1 = agms[i]
    n2, p2 = agms[i + 1]
    status.text(f"⏱ Calculating {n1} → {n2} ({i + 1}/{total}) …")
    progress.progress((i + 1) / total)

    # Snap AGMs to centerline (meters)
    p1_m = Point(*to_m.transform(p1.x, p1.y))
    p2_m = Point(*to_m.transform(p2.x, p2.y))
    s1 = line_m.project(p1_m)
    s2 = line_m.project(p2_m)
    sp1 = line_m.interpolate(s1)
    sp2 = line_m.interpolate(s2)

    # Skip if AGMs are too far from centerline
    if sp1.distance(p1_m) > MAX_SNAP_M or sp2.distance(p2_m) > MAX_SNAP_M:
        continue

    a, b = (s1, s2) if s1 <= s2 else (s2, s1)
    if abs(b - a) < 1e-6:
        continue

    # Sample points along the segment in meters
    steps = max(2, int(math.ceil((b - a) / RESAMPLE_M)) + 1)
    si = np.linspace(a, b, steps)
    pts_m = [line_m.interpolate(float(s)) for s in si]
    # Back to lon/lat for geodesic + terrain
    pts_ll = np.array([to_ll.transform(p.x, p.y) for p in pts_m], dtype=float)

    # Elevation sampling (bilinear) + smoothing
    elev = cache.elevations_bulk(pts_ll[:, 0], pts_ll[:, 1])
    elev = moving_avg(elev, SMOOTH_WINDOW_M, RESAMPLE_M)

    # Accumulate 3D distance (geodesic XY + vertical Z)
    dist_m = 0.0
    for j in range(len(pts_ll) - 1):
        lon1, lat1 = float(pts_ll[j, 0]), float(pts_ll[j, 1])
        lon2, lat2 = float(pts_ll[j + 1, 0]), float(pts_ll[j + 1, 1])
        _, _, dxy_m = GEOD.inv(lon1, lat1, lon2, lat2)
        dz = float(elev[j + 1] - elev[j])
        if abs(dz) < DZ_THRESH_M:
            dz = 0.0
        dist_m += math.hypot(dxy_m, dz)

    dist_ft = dist_m * FT_PER_M
    dist_mi = dist_ft / 5280.0
    cum_miles += dist_mi

    rows.append({
        "From AGM": n1,
        "To AGM": n2,
        "Distance (feet)": round(dist_ft, 2),
        "Distance (miles)": round(dist_mi, 6),
        "Cumulative (miles)": round(cum_miles, 6)
    })

progress.progress(1.0)
status.text("✅ Complete.")

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)
st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                   "terrain_distances.csv", "text/csv")
