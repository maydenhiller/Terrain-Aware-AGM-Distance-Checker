# app.py — Terrain-Aware AGM Distances (Geodesic + Elevation, 25 m spacing)
# - Accurate great-circle (WGS84) distances with elevation (Terrain-RGB)
# - Robust KML/KMZ parsing:
#     * AGMs from <Folder><name>AGMs</name> ... <Point><coordinates>lon,lat,z
#     * CENTERLINE from <Folder><name>CENTERLINE</name> ... LineString coords (2D/3D, multi parts)
#     * Ignores AGMs whose name starts with "SP"
# - Snaps AGMs to nearest point on CENTERLINE (local Transverse Mercator only for snapping)
# - 25 m resampling between snapped AGMs; bilinear elevation sampling; simple smoothing
# - Progress bar + CSV export

import io, math, zipfile, xml.etree.ElementTree as ET
import numpy as np, pandas as pd, requests, streamlit as st
from PIL import Image
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import linemerge
from pyproj import Geod, Transformer

# ---------------- UI & CONFIG ----------------
st.set_page_config("Terrain AGM Distance — Geodesic", layout="wide")
st.title("📏 Terrain-Aware AGM Distance Calculator — Geodesic + Elevation")

# Secrets: allow either "MAPBOX_TOKEN" or nested ["mapbox"]["token"]
MAPBOX_TOKEN = st.secrets.get("MAPBOX_TOKEN") or (
    st.secrets.get("mapbox", {}).get("token") if isinstance(st.secrets.get("mapbox"), dict) else None
)
if not MAPBOX_TOKEN:
    st.error("Missing Mapbox token. Add `MAPBOX_TOKEN` (or `mapbox.token`) to `st.secrets`.")
    st.stop()

# Tunables (chosen for speed without losing accuracy)
RESAMPLE_M = 25                 # ← your requested spacing
SMOOTH_WINDOW_M = 50            # simple moving average window (≈ 2 samples)
ELEV_DZ_THRESHOLD = 0.25        # ignore tiny elevation noise (meters)
MAPBOX_ZOOM = 14                # good compromise of detail/speed for long lines
FT_PER_M = 3.28084
GEOD = Geod(ellps="WGS84")
MAX_SNAP_M = 120                # reject AGMs too far from centerline (safety)

# ---------------- Terrain-RGB helper ----------------
# Official tileset (v4 raster): https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw
TERRAIN_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"

def decode_terrain_rgb_triplet(r, g, b):
    # Elevation meters = -10000 + (R*256^2 + G*256 + B) * 0.1
    return -10000.0 + (int(r) * 256 * 256 + int(g) * 256 + int(b)) * 0.1

class TerrainRGBCache:
    def __init__(self, token: str, zoom: int):
        self.token = token
        self.z = int(zoom)
        self.cache = {}  # (z,x,y) -> np.array(H,W,3) uint8

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
            # Fallback: remember None to avoid retries
            self.cache[key] = None
            st.warning(f"[Mapbox] HTTP {r.status_code} for tile {self.z}/{x_tile}/{y_tile}")
            return None
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)  # (256,256,3)
        self.cache[key] = arr
        return arr

    def elevations_bulk(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        """Vectorized elevations with tile-grouped fetch + bilinear sampling."""
        z = self.z
        xt, yt = self.lonlat_to_tile(lons, lats, z)
        tx = np.floor(xt).astype(np.int32)
        ty = np.floor(yt).astype(np.int32)
        xp = (xt - tx) * 255.0
        yp = (yt - ty) * 255.0

        # Group indices by tile key to fetch/decode each tile once
        out = np.zeros_like(lons, dtype=np.float64)
        keys, inv = np.unique(np.stack([tx, ty], axis=1), axis=0, return_inverse=True)
        for tile_idx, (x_tile, y_tile) in enumerate(keys):
            arr = self.fetch_tile(int(x_tile), int(y_tile))
            # If tile couldn't be fetched, leave zeros (rare, small effect)
            if arr is None:
                continue

            # Indices in this tile
            mask = (inv == tile_idx)
            xps = xp[mask]
            yps = yp[mask]

            # Pixel corners
            x0 = np.clip(xps.astype(np.int32), 0, 254)
            y0 = np.clip(yps.astype(np.int32), 0, 254)
            x1 = x0 + 1
            y1 = y0 + 1
            dx = xps - x0
            dy = yps - y0

            # Gather RGB at corners
            p00 = arr[y0, x0]  # (N,3)
            p10 = arr[y0, x1]
            p01 = arr[y1, x0]
            p11 = arr[y1, x1]

            # Decode elevations
            e00 = -10000.0 + (p00[:, 0].astype(np.int64) * 65536 + p00[:, 1].astype(np.int64) * 256 + p00[:, 2].astype(np.int64)) * 0.1
            e10 = -10000.0 + (p10[:, 0].astype(np.int64) * 65536 + p10[:, 1].astype(np.int64) * 256 + p10[:, 2].astype(np.int64)) * 0.1
            e01 = -10000.0 + (p01[:, 0].astype(np.int64) * 65536 + p01[:, 1].astype(np.int64) * 256 + p01[:, 2].astype(np.int64)) * 0.1
            e11 = -10000.0 + (p11[:, 0].astype(np.int64) * 65536 + p11[:, 1].astype(np.int64) * 256 + p11[:, 2].astype(np.int64)) * 0.1

            # Bilinear interpolation
            vals = e00 * (1 - dx) * (1 - dy) + e10 * dx * (1 - dy) + e01 * (1 - dx) * dy + e11 * dx * dy
            out[mask] = vals

        return out

# ---------------- Utilities ----------------
def moving_average(values: np.ndarray, window_m: float, spacing_m: float) -> np.ndarray:
    if len(values) < 3:
        return values
    k = max(3, int(round(window_m / max(spacing_m, 1e-6))))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(values, kernel, mode="same")

def strip_namespaces(elem: ET.Element):
    # Make KML parsing robust regardless of namespaces
    elem.tag = elem.tag.split("}", 1)[-1]
    new_attrib = {}
    for k, v in elem.attrib.items():
        new_attrib[k.split("}", 1)[-1]] = v
    elem.attrib.clear()
    elem.attrib.update(new_attrib)
    for c in list(elem):
        strip_namespaces(c)

def parse_kml_kmz(uploaded) -> tuple[list[tuple[str, Point]], list[LineString]]:
    """Find AGMs and CENTERLINE (robust to nesting, namespaces, MultiGeometry, 2D/3D)."""
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
        # Some KMZ writers insert BOM or stray bytes; try to locate the first '<'
        s = raw
        if isinstance(s, bytes):
            i = s.find(b"<")
            if i > 0:
                s = s[i:]
        else:
            i = s.find("<")
            if i > 0:
                s = s[i:]
        root = ET.fromstring(s)

    strip_namespaces(root)

    agms: list[tuple[str, Point]] = []
    centerline_parts: list[LineString] = []

    for folder in root.findall(".//Folder"):
        nm = folder.find("name")
        if nm is None or not nm.text:
            continue
        fname = nm.text.strip().upper()

        if fname == "AGMS":
            # any Placemark with a Point/coordinates
            for pm in folder.findall(".//Placemark"):
                name_el = pm.find("name")
                if name_el is None or not name_el.text:
                    continue
                label = name_el.text.strip()
                if label.upper().startswith("SP"):
                    continue  # ignore "SP..." as requested

                coord_el = pm.find(".//Point/coordinates")
                if coord_el is None or not coord_el.text:
                    continue
                txt = coord_el.text.strip().replace("\n", " ").replace("\t", " ")
                # handle lon,lat[,alt]
                tok = [t for t in txt.split(",") if t]
                try:
                    lon = float(tok[0]); lat = float(tok[1])
                except Exception:
                    continue
                agms.append((label, Point(lon, lat)))

        elif fname == "CENTERLINE":
            # Grab ALL LineString coordinates (including inside MultiGeometry)
            for coords_el in folder.findall(".//LineString/coordinates"):
                txt = coords_el.text or ""
                pts = []
                for pair in txt.replace("\n", " ").replace("\t", " ").split():
                    if not pair:
                        continue
                    parts = pair.split(",")
                    if len(parts) < 2:
                        continue
                    try:
                        lon = float(parts[0]); lat = float(parts[1])
                    except Exception:
                        continue
                    pts.append((lon, lat))
                if len(pts) >= 2:
                    centerline_parts.append(LineString(pts))

    # Sort AGMs numerically by digits in name (fallback -1)
    def agm_key(item):
        name = item[0]
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else -1

    agms.sort(key=agm_key)

    # Merge centerline parts where connected; keep list of LineStrings
    merged = []
    if len(centerline_parts) == 1:
        merged = centerline_parts
    elif len(centerline_parts) > 1:
        merged_geom = linemerge(MultiLineString(centerline_parts))
        if isinstance(merged_geom, LineString):
            merged = [merged_geom]
        elif isinstance(merged_geom, MultiLineString):
            merged = list(merged_geom.geoms)
        else:
            merged = centerline_parts  # fallback: unmerged

    return agms, merged

def build_local_meter_crs(line_ll: LineString):
    # Local TMERC centered on line extent center — used ONLY for snapping & stationing
    xs, ys = zip(*list(line_ll.coords))
    lon0, lat0 = float(np.mean(xs)), float(np.mean(ys))
    proj = f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    to_m   = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    to_ll  = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)
    return to_m, to_ll

def choose_best_centerline(lines: list[LineString], p1_ll: Point, p2_ll: Point) -> LineString | None:
    """Pick the line that minimizes total snap distance for the two points (in meters)."""
    if not lines:
        return None
    best = None
    best_d = float("inf")
    for ln in lines:
        to_m, _ = build_local_meter_crs(ln)
        ln_m = LineString(list(zip(*to_m.transform(*zip(*ln.coords)))))
        p1_m = Point(*to_m.transform(p1_ll.x, p1_ll.y))
        p2_m = Point(*to_m.transform(p2_ll.x, p2_ll.y))
        d = p1_m.distance(ln_m) + p2_m.distance(ln_m)
        if d < best_d:
            best_d = d
            best = ln
    return best

# ---------------- Main ----------------
uploaded = st.file_uploader("Upload a KML or KMZ with AGMs + CENTERLINE", type=["kml", "kmz"])
if not uploaded:
    st.stop()

agms, centerlines = parse_kml_kmz(uploaded)
st.write(f"**{len(agms)}** AGMs | **{len(centerlines)}** centerline part(s)")
if len(agms) < 2 or not centerlines:
    st.warning("Need both AGMs and CENTERLINE.")
    st.stop()

rows = []
cum_mi = 0.0
bar = st.progress(0.0)
status = st.empty()

cache = TerrainRGBCache(MAPBOX_TOKEN, MAPBOX_ZOOM)

for i in range(len(agms) - 1):
    n1, a1 = agms[i]
    n2, a2 = agms[i + 1]
    status.text(f"⏱ Calculating {n1} → {n2} ({i+1}/{len(agms)-1}) …")

    # pick best centerline for this pair
    line_ll = choose_best_centerline(centerlines, a1, a2)
    if line_ll is None:
        continue

    # local CRS for snapping + stationing
    to_m, to_ll = build_local_meter_crs(line_ll)
    line_m = LineString(list(zip(*to_m.transform(*zip(*line_ll.coords)))))

    p1_m = Point(*to_m.transform(a1.x, a1.y))
    p2_m = Point(*to_m.transform(a2.x, a2.y))

    # Reject if too far away (bad AGM or wrong line)
    d1 = p1_m.distance(line_m)
    d2 = p2_m.distance(line_m)
    if d1 > MAX_SNAP_M or d2 > MAX_SNAP_M:
        # skip this pair if clearly off the line
        rows.append({
            "From AGM": n1, "To AGM": n2,
            "Distance (feet)": 0.0, "Distance (miles)": 0.0,
            "Cumulative (miles)": round(cum_mi, 6),
            "Note": f"Skipped (snap too far: {d1:.1f} m / {d2:.1f} m)"
        })
        bar.progress((i+1) / (len(agms)-1))
        continue

    s1 = line_m.project(p1_m)
    s2 = line_m.project(p2_m)
    if abs(s2 - s1) < 1e-6:
        # zero-length segment
        rows.append({
            "From AGM": n1, "To AGM": n2,
            "Distance (feet)": 0.0, "Distance (miles)": 0.0,
            "Cumulative (miles)": round(cum_mi, 6),
            "Note": "Zero station difference"
        })
        bar.progress((i+1) / (len(agms)-1))
        continue

    s_start, s_end = (s1, s2) if s1 < s2 else (s2, s1)
    # sample stations every RESAMPLE_M, include end
    count = max(2, int(round((s_end - s_start) / RESAMPLE_M)) + 1)
    stations = np.linspace(s_start, s_end, count)
    # coordinates in local meters, then back to lon/lat for elevation and geodesic
    pts_m = [line_m.interpolate(s) for s in stations]
    pts_ll = np.array([to_ll.transform(p.x, p.y) for p in pts_m])  # (N,2) lon,lat

    # Elevations (vectorized)
    elev = cache.elevations_bulk(pts_ll[:, 0], pts_ll[:, 1])
    # Smooth mildly
    elev = moving_average(elev, SMOOTH_WINDOW_M, (stations[1] - stations[0]))

    # Accumulate 3D geodesic
    dist_m = 0.0
    for j in range(len(pts_ll) - 1):
        lon1, lat1 = pts_ll[j]
        lon2, lat2 = pts_ll[j + 1]
        _, _, dxy = GEOD.inv(lon1, lat1, lon2, lat2)   # meters along ellipsoid
        dz = elev[j + 1] - elev[j]
        if abs(dz) < ELEV_DZ_THRESHOLD:
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

    bar.progress((i+1) / (len(agms)-1))

status.success("✅ Complete.")
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)
st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                   file_name="terrain_distances.csv", mime="text/csv")
