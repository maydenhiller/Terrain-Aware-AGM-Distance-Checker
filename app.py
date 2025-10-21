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
from shapely.ops import substring
from pyproj import Transformer, CRS, Geod

# =========================
# CONFIG & UI
# =========================

st.set_page_config(page_title="Terrain-Aware AGM Distance Calculator", layout="wide")
st.title("Terrain-Aware AGM Distance Calculator (with diagnostics)")

MAPBOX_TOKEN = st.secrets["mapbox"]["token"]
TERRAIN_TILE_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"

with st.sidebar:
    st.header("Settings")
    mapbox_zoom = st.slider("Terrain tile zoom (higher = finer elevation)", 15, 17, 17)
    interp_spacing_m = st.slider("Sampling spacing along path (meters)", 0.5, 5.0, 1.0, 0.5)
    smooth_window = st.slider("Elevation smoothing window (points)", 1, 21, 5, 2)
    densify_spacing_m = st.slider("Centerline densify spacing (meters)", 5, 50, 10, 5)
    use_densify = st.checkbox("Use centerline densify (recommended)", value=True)
    use_3d = st.checkbox("Use 3D distance (include elevation)", value=True)
    st.caption("Diagnostics below will show UTM 2D, UTM 3D, and Geodesic 2D for each segment.")

# =========================
# HELPERS
# =========================

def agm_sort_key(name_geom):
    name = name_geom[0]
    base_digits = ''.join(filter(str.isdigit, name))
    base = int(base_digits) if base_digits else -1
    suffix = ''.join(filter(str.isalpha, name)).upper()
    return (base, suffix)

def parse_kml_kmz(uploaded_file):
    if uploaded_file.name.endswith(".kmz"):
        with zipfile.ZipFile(uploaded_file) as zf:
            kml_file = next((f for f in zf.namelist() if f.endswith(".kml")), None)
            if not kml_file:
                return [], None
            with zf.open(kml_file) as f:
                kml_data = f.read()
    else:
        kml_data = uploaded_file.read()

    root = ET.fromstring(kml_data)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    agms = []
    centerline = None

    for folder in root.findall(".//kml:Folder", ns):
        name_el = folder.find("kml:name", ns)
        if name_el is None or not name_el.text:
            continue
        folder_name = name_el.text.strip().lower()

        if folder_name == "agms":
            for placemark in folder.findall("kml:Placemark", ns):
                pname = placemark.find("kml:name", ns)
                coords = placemark.find(".//kml:coordinates", ns)
                if pname is None or coords is None:
                    continue
                try:
                    lon, lat, *_ = map(float, coords.text.strip().split(","))
                    agms.append((pname.text.strip(), Point(lon, lat)))
                except Exception:
                    continue

        elif folder_name == "centerline":
            for placemark in folder.findall("kml:Placemark", ns):
                coords = placemark.find(".//kml:coordinates", ns)
                if coords is None:
                    continue
                try:
                    pts = []
                    for pair in coords.text.strip().split():
                        lon, lat, *_ = map(float, pair.split(","))
                        pts.append((lon, lat))
                    if len(pts) >= 2:
                        centerline = LineString(pts)
                except Exception:
                    continue

    agms.sort(key=agm_sort_key)
    return agms, centerline

# ---- CRS / transformers ----

def get_local_utm_crs(geom_ll: LineString) -> CRS:
    xs = [c[0] for c in geom_ll.coords]
    ys = [c[1] for c in geom_ll.coords]
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    zone = int((cx + 180.0) / 6.0) + 1
    is_north = cy >= 0.0
    epsg = 32600 + zone if is_north else 32700 + zone
    return CRS.from_epsg(epsg)

def xf_ll_to_metric(crs_metric: CRS):
    return Transformer.from_crs("EPSG:4326", crs_metric, always_xy=True)

def xf_metric_to_ll(crs_metric: CRS):
    return Transformer.from_crs(crs_metric, "EPSG:4326", always_xy=True)

def transform_linestring(ls: LineString, transformer: Transformer) -> LineString:
    xs, ys = zip(*[(c[0], c[1]) for c in ls.coords])
    X, Y = transformer.transform(xs, ys)
    return LineString(list(zip(X, Y)))

def transform_point(pt: Point, transformer: Transformer) -> Point:
    x, y = transformer.transform(pt.x, pt.y)
    return Point(x, y)

# ---- Densify centerline in meters ----

def densify_line_metric(line_metric: LineString, spacing_m: float) -> LineString:
    L = line_metric.length
    if L <= spacing_m:
        return line_metric
    dists = np.arange(0.0, L, spacing_m)
    pts = [line_metric.interpolate(d) for d in dists]
    pts.append(line_metric.interpolate(L))
    return LineString([(p.x, p.y) for p in pts])

# ---- Accurate slicing using metric CRS ----

def slice_centerline_metric(centerline_metric: LineString, p1_metric: Point, p2_metric: Point):
    d1 = centerline_metric.project(p1_metric)
    d2 = centerline_metric.project(p2_metric)
    if np.isclose(d1, d2):
        return None, d1, d2
    start, end = (d1, d2) if d1 < d2 else (d2, d1)
    seg = substring(centerline_metric, start, end, normalized=False)
    if seg is None or seg.length == 0.0 or len(seg.coords) < 2:
        return None, d1, d2
    return seg, start, end

# ---- Sampling along the metric segment ----

def interpolate_line_metric(line_metric: LineString, spacing_m: float):
    L = line_metric.length
    if L <= spacing_m:
        return [line_metric.interpolate(0.0), line_metric.interpolate(L)]
    steps = max(int(L / spacing_m), 1)
    dists = [i * spacing_m for i in range(steps)]
    pts = [line_metric.interpolate(d) for d in dists]
    pts.append(line_metric.interpolate(L))
    return pts

# =========================
# Mapbox Terrain-RGB (bilinear)
# =========================

def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return int(x), int(y), x, y

def decode_terrain_rgb(r, g, b):
    return -10000.0 + (r * 256.0 * 256.0 + g * 256.0 + b) * 0.1

class TerrainTileCache:
    def __init__(self, token, zoom=17):
        self.token = token
        self.zoom = zoom
        self.cache = {}

    def get_tile_array(self, z, x, y):
        key = (z, x, y)
        arr = self.cache.get(key)
        if arr is not None:
            return arr
        url = TERRAIN_TILE_URL.format(z=z, x=x, y=y)
        resp = requests.get(url, params={"access_token": self.token}, timeout=20)
        if resp.status_code != 200:
            return None
        try:
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)
        except Exception:
            return None
        self.cache[key] = arr
        return arr

    def elevation_at_bilinear(self, lon, lat):
        z = self.zoom
        x_tile, y_tile, x_f, y_f = lonlat_to_tile(lon, lat, z)
        x_pix_f = (x_f - x_tile) * 256.0
        y_pix_f = (y_f - y_tile) * 256.0

        x0 = int(math.floor(x_pix_f))
        y0 = int(math.floor(y_pix_f))
        dx = x_pix_f - x0
        dy = y_pix_f - y0

        x0 = max(0, min(255, x0))
        y0 = max(0, min(255, y0))
        x1 = min(x0 + 1, 255)
        y1 = min(y0 + 1, 255)

        arr = self.get_tile_array(z, x_tile, y_tile)
        if arr is None:
            return None

        p00 = decode_terrain_rgb(*arr[y0, x0])
        p10 = decode_terrain_rgb(*arr[y0, x1])
        p01 = decode_terrain_rgb(*arr[y1, x0])
        p11 = decode_terrain_rgb(*arr[y1, x1])

        elev = (
            p00 * (1 - dx) * (1 - dy)
            + p10 * dx * (1 - dy)
            + p01 * (1 - dx) * dy
            + p11 * dx * dy
        )
        return float(elev)

def get_elevations_ll(points_ll, cache: TerrainTileCache):
    elevations = []
    for (lon, lat) in points_ll:
        e = cache.elevation_at_bilinear(lon, lat)
        elevations.append(0.0 if (e is None or not np.isfinite(e)) else e)
    return elevations

def smooth_elevations(elevs, window: int):
    if window <= 1:
        return elevs
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(elevs, kernel, mode="same").tolist()

# ---- Geodesic helper (WGS84) ----

GEOD = Geod(ellps="WGS84")

def geodesic_length_ll(coords_ll):
    """Total geodesic 2D length (meters) along lon/lat polyline coords."""
    if len(coords_ll) < 2:
        return 0.0
    total = 0.0
    for i in range(len(coords_ll) - 1):
        lon1, lat1 = coords_ll[i]
        lon2, lat2 = coords_ll[i + 1]
        _, _, dist = GEOD.inv(lon1, lat1, lon2, lat2)  # meters
        total += dist
    return total

# =========================
# MAIN APP
# =========================

uploaded_file = st.file_uploader("Upload KML or KMZ file", type=["kml", "kmz"])

if uploaded_file:
    agms, centerline_ll = parse_kml_kmz(uploaded_file)

    st.subheader("📌 AGM summary")
    st.text(f"Total AGMs found: {len(agms)}")
    st.subheader("📈 CENTERLINE status")
    st.text("CENTERLINE found" if centerline_ll else "CENTERLINE missing")

    if not centerline_ll or len(agms) < 2:
        st.warning("Missing CENTERLINE or insufficient AGM points.")
    else:
        # Metric CRS (local UTM; fallback to EPSG:5070)
        try:
            crs_metric = get_local_utm_crs(centerline_ll)
        except Exception:
            crs_metric = CRS.from_epsg(5070)
        to_metric = xf_ll_to_metric(crs_metric)
        to_ll = xf_metric_to_ll(crs_metric)

        centerline_metric = transform_linestring(centerline_ll, to_metric)
        if use_densify:
            centerline_metric = densify_line_metric(centerline_metric, densify_spacing_m)

        # Mapbox elevation cache
        tile_cache = TerrainTileCache(token=MAPBOX_TOKEN, zoom=mapbox_zoom)

        rows = []
        diag_rows = []
        cumulative_miles = 0.0
        skipped = 0

        # Pre-transform AGMs to metric for projection/slice
        agms_metric = [(name, transform_point(pt, to_metric)) for name, pt in agms]

        for i in range(len(agms) - 1):
            name1, pt1_ll = agms[i]
            name2, pt2_ll = agms[i + 1]
            _, pt1_m = agms_metric[i]
            _, pt2_m = agms_metric[i + 1]

            segment_m, s0, s1 = slice_centerline_metric(centerline_metric, pt1_m, pt2_m)
            if segment_m is None or segment_m.length == 0.0 or len(segment_m.coords) < 2:
                skipped += 1
                continue

            # ==== UTM 2D baseline ====
            utm2d_m = segment_m.length

            # ==== Interpolate for elevation & diagnostics ====
            interp_pts_m = interpolate_line_metric(segment_m, spacing_m=interp_spacing_m)
            xs = [p.x for p in interp_pts_m]
            ys = [p.y for p in interp_pts_m]
            lons, lats = to_ll.transform(xs, ys)
            interp_pts_ll = list(zip(lons, lats))

            # ==== Geodesic 2D along identical lon/lat polyline ====
            geo2d_m = geodesic_length_ll(interp_pts_ll)

            # ==== Elevations (bilinear) ====
            elevations = get_elevations_ll(interp_pts_ll, tile_cache)
            elevations = smooth_elevations(elevations, smooth_window)

            # ==== UTM 3D accumulation ====
            utm3d_m = 0.0
            if use_3d:
                for j in range(len(interp_pts_m) - 1):
                    dx = xs[j + 1] - xs[j]
                    dy = ys[j + 1] - ys[j]
                    dz = elevations[j + 1] - elevations[j]
                    utm3d_m += math.sqrt(dx * dx + dy * dy + dz * dz)
            else:
                utm3d_m = utm2d_m  # report 2D only if 3D disabled

            # ==== Per-segment grade diagnostics ====
            dxy = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
            dzs = np.diff(np.array(elevations))
            grades = np.divide(np.abs(dzs), np.maximum(dxy, 1e-6))  # rise/run
            avg_grade = float(np.mean(grades)) if len(grades) else 0.0
            p95_grade = float(np.percentile(grades, 95)) if len(grades) else 0.0

            # ==== Output rows ====
            dist_ft = utm3d_m * 3.28084
            dist_mi = dist_ft / 5280.0
            cumulative_miles += dist_mi

            rows.append({
                "From AGM": name1,
                "To AGM": name2,
                "Distance 3D (ft)": round(dist_ft, 2),
                "Distance 3D (mi)": round(dist_mi, 6),
                "Cumulative (mi)": round(cumulative_miles, 6)
            })

            diag_rows.append({
                "From AGM": name1,
                "To AGM": name2,
                "UTM 2D (m)": round(utm2d_m, 3),
                "UTM 3D (m)": round(utm3d_m, 3),
                "Geodesic 2D (m)": round(geo2d_m, 3),
                "Avg grade": round(avg_grade, 3),
                "95% grade": round(p95_grade, 3),
                "%3D over 2D": round(100.0 * (utm3d_m - utm2d_m) / max(utm2d_m, 1e-6), 2),
                "%UTM2D over Geo2D": round(100.0 * (utm2d_m - geo2d_m) / max(geo2d_m, 1e-6), 2),
            })

        # ==== Tables ====
        st.subheader("📊 Reported distances")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        st.subheader("🧪 Diagnostics (compare the three distance paths)")
        ddf = pd.DataFrame(diag_rows)
        st.dataframe(ddf, use_container_width=True)

        st.text(f"Skipped segments: {skipped}")

        # ==== CSV download ====
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV (3D distances)", csv, "terrain_distances.csv", "text/csv")

        diag_csv = ddf.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV (diagnostics)", diag_csv, "terrain_diagnostics.csv", "text/csv")
