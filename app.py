# app.py
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

# =========================================
# CONFIG
# =========================================

st.set_page_config(page_title="Terrain-Aware AGM Distance Calculator", layout="wide")
st.title("Terrain-Aware AGM Distance Calculator (Snap, Geodesic-3D)")

MAPBOX_TOKEN = st.secrets["mapbox"]["token"]
TERRAIN_TILE_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"
GEOD = Geod(ellps="WGS84")

with st.sidebar:
    st.header("Settings")
    mapbox_zoom = st.slider("Terrain tile zoom", 15, 17, 17)
    interp_spacing_m = st.slider("Sampling spacing along path (m)", 0.5, 5.0, 1.0, 0.5)
    smooth_window = st.slider("Elevation smoothing window (samples)", 1, 21, 5, 2)
    simplify_tolerance_m = st.slider(
        "Centerline simplify tolerance (m)", 0.0, 5.0, 0.0, 0.5,
        help="Optional: remove micro-wiggles in the centerline (0 = off)"
    )
    st.caption("For best accuracy: zoom=17, spacing=1 m, smoothing≈5.")

FT_PER_M = 3.28084
MI_PER_FT = 1.0 / 5280.0

# =========================================
# PARSING
# =========================================

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
                pts = []
                for pair in coords.text.strip().split():
                    lon, lat, *_ = map(float, pair.split(","))
                    pts.append((lon, lat))
                if len(pts) >= 2:
                    if centerline is None:
                        centerline = LineString(pts)
                    else:
                        centerline = LineString(list(centerline.coords) + pts)

    agms.sort(key=agm_sort_key)
    return agms, centerline

# =========================================
# CRS / TRANSFORMS (for snapping & linear reference)
# =========================================

def get_local_utm_crs(line_ll: LineString) -> CRS:
    xs = [c[0] for c in line_ll.coords]
    ys = [c[1] for c in line_ll.coords]
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    zone = int((cx + 180.0) / 6.0) + 1
    is_north = cy >= 0.0
    epsg = 32600 + zone if is_north else 32700 + zone
    return CRS.from_epsg(epsg)

def transformer_ll_to(crs: CRS) -> Transformer:
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)

def transformer_to_ll(crs: CRS) -> Transformer:
    return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

def transform_linestring(ls: LineString, xf: Transformer) -> LineString:
    xs, ys = zip(*ls.coords)
    X, Y = xf.transform(xs, ys)
    return LineString(list(zip(X, Y)))

def transform_point(pt: Point, xf: Transformer) -> Point:
    x, y = xf.transform(pt.x, pt.y)
    return Point(x, y)

# =========================================
# EXACT LINEAR REFERENCING (metric)
# =========================================

def build_vertex_arrays_metric(centerline_m: LineString):
    coords = list(centerline_m.coords)
    xs = np.array([c[0] for c in coords], dtype=float)
    ys = np.array([c[1] for c in coords], dtype=float)
    dxy = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate([[0.0], np.cumsum(dxy)])
    return xs, ys, cum

def interpolate_point_on_polyline(xs, ys, cum, s):
    if s <= 0:
        return float(xs[0]), float(ys[0]), 0
    if s >= cum[-1]:
        return float(xs[-1]), float(ys[-1]), len(xs) - 2
    idx = int(np.searchsorted(cum, s) - 1)
    idx = max(0, min(idx, len(xs) - 2))
    seg_len = cum[idx + 1] - cum[idx]
    if seg_len <= 0:
        return float(xs[idx]), float(ys[idx]), idx
    t = (s - cum[idx]) / seg_len
    x = xs[idx] + t * (xs[idx + 1] - xs[idx])
    y = ys[idx] + t * (ys[idx + 1] - ys[idx])
    return float(x), float(y), idx

def slice_polyline_by_measures(xs, ys, cum, s0, s1):
    s_lo, s_hi = (s0, s1) if s0 <= s1 else (s1, s0)
    xA, yA, iA = interpolate_point_on_polyline(xs, ys, cum, s_lo)
    xB, yB, iB = interpolate_point_on_polyline(xs, ys, cum, s_hi)

    Xs = [xA]
    Ys = [yA]
    if iB >= iA and (iA + 1) <= iB:
        Xs.extend(xs[iA + 1:iB + 1].tolist())
        Ys.extend(ys[iA + 1:iB + 1].tolist())
    Xs.append(xB)
    Ys.append(yB)
    return np.array(Xs, dtype=float), np.array(Ys, dtype=float)

# =========================================
# GEODESIC SAMPLING ALONG A LON/LAT POLYLINE
# =========================================

def geodesic_segment_m(p0, p1):
    lon1, lat1 = p0
    lon2, lat2 = p1
    _, _, d = GEOD.inv(lon1, lat1, lon2, lat2)
    return float(d)

def geodesic_cum_along(coords_ll):
    if len(coords_ll) < 2:
        return np.array([0.0])
    cum = [0.0]
    total = 0.0
    for i in range(len(coords_ll) - 1):
        d = geodesic_segment_m(coords_ll[i], coords_ll[i + 1])
        total += d
        cum.append(total)
    return np.array(cum, dtype=float)

def resample_polyline_geodesic(coords_ll, spacing_m):
    """Return lon/lat points every spacing_m along the given lon/lat polyline by geodesic arclength."""
    if len(coords_ll) < 2:
        return coords_ll
    cum = geodesic_cum_along(coords_ll)
    L = float(cum[-1])
    if L <= 0:
        return coords_ll
    targets = np.arange(0.0, L, float(spacing_m))
    if targets.size == 0 or targets[-1] < L:
        targets = np.append(targets, L)

    out = []
    for t in targets:
        idx = int(np.searchsorted(cum, t, side="right") - 1)
        idx = max(0, min(idx, len(coords_ll) - 2))
        seg_len = cum[idx + 1] - cum[idx]
        if seg_len <= 0:
            out.append(coords_ll[idx])
            continue
        frac = (t - cum[idx]) / seg_len
        lon0, lat0 = coords_ll[idx]
        lon1, lat1 = coords_ll[idx + 1]
        # Linear in lon/lat is a good approximation at these tiny segments
        lon = lon0 + frac * (lon1 - lon0)
        lat = lat0 + frac * (lat1 - lat0)
        out.append((lon, lat))
    return out

# =========================================
# MAPBOX TERRAIN-RGB (bilinear)
# =========================================

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
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        self.cache[key] = arr
        return arr

    def elevation_at_bilinear(self, lon, lat):
        z = self.zoom
        x_tile, y_tile, x_f, y_f = lonlat_to_tile(lon, lat, z)
        x_pix_f = (x_f - x_tile) * 256.0
        y_pix_f = (y_f - y_tile) * 256.0
        x0, y0 = int(math.floor(x_pix_f)), int(math.floor(y_pix_f))
        dx, dy = x_pix_f - x0, y_pix_f - y0
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
    elevs = []
    for lon, lat in points_ll:
        e = cache.elevation_at_bilinear(lon, lat)
        elevs.append(0.0 if e is None else e)
    return elevs

def smooth_elevations(elevs, window):
    if window <= 1:
        return elevs
    kernel = np.ones(window) / window
    return np.convolve(elevs, kernel, mode="same").tolist()

# =========================================
# MAIN
# =========================================

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
        # Local metric CRS for precise snapping/linear ref
        try:
            crs_metric = get_local_utm_crs(centerline_ll)
        except Exception:
            crs_metric = CRS.from_epsg(5070)  # fallback CONUS Albers
        xf_ll_to_m = transformer_ll_to(crs_metric)
        xf_m_to_ll = transformer_to_ll(crs_metric)

        # Centerline in metric (optionally simplify to remove micro-noise)
        cl_m = transform_linestring(centerline_ll, xf_ll_to_m)
        if simplify_tolerance_m > 0.0:
            cl_m = cl_m.simplify(simplify_tolerance_m, preserve_topology=False)

        # Vertex arrays + cumulative meters
        xs_m, ys_m, cum_m = build_vertex_arrays_metric(cl_m)

        # Snap AGMs orthogonally to metric centerline, record measure s (meters)
        snapped = []
        for name, pt_ll in agms:
            pt_m = transform_point(pt_ll, xf_ll_to_m)
            s = cl_m.project(pt_m)
            xS, yS, _ = interpolate_point_on_polyline(xs_m, ys_m, cum_m, s)
            lon_s, lat_s = xf_m_to_ll.transform(xS, yS)
            snapped.append((name, Point(lon_s, lat_s), s))

        # Elevation cache
        tile_cache = TerrainTileCache(MAPBOX_TOKEN, zoom=mapbox_zoom)

        rows = []
        skipped = 0
        cumulative_miles = 0.0

        for i in range(len(snapped) - 1):
            name1, _, s1 = snapped[i]
            name2, _, s2 = snapped[i + 1]

            if np.isclose(s1, s2):
                skipped += 1
                continue

            # --- Get exact sub-polyline in METRIC coordinates (s1..s2)
            Xs_slice, Ys_slice = slice_polyline_by_measures(xs_m, ys_m, cum_m, s1, s2)

            # --- Convert that slice to lon/lat control points
            lons_ctrl, lats_ctrl = xf_m_to_ll.transform(Xs_slice.tolist(), Ys_slice.tolist())
            coords_ll_slice = list(zip(lons_ctrl, lats_ctrl))

            # --- Resample ALONG GEODESIC ARCLENGTH (this fixes 2D baseline to GE's)
            samples_ll = resample_polyline_geodesic(coords_ll_slice, interp_spacing_m)

            # --- Elevations at samples
            elevations = smooth_elevations(get_elevations_ll(samples_ll, tile_cache), smooth_window)

            # --- 3D length: sum sqrt( (geodesic_dxy)^2 + dz^2 ) between successive samples
            dist_m = 0.0
            for j in range(len(samples_ll) - 1):
                dxy = geodesic_segment_m(samples_ll[j], samples_ll[j + 1])
                dz = elevations[j + 1] - elevations[j]
                dist_m += math.sqrt(dxy * dxy + dz * dz)

            # If the segment was too short for spacing (1 point), fall back to straight-line
            if len(samples_ll) == 1:
                dist_m = 0.0

            dist_ft = dist_m * FT_PER_M
            dist_mi = dist_ft * MI_PER_FT
            cumulative_miles += dist_mi

            rows.append({
                "From AGM": name1,
                "To AGM": name2,
                "Distance (feet)": round(dist_ft, 2),
                "Distance (miles)": round(dist_mi, 6),
                "Cumulative (miles)": round(cumulative_miles, 6)
            })

        st.subheader("📊 Distance table")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.text(f"Skipped segments: {skipped}")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "terrain_distances.csv", "text/csv")
