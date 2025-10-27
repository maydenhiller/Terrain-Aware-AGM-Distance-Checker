# Terrain-Aware AGM Distance Calculator — Accurate 3D
# Key fixes vs prior builds:
#  - DEM-aware sampling step (>= ~1.5× meters-per-pixel) to avoid noise-driven path inflation
#  - Two-stage elevation smoothing (median -> moving average)
#  - Optional supersampled elevation per point (center + 4 offsets)
#  - Exact linear-referenced slicing on a single, correct centerline part
#  - AGM snap is orthogonal in a local metric CRS

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
from pyproj import CRS, Transformer

# =========================
# CONFIG / UI
# =========================

st.set_page_config(page_title="Terrain-Aware AGM Distance Calculator", layout="wide")
st.title("Terrain-Aware AGM Distance Calculator (Accurate 3D with DEM-aware sampling)")

MAPBOX_TOKEN = st.secrets["mapbox"]["token"]
TERRAIN_TILE_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"

with st.sidebar:
    st.header("Settings")
    mapbox_zoom = st.slider("Terrain tile zoom", 15, 17, 17)
    interp_spacing_m_ui = st.slider(
        "Requested sampling spacing along path (m)",
        0.5, 10.0, 1.0, 0.5,
        help="Actual spacing will be max(this, DEM-based min) to avoid elevation noise."
    )
    smooth_median_window = st.slider(
        "Elevation median window (samples)",
        1, 25, 9, 2,
        help="Robust denoising. Use odd numbers; 9–15 is a good start. 1 disables."
    )
    smooth_mean_window = st.slider(
        "Elevation moving-average window (samples)",
        1, 25, 7, 2,
        help="Second pass smoothing. 5–11 is typical. 1 disables."
    )
    supersample = st.checkbox(
        "Supersample elevation per point (center + 4 offsets)", value=True,
        help="Stabilizes elevations at pixel/tile seams (light extra cost)."
    )
    simplify_tolerance_m = st.slider(
        "Simplify each centerline part (m)",
        0.0, 5.0, 0.0, 0.5,
        help="Optional: remove micro-wiggles. 0 = off."
    )
    snap_max_offset_m = st.slider(
        "Max AGM snap offset (m)", 5.0, 150.0, 60.0, 5.0,
        help="Skip a segment if an AGM lies farther than this from every centerline part."
    )
    st.caption("Best results: zoom=17, spacing≈auto, median 9–15, mean 5–11, supersample ON.")

FT_PER_M = 3.28084
MI_PER_FT = 1.0 / 5280.0
INITIAL_RES_M_PER_PX = 156543.03392  # Web Mercator @ equator, z=0

# =========================
# KML / KMZ PARSER
# =========================

def agm_sort_key(name_geom):
    name = name_geom[0]
    base_digits = ''.join(filter(str.isdigit, name))
    base = int(base_digits) if base_digits else -1
    suffix = ''.join(filter(str.isalpha, name)).upper()
    return (base, suffix)

def parse_kml_kmz(uploaded_file):
    """Returns: AGMs[(name, Point lon/lat)], CenterlineParts[List[LineString lon/lat]]"""
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
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    agms = []
    parts = []  # collect each Placemark polyline separately

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
                    parts.append(LineString(pts))

    agms.sort(key=agm_sort_key)
    return agms, parts

# =========================
# CRS / TRANSFORMS
# =========================

def get_local_utm_crs(lines_ll) -> CRS:
    """Pick a local UTM based on all centerline parts."""
    xs = []
    ys = []
    for ls in lines_ll:
        xs.extend([c[0] for c in ls.coords])
        ys.extend([c[1] for c in ls.coords])
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
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

# =========================
# LINEAR REFERENCING (metric)
# =========================

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

def linear_reference_samples(xs, ys, cum, s0, s1, spacing_m):
    """Even samples along metric polyline between s0..s1.
       Horizontal sum will equal |s1 - s0| exactly."""
    s_lo, s_hi = (s0, s1) if s0 <= s1 else (s1, s0)
    L = float(abs(s1 - s0))
    if L <= 0:
        return np.array([]), np.array([]), np.array([])

    targets = np.arange(0.0, L, float(spacing_m))
    if targets.size == 0 or targets[-1] < L:
        targets = np.append(targets, L)

    t_abs = s_lo + targets
    idxs = np.searchsorted(cum, t_abs, side="right") - 1
    idxs = np.clip(idxs, 0, len(xs) - 2)

    seg_len = (cum[idxs + 1] - cum[idxs])
    seg_len = np.where(seg_len <= 0, 1.0, seg_len)
    frac = (t_abs - cum[idxs]) / seg_len

    dx = (xs[idxs + 1] - xs[idxs])
    dy = (ys[idxs + 1] - ys[idxs])
    Xs = xs[idxs] + frac * dx
    Ys = ys[idxs] + frac * dy

    return Xs, Ys, targets  # metric XY; local cumulative (0..L)

# =========================
# DEM helpers (Terrain-RGB)
# =========================

def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return int(x), int(y), x, y

def decode_terrain_rgb(r, g, b):
    return -10000.0 + (r * 256.0 * 256.0 + g * 256.0 + b) * 0.1

def meters_per_pixel(lat_deg, z):
    # Web Mercator ground resolution
    return (INITIAL_RES_M_PER_PX * math.cos(math.radians(lat_deg))) / (2 ** z)

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

# Supersampled elevation around a point (center + 4 offsets at ~0.6 px)
def elevation_supersampled(lon, lat, cache: TerrainTileCache, xf_ll_to_m: Transformer, xf_m_to_ll: Transformer):
    z = cache.zoom
    mpp = meters_per_pixel(lat, z)
    # offset radius ~ 0.6 px in meters
    r = 0.6 * mpp
    # transform to metric, offset, back to lon/lat
    x, y = xf_ll_to_m.transform(lon, lat)
    samples = [(lon, lat)]
    for dx, dy in [(r, 0), (-r, 0), (0, r), (0, -r)]:
        lx, ly = xf_m_to_ll.transform(x + dx, y + dy)
        samples.append((lx, ly))
    vals = []
    for lo, la in samples:
        e = cache.elevation_at_bilinear(lo, la)
        if e is not None and np.isfinite(e):
            vals.append(e)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))

def get_elevations_ll(points_ll, cache: TerrainTileCache, xf_ll_to_m: Transformer, xf_m_to_ll: Transformer, supersample: bool):
    elevs = []
    for lon, lat in points_ll:
        if supersample:
            e = elevation_supersampled(lon, lat, cache, xf_ll_to_m, xf_m_to_ll)
        else:
            e = cache.elevation_at_bilinear(lon, lat)
            e = 0.0 if (e is None or not np.isfinite(e)) else float(e)
        elevs.append(e if e is not None else 0.0)
    return elevs

def smooth_elevations_two_stage(elevs, median_win, mean_win):
    arr = np.asarray(elevs, dtype=float)
    if median_win > 1:
        # robust median (use pandas centered rolling median)
        s = pd.Series(arr)
        arr = s.rolling(window=median_win, center=True, min_periods=1).median().to_numpy()
    if mean_win > 1:
        kernel = np.ones(mean_win) / float(mean_win)
        arr = np.convolve(arr, kernel, mode="same")
    return arr.tolist()

# =========================
# CORE LOGIC
# =========================

def pick_centerline_part_for_pair(parts_m, pt1_ll, pt2_ll, xf_ll_to_m, max_offset_m):
    """Choose the single centerline part both AGMs should snap to, minimizing total offset.
       Returns (part_index, s1, s2) or (None, None, None)."""
    best = None
    p1_m = transform_point(pt1_ll, xf_ll_to_m)
    p2_m = transform_point(pt2_ll, xf_ll_to_m)
    for idx, part_m in enumerate(parts_m):
        s1 = part_m.project(p1_m)
        s2 = part_m.project(p2_m)
        snap1 = part_m.interpolate(s1)
        snap2 = part_m.interpolate(s2)
        off1 = ((p1_m.x - snap1.x)**2 + (p1_m.y - snap1.y)**2) ** 0.5
        off2 = ((p2_m.x - snap2.x)**2 + (p2_m.y - snap2.y)**2) ** 0.5
        if off1 <= max_offset_m and off2 <= max_offset_m:
            tot = off1 + off2
            if (best is None) or (tot < best[0]):
                best = (tot, idx, s1, s2)
    if best is None:
        return None, None, None
    _, idx, s1, s2 = best
    return idx, s1, s2

def resample_along_part(part_m: LineString, s1, s2, spacing_m):
    xs, ys, cum = build_vertex_arrays_metric(part_m)
    Xs, Ys, targets = linear_reference_samples(xs, ys, cum, s1, s2, spacing_m)
    return Xs, Ys, targets

def compute_3d_distance_with_dem(Xs, Ys, targets, xf_m_to_ll, tile_cache, xf_ll_to_m, supersample_flag, med_k, mean_k, lat_hint):
    if Xs.size == 0:
        return 0.0

    # Convert to lon/lat for elevation
    lons, lats = xf_m_to_ll.transform(Xs.tolist(), Ys.tolist())
    pts_ll = list(zip(lons, lats))

    # Determine DEM-aware minimum spacing (>= ~1.5 px)
    mpp = meters_per_pixel(lat_hint, tile_cache.zoom)
    min_spacing = max(2.0, 1.5 * mpp)

    # Ensure horizontal steps are not below min_spacing by thinning the sample stream
    if len(targets) > 1:
        keep = [0]
        for i in range(1, len(targets)):
            if (targets[i] - targets[keep[-1]]) >= min_spacing or i == len(targets) - 1:
                keep.append(i)
        pts_ll = [pts_ll[i] for i in keep]
        targets = [targets[i] for i in keep]

    # Elevations (optionally supersampled), then robust two-stage smoothing
    elevs = get_elevations_ll(pts_ll, tile_cache, xf_ll_to_m, xf_m_to_ll, supersample_flag)
    elevs = smooth_elevations_two_stage(elevs, med_k, mean_k)

    # Accumulate 3D using exact metric dh from targets
    dist_m = 0.0
    for j in range(len(targets) - 1):
        dh = float(targets[j + 1] - targets[j])
        dz = float(elevs[j + 1] - elevs[j])
        dist_m += math.sqrt(dh * dh + dz * dz)

    if len(targets) <= 1:
        dist_m = 0.0
    return dist_m

# =========================
# MAIN APP
# =========================

uploaded_file = st.file_uploader("Upload KML or KMZ file", type=["kml", "kmz"])

if uploaded_file:
    agms, centerline_parts_ll = parse_kml_kmz(uploaded_file)

    st.subheader("📌 AGM summary")
    st.text(f"Total AGMs found: {len(agms)}")
    st.subheader("📈 CENTERLINE parts")
    st.text(f"Found {len(centerline_parts_ll)} centerline part(s)")

    if len(centerline_parts_ll) == 0 or len(agms) < 2:
        st.warning("Missing CENTERLINE parts or insufficient AGM points.")
    else:
        # Local metric CRS from all parts
        try:
            crs_metric = get_local_utm_crs(centerline_parts_ll)
        except Exception:
            crs_metric = CRS.from_epsg(5070)  # fallback
        xf_ll_to_m = transformer_ll_to(crs_metric)
        xf_m_to_ll = transformer_to_ll(crs_metric)

        # Transform/simplify parts -> metric
        parts_m = []
        for ls in centerline_parts_ll:
            lm = transform_linestring(ls, xf_ll_to_m)
            if simplify_tolerance_m > 0.0:
                lm = lm.simplify(simplify_tolerance_m, preserve_topology=False)
            if lm is not None and lm.length > 0 and len(lm.coords) >= 2:
                parts_m.append(lm)

        if not parts_m:
            st.warning("All centerline parts were degenerate after transform/simplify.")
        else:
            tile_cache = TerrainTileCache(MAPBOX_TOKEN, zoom=mapbox_zoom)

            rows = []
            skipped = 0
            cumulative_miles = 0.0

            # Use center latitude as DEM resolution hint
            all_lats = [c[1] for p in centerline_parts_ll for c in p.coords]
            lat_hint = float(np.mean(all_lats)) if all_lats else 35.0

            for i in range(len(agms) - 1):
                name1, pt1_ll = agms[i]
                name2, pt2_ll = agms[i + 1]

                part_idx, s1, s2 = pick_centerline_part_for_pair(
                    parts_m, pt1_ll, pt2_ll, xf_ll_to_m, snap_max_offset_m
                )
                if part_idx is None:
                    skipped += 1
                    continue

                part_m = parts_m[part_idx]

                # DEM-aware spacing: do not over-sample
                mpp = meters_per_pixel(lat_hint, mapbox_zoom)
                recommended_min = max(2.0, 1.5 * mpp)
                spacing_used = max(float(interp_spacing_m_ui), recommended_min)

                # Linear-referenced sampling along the chosen part
                Xs, Ys, targets = resample_along_part(part_m, s1, s2, spacing_used)
                if Xs.size == 0 or len(targets) < 1:
                    skipped += 1
                    continue

                # Accurate 3D accumulation with smoothing & supersampling
                dist_m = compute_3d_distance_with_dem(
                    Xs, Ys, targets,
                    xf_m_to_ll, tile_cache,
                    xf_ll_to_m,
                    supersample,
                    int(smooth_median_window) | 1,  # ensure odd
                    int(smooth_mean_window),
                    lat_hint
                )

                # If spacing produced a single sample, fallback to exact 2D along the part
                if len(targets) == 1:
                    dist_m = abs(s2 - s1)

                dist_ft = dist_m * FT_PER_M
                dist_mi = dist_ft * MI_PER_FT
                cumulative_miles += dist_mi

                rows.append({
                    "From AGM": name1,
                    "To AGM": name2,
                    "Distance (feet)": round(dist_ft, 2),
                    "Distance (miles)": round(dist_mi, 6),
                    "Cumulative (miles)": round(cumulative_miles, 6),
                    "Centerline part #": part_idx,
                    "Spacing used (m)": round(spacing_used, 3),
                })

            st.subheader("📊 Distance table")
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.text(f"Skipped segments: {skipped}")

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "terrain_distances.csv", "text/csv")
