# app.py
"""
Terrain-Aware AGM Distance Checker — densify centerline to handle tight curves
Full script: paste into your Streamlit app directory and run.
"""

import io
import math
import re
import zipfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from pyproj import Transformer, Geod

# ---------------- USER CONFIG ----------------
# HARD-CODED MAPBOX TOKEN (replace or use st.secrets if you prefer)
MAPBOX_TOKEN = "pk.eyJ1IjoibWF5ZGVuaGlsbGVyIiwiYSI6ImNtZ2ljMnN5ejA3amwyam9tNWZnYnZibWwifQ.GXoTyHdvCYtr7GvKIW9LPA"

# Chosen parameters (user requested: Mapbox Terrain-RGB, 10 m sampling, measure along centerline)
RESAMPLE_M = 10.0         # sample spacing along the densified centerline (meters)
DENSIFY_MAX_SEG_M = 10.0  # densify original vertices so no chord > this (meters). set <= RESAMPLE_M
SMOOTH_WINDOW_M = 0.0     # smoothing window in meters (0 = disabled)
MAX_SNAP_M = 200.0        # max distance (m) from AGM to centerline to allow snapping
TERRAIN_Z = 15            # Mapbox terrain-rgb zoom (15 is reasonable)
MAX_WORKERS = 8           # parallel tile fetch threads
FT_PER_M = 3.28084
MI_PER_FT = 1.0 / 5280.0
GEOD = Geod(ellps="WGS84")

st.set_page_config(page_title="Terrain-Aware AGM Distance Checker (Densified)", layout="wide")
st.title("Terrain-Aware AGM Distance Checker — Densified Centerline (better on curves)")

# ---------------- UTIL: robust kml/kmz parsing ----------------
def strip_ns_text(s: str) -> str:
    """Remove namespace declarations and prefixes to avoid XML parse errors ('unbound prefix')."""
    s2 = s
    # remove xmlns:prefix declarations
    s2 = re.sub(r'\s+xmlns:[A-Za-z0-9_:-]+="[^"]+"', '', s2)
    # remove xsi:... attributes
    s2 = re.sub(r'\s+xsi:[A-Za-z0-9_:-]+="[^"]+"', '', s2)
    # remove element prefixes like <gx: or </gx:
    s2 = re.sub(r'<(/?)[A-Za-z0-9_:-]+:([A-Za-z0-9_:-]+)', r'<\1\2', s2)
    # remove namespace prefixes in closing tags leftover
    s2 = re.sub(r'</[A-Za-z0-9_:-]+:([A-Za-z0-9_:-]+)>', r'</\1>', s2)
    return s2

def parse_kml_kmz(uploaded_file):
    """
    Return:
      agms: list of (name:str, shapely.Point(lon,lat))
      centerlines: list of shapely.LineString (lon, lat)
    Rules:
      - Looks for Folder name exactly 'AGMs' and 'CENTERLINE' (case-insensitive)
      - Ignores AGM names starting with 'SP' (case-insensitive)
      - Robust to KMZ (finds first .kml inside)
    """
    data = None
    if uploaded_file.name.lower().endswith(".kmz"):
        with zipfile.ZipFile(uploaded_file) as zf:
            kml_name = next((n for n in zf.namelist() if n.lower().endswith(".kml")), None)
            if not kml_name:
                return [], []
            data = zf.read(kml_name).decode("utf-8", errors="ignore")
    else:
        data = uploaded_file.read().decode("utf-8", errors="ignore")

    data = strip_ns_text(data)

    # Use a lightweight xml parse that tolerates content; we will search with regex for coords
    # Find AGMs folder blocks and CENTERLINE folder blocks
    agms = []
    centerlines = []

    # Find <Folder>...</Folder> blocks
    folder_pattern = re.compile(r'<Folder\b[^>]*>(.*?)</Folder>', re.DOTALL | re.IGNORECASE)
    name_pattern = re.compile(r'<name>\s*([^<]+?)\s*</name>', re.IGNORECASE)
    placemark_pattern = re.compile(r'<Placemark\b[^>]*>(.*?)</Placemark>', re.DOTALL | re.IGNORECASE)
    point_coords_pat = re.compile(r'<Point[^>]*>.*?<coordinates>([^<]+)</coordinates>.*?</Point>', re.DOTALL | re.IGNORECASE)
    linestring_coords_pat = re.compile(r'<LineString[^>]*>.*?<coordinates>(.*?)</coordinates>.*?</LineString>', re.DOTALL | re.IGNORECASE)

    for fmatch in folder_pattern.finditer(data):
        ftext = fmatch.group(1)
        # get folder name
        fname_m = name_pattern.search(ftext)
        fname = fname_m.group(1).strip() if fname_m else ""
        fname_up = fname.upper()
        if fname_up == "AGMS":
            for pm in placemark_pattern.finditer(ftext):
                p_text = pm.group(1)
                # name
                nm_m = name_pattern.search(p_text)
                if not nm_m:
                    continue
                nm = nm_m.group(1).strip()
                if nm.upper().startswith("SP"):
                    continue
                # point
                pc_m = point_coords_pat.search(p_text)
                if not pc_m:
                    # sometimes coordinates are directly under Placemark/coordinates (fallback)
                    pc_alt = re.search(r'<coordinates>([^<]+)</coordinates>', p_text, re.IGNORECASE)
                    if not pc_alt:
                        continue
                    coords_txt = pc_alt.group(1).strip()
                else:
                    coords_txt = pc_m.group(1).strip()
                try:
                    lon, lat, *_ = [float(x) for x in coords_txt.split(",") if x.strip() != ""]
                except Exception:
                    continue
                agms.append((nm, Point(lon, lat)))
        elif fname_up == "CENTERLINE":
            for pm in placemark_pattern.finditer(ftext):
                p_text = pm.group(1)
                ls_m = linestring_coords_pat.search(p_text)
                if not ls_m:
                    continue
                coords_txt = ls_m.group(1).strip()
                pts = []
                for c in re.split(r'\s+', coords_txt.strip()):
                    if not c:
                        continue
                    parts = c.split(",")
                    if len(parts) < 2:
                        continue
                    try:
                        lon = float(parts[0]); lat = float(parts[1])
                        pts.append((lon, lat))
                    except Exception:
                        continue
                if len(pts) >= 2:
                    centerlines.append(LineString(pts))

    # sort AGMs numerically where possible (keeps earlier behavior)
    def agm_sort_key(a):
        name = a[0]
        digits = ''.join([ch for ch in name if ch.isdigit()])
        return int(digits) if digits else -1
    agms.sort(key=agm_sort_key)
    return agms, centerlines

# ---------------- TERRAIN CACHE (Mapbox Terrain-RGB) ----------------
def lonlat_to_tile_xyz(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return int(x), int(y), x, y

def pixel_coords_in_tile(x_float, y_float):
    # Mapbox tiles 256x256; using pixel coords [0..255]
    x_pix = int((x_float - math.floor(x_float)) * 255.0)
    y_pix = int((y_float - math.floor(y_float)) * 255.0)
    x_pix = max(0, min(255, x_pix))
    y_pix = max(0, min(255, y_pix))
    return x_pix, y_pix

class TerrainCache:
    def __init__(self, token, z=TERRAIN_Z, max_workers=MAX_WORKERS):
        self.token = token
        self.z = int(z)
        self.cache = {}  # (z,x,y) -> numpy array (H,W,3)
        self.max_workers = max_workers

    def _tile_url(self, z, x, y):
        # Mapbox Terrain-RGB endpoint
        return f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"

    def fetch_tile(self, z, x, y):
        key = (z, x, y)
        if key in self.cache:
            return self.cache[key]
        url = self._tile_url(z, x, y)
        try:
            r = requests.get(url, params={"access_token": self.token}, timeout=12)
        except Exception as e:
            # print failure and return None
            st.debug(f"[Mapbox fetch error] {e} for {z}/{x}/{y}")
            return None
        if r.status_code != 200:
            st.debug(f"[Mapbox error] {r.status_code} for tile {z}/{x}/{y}")
            return None
        try:
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)
            self.cache[key] = arr
            return arr
        except Exception as e:
            st.debug(f"[Mapbox tile decode error] {e}")
            return None

    def prefetch_tiles(self, lonlats):
        # Build set of (z,x,y) tiles needed
        tiles = set()
        n = 2 ** self.z
        for lon, lat in lonlats:
            x_f = (lon + 180.0) / 360.0 * n
            lat_r = math.radians(lat)
            y_f = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
            tiles.add((self.z, int(math.floor(x_f)), int(math.floor(y_f))))
        if not tiles:
            return
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self.fetch_tile, *t) for t in tiles]
            for _ in as_completed(futures):
                pass

    @staticmethod
    def decode_rgb_triplet(trip):
        # trip is (R,G,B) array-like (ints)
        r, g, b = int(trip[0]), int(trip[1]), int(trip[2])
        return -10000.0 + (r * 256 * 256 + g * 256 + b) * 0.1

    def elevations_bulk(self, lons, lats):
        """Return numpy array of elevations for arrays of lons/lats (same length)."""
        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)
        out = np.full(lons.shape, np.nan, dtype=float)
        if lons.size == 0:
            return out
        z = self.z
        n = 2 ** z
        xt = (lons + 180.0) / 360.0 * n
        lat_r = np.radians(lats)
        yt = (1.0 - np.log(np.tan(lat_r) + 1.0 / np.cos(lat_r)) / math.pi) / 2.0 * n

        x_tile = np.floor(xt).astype(int)
        y_tile = np.floor(yt).astype(int)
        # fractional pixel positions in 0..255 (using 256 px -> 0..255 indices)
        xp = (xt - x_tile) * 255.0
        yp = (yt - y_tile) * 255.0

        for i in range(len(lons)):
            key = (z, int(x_tile[i]), int(y_tile[i]))
            arr = self.fetch_tile(*key)
            if arr is None:
                continue
            # bilinear interpolation of surrounding pixel values
            x0 = int(min(254, max(0, int(math.floor(xp[i])))))
            y0 = int(min(254, max(0, int(math.floor(yp[i])))))
            x1 = x0 + 1
            y1 = y0 + 1
            dx = xp[i] - x0
            dy = yp[i] - y0
            try:
                p00 = self.decode_rgb_triplet(arr[y0, x0])
                p10 = self.decode_rgb_triplet(arr[y0, x1])
                p01 = self.decode_rgb_triplet(arr[y1, x0])
                p11 = self.decode_rgb_triplet(arr[y1, x1])
                val = p00 * (1 - dx) * (1 - dy) + p10 * dx * (1 - dy) + p01 * (1 - dx) * dy + p11 * dx * dy
                out[i] = float(val)
            except Exception:
                continue
        return out

# ---------------- GEODESIC DENSIFY + SAMPLING ----------------
def densify_linestring_geodesic(ls: LineString, max_seg_m=DENSIFY_MAX_SEG_M):
    """
    Densify a LineString geodesically so that no segment is longer than max_seg_m.
    Returns new LineString with more vertices (lon, lat).
    """
    coords = list(ls.coords)
    new_pts = []
    for a, b in zip(coords[:-1], coords[1:]):
        lon1, lat1 = a
        lon2, lat2 = b
        # distance (m) along geodesic between endpoints
        _, _, dist = GEOD.inv(lon1, lat1, lon2, lat2)
        # always include start point
        new_pts.append((lon1, lat1))
        if dist <= max_seg_m:
            continue
        nseg = int(math.ceil(dist / max_seg_m))
        # interpolate nseg-1 interior points using geod.npts
        # pyproj.Geod.npts returns list of (lon, lat) for equally spaced points between endpoints (exclusive)
        try:
            npts = GEOD.npts(lon1, lat1, lon2, lat2, nseg - 1)
            for p in npts:
                new_pts.append((p[0], p[1]))
        except Exception:
            # fallback: linear interpolation in lon/lat if npts fails
            for k in range(1, nseg):
                f = k / float(nseg)
                new_pts.append((lon1 + (lon2 - lon1) * f, lat1 + (lat2 - lat1) * f))
    # add final endpoint
    new_pts.append(coords[-1])
    return LineString(new_pts)

def sample_along_linestring(ls: LineString, spacing_m=RESAMPLE_M):
    """
    Produce a list of (lon, lat) sampled along geodesic length at approx spacing_m.
    We compute cumulative geodesic lengths for dense vertex list and interpolate positions at target distances.
    """
    coords = list(ls.coords)
    if len(coords) < 2:
        return []

    # compute cumulative geodesic distances along the vertex chain
    seg_lengths = []
    for (lon1, lat1), (lon2, lat2) in zip(coords[:-1], coords[1:]):
        _, _, d = GEOD.inv(lon1, lat1, lon2, lat2)
        seg_lengths.append(d)
    cum = [0.0]
    for s in seg_lengths:
        cum.append(cum[-1] + s)
    total = cum[-1]
    if total == 0:
        return [coords[0]]
    # target distances along path
    num_steps = max(1, int(math.ceil(total / spacing_m)))
    targets = [i * spacing_m for i in range(num_steps)]
    if targets[-1] < total:
        targets.append(total)

    # walk along segments and interpolate
    sampled = []
    seg_idx = 0
    for t in targets:
        # advance seg_idx until target within this segment
        while seg_idx < len(seg_lengths) and t > cum[seg_idx + 1]:
            seg_idx += 1
        if seg_idx >= len(seg_lengths):
            sampled.append(coords[-1])
            continue
        seg_start = cum[seg_idx]
        seg_len = seg_lengths[seg_idx]
        if seg_len == 0:
            frac = 0.0
        else:
            frac = (t - seg_start) / seg_len
        lon1, lat1 = coords[seg_idx]
        lon2, lat2 = coords[seg_idx + 1]
        # use geodesic direct solve to compute point at fraction along this segment
        # compute forward azimuth and distance to point
        az12, az21, _ = GEOD.inv(lon1, lat1, lon2, lat2)
        dist_to_point = frac * seg_len
        try:
            lon_p, lat_p, _ = GEOD.fwd(lon1, lat1, az12, dist_to_point)
            sampled.append((lon_p, lat_p))
        except Exception:
            # fallback linear
            sampled.append((lon1 + (lon2 - lon1) * frac, lat1 + (lat2 - lat1) * frac))
    return sampled

# ---------------- MISC ----------------
def moving_average(arr, window_m, spacing_m):
    if window_m <= 0:
        return arr
    arr = np.asarray(arr, dtype=float)
    n = max(1, int(round(window_m / max(spacing_m, 1e-9))))
    if n % 2 == 0:
        n += 1
    kernel = np.ones(n, dtype=float) / float(n)
    # use 'same' convolution (pad)
    return np.convolve(arr, kernel, mode="same")

# ---------------- MAIN STREAMLIT APP ----------------
uploaded = st.file_uploader("Upload KML or KMZ", type=["kml", "kmz"])
if not uploaded:
    st.info("Upload a KML or KMZ that contains an AGMs folder and a CENTERLINE folder.")
    st.stop()

# parse
with st.spinner("Parsing KML/KMZ..."):
    agms, centerlines = parse_kml_kmz(uploaded)

st.write(f"{len(agms)} AGMs | {len(centerlines)} centerline part(s)")

if not agms or not centerlines:
    st.warning("Need both AGMs and CENTERLINE. The parser looks for Folder named 'AGMs' and 'CENTERLINE'.")
    st.stop()

# pick first centerline part for now (keeps behavior)
centerline_ll = centerlines[0]  # lon,lat coordinates

# densify centerline geodesically to ensure tight curves are represented
with st.spinner("Densifying centerline to handle curves (this may add many vertices)..."):
    dense_centerline = densify_linestring_geodesic(centerline_ll, max_seg_m=DENSIFY_MAX_SEG_M)

# sample positions along densified line at RESAMPLE_M
with st.spinner("Sampling along centerline..."):
    sampled_ll = sample_along_linestring(dense_centerline, spacing_m=RESAMPLE_M)
    if len(sampled_ll) < 2:
        st.error("Centerline sampling produced insufficient points.")
        st.stop()

# prepare terrain cache and prefetch tiles for sampled points (parallel)
terrain = TerrainCache(MAPBOX_TOKEN, z=TERRAIN_Z, max_workers=MAX_WORKERS)
with st.spinner("Prefetching Mapbox Terrain-RGB tiles (parallel)..."):
    terrain.prefetch_tiles(sampled_ll)

# Prepare projection for snapping (use Web Mercator for metric nearest-point)
proj_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
proj_to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

# shapely LineString in metric for nearest point & snap
dense_xy = [proj_to_3857.transform(lon, lat) for lon, lat in dense_centerline.coords]
dense_line_m = LineString(dense_xy)

# Build DataFrame results
rows = []
cum_mi = 0.0

# Streamlit progress
total_pairs = max(0, len(agms) - 1)
progress = st.progress(0)
status = st.empty()

# Precompute sampled lons/lats array for elevation fetch
sampled_arr = np.array(sampled_ll)  # shape (N,2)
elevs_sampled = terrain.elevations_bulk(sampled_arr[:, 0], sampled_arr[:, 1])
if SMOOTH_WINDOW_M > 0:
    elevs_sampled = moving_average(elevs_sampled, SMOOTH_WINDOW_M, RESAMPLE_M)

# Helper: get elevation for a lon,lat by nearest sampled index (fast) or bilinear direct
def elevation_for_lonlat(lon, lat):
    # find nearest sampled point (cheap)
    dists = np.hypot(sampled_arr[:, 0] - lon, sampled_arr[:, 1] - lat)
    idx = int(np.argmin(dists))
    val = elevs_sampled[idx]
    if np.isfinite(val):
        return float(val)
    # fallback: single-point fetch
    v = terrain.elevations_bulk(np.array([lon]), np.array([lat]))
    return float(v[0]) if (v.size and np.isfinite(v[0])) else 0.0

# iterate AGM pairs
for i in range(len(agms) - 1):
    name1, p1 = agms[i]
    name2, p2 = agms[i + 1]
    status.text(f"Calculating {name1} → {name2} ({i+1}/{total_pairs})")
    progress.progress((i + 1) / max(1, total_pairs))

    # Snap AGM points to nearest point on centerline (in metric)
    p1_m = Point(proj_to_3857.transform(p1.x, p1.y))
    p2_m = Point(proj_to_3857.transform(p2.x, p2.y))
    # distances
    d1 = p1_m.distance(dense_line_m)
    d2 = p2_m.distance(dense_line_m)
    if d1 > MAX_SNAP_M or d2 > MAX_SNAP_M:
        # skip if too far
        st.warning(f"Skipping pair {name1}→{name2}: AGM too far from centerline (d1={d1:.1f} m d2={d2:.1f} m).")
        continue
    # nearest points on dense_line_m
    # shapely nearest_points returns (p_on_line, p_original) or vice versa; use the version that returns pair
    nearest1 = nearest_points(dense_line_m, p1_m)[0]
    nearest2 = nearest_points(dense_line_m, p2_m)[0]
    # convert snapped positions back to lon/lat
    lon1_snap, lat1_snap = proj_to_ll.transform(nearest1.x, nearest1.y)
    lon2_snap, lat2_snap = proj_to_ll.transform(nearest2.x, nearest2.y)

    # find indices along sampled_ll nearest to snapped points to get subarray of sampled points between them
    sampled_lons = sampled_arr[:, 0]; sampled_lats = sampled_arr[:, 1]
    # compute geodesic distances along path cumulative to find positions
    # find index of nearest sampled point
    idx1 = int(np.argmin(np.hypot(sampled_lons - lon1_snap, sampled_lats - lat1_snap)))
    idx2 = int(np.argmin(np.hypot(sampled_lons - lon2_snap, sampled_lats - lat2_snap)))
    if idx1 == idx2:
        # ensure at least one step: include neighbor if possible
        if idx2 + 1 < len(sampled_arr):
            idx2 = idx2 + 1
        elif idx1 - 1 >= 0:
            idx1 = idx1 - 1
        else:
            # degenerate small segment -> compute direct geodesic
            h_dist = GEOD.inv(lon1_snap, lat1_snap, lon2_snap, lat2_snap)[2]
            e1 = elevation_for_lonlat(lon1_snap, lat1_snap)
            e2 = elevation_for_lonlat(lon2_snap, lat2_snap)
            dz = e2 - e1
            dist3d = math.hypot(h_dist, dz)
            feet = dist3d * FT_PER_M
            miles = feet * MI_PER_FT
            cum_mi += miles
            rows.append({
                "From AGM": name1,
                "To AGM": name2,
                "Distance (feet)": round(feet, 2),
                "Distance (miles)": round(miles, 6),
                "Cumulative (miles)": round(cum_mi, 6)
            })
            continue

    # Ensure indices in order
    if idx1 < idx2:
        seg_idxs = range(idx1, idx2 + 1)
    else:
        seg_idxs = range(idx2, idx1 + 1)

    seg_lons = sampled_arr[list(seg_idxs), 0]
    seg_lats = sampled_arr[list(seg_idxs), 1]

    # compute elevations for these sampled points (use precomputed elevs_sampled slice)
    seg_elevs = elevs_sampled[list(seg_idxs)]
    # If any NaN, fill by fetching individual elevations
    nan_mask = ~np.isfinite(seg_elevs)
    if np.any(nan_mask):
        for k in np.where(nan_mask)[0]:
            lonx = seg_lons[k]; latx = seg_lats[k]
            seg_elevs[k] = elevation_for_lonlat(lonx, latx)

    # compute 3D distance along consecutive pairs
    dist_m = 0.0
    for k in range(len(seg_lons) - 1):
        lon_a, lat_a = seg_lons[k], seg_lats[k]
        lon_b, lat_b = seg_lons[k + 1], seg_lats[k + 1]
        _, _, horiz = GEOD.inv(lon_a, lat_a, lon_b, lat_b)
        dz = seg_elevs[k + 1] - seg_elevs[k]
        dist_m += math.hypot(horiz, dz)

    feet = dist_m * FT_PER_M
    miles = feet * MI_PER_FT
    cum_mi += miles
    rows.append({
        "From AGM": name1,
        "To AGM": name2,
        "Distance (feet)": round(feet, 2),
        "Distance (miles)": round(miles, 6),
        "Cumulative (miles)": round(cum_mi, 6)
    })

status.text("Complete.")
progress.progress(1.0)

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "terrain_distances.csv", "text/csv")
