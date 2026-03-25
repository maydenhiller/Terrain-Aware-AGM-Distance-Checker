import io
import math
import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image


EARTH_R_FT = 20925524.9
TILE_ZOOM = 14
TILE_SIZE = 256

# Centerline segment length: 5 m ≈ 16.4 ft. Balances path fidelity with speed (was 3.28 ft).
MAX_CENTERLINE_SEGMENT_FT = 16.4
# Sample elevation every this many ft along 2D chainage (higher = fewer Mapbox tiles)
ELEVATION_SAMPLE_INTERVAL_FT = 2000.0
# Odd window >= 3; moving average on sampled-then-interpolated elevations to reduce DEM pixel noise (0 = off)
ELEVATION_SMOOTH_WINDOW = 15
TERRAIN_MAX_WORKERS = 10

# Polyline merge: endpoints closer than this are treated as one junction (ft)
MERGE_ENDPOINT_TOL_FT = 150.0
SP_PREFIX = "SP"

NEGATIVE_KEYWORDS = [
    "weld",
    "riser",
    "launch bo",
    "launch blowoff",
    "launch door",
    "pig sig",
    "u/s blowoff",
    "door",
    "nominal",
]


def _get_mapbox_token() -> str | None:
    try:
        token = st.secrets["mapbox"]["token"]
        return token.strip() if token else None
    except Exception:
        return None


# ---------------- GEO ----------------


def haversine_ft(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_R_FT * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _to_local_xy_ft(lat, lon, ref_lat_rad):
    x = math.radians(lon) * EARTH_R_FT * math.cos(ref_lat_rad)
    y = math.radians(lat) * EARTH_R_FT
    return np.array([x, y], dtype=float)


def _from_local_xy_ft(xy, ref_lat_rad):
    x, y = float(xy[0]), float(xy[1])
    lat = math.degrees(y / EARTH_R_FT)
    denom = EARTH_R_FT * math.cos(ref_lat_rad)
    lon = math.degrees(x / denom) if denom != 0 else 0.0
    return lat, lon


# -------- CENTERLINE DENSIFY --------


def dedupe_centerline(line, min_sep_ft=1.0):
    """Remove consecutive points that are within min_sep_ft so no zero-length segments."""
    if not line or len(line) < 2:
        return line
    out = [line[0]]
    for i in range(1, len(line)):
        if haversine_ft(out[-1][0], out[-1][1], line[i][0], line[i][1]) >= min_sep_ft:
            out.append(line[i])
    if len(out) < 2:
        return line
    return out


def densify_centerline(line, max_seg_ft, elevations=None):
    """Insert points so no segment exceeds max_seg_ft. If elevations (len=len(line)) given, return (out_pts, out_elevs) with interpolated elevs."""
    if not line or len(line) < 2 or max_seg_ft <= 0:
        return (line, elevations) if elevations is not None and len(elevations) == len(line) else line
    out = [line[0]]
    out_elev = [elevations[0]] if elevations is not None and len(elevations) == len(line) else None
    for i in range(len(line) - 1):
        a_lat, a_lon = line[i]
        b_lat, b_lon = line[i + 1]
        elev_a = elevations[i] if out_elev is not None else None
        elev_b = elevations[i + 1] if out_elev is not None else None
        h = haversine_ft(a_lat, a_lon, b_lat, b_lon)
        if h <= max_seg_ft:
            out.append((b_lat, b_lon))
            if out_elev is not None:
                out_elev.append(elev_b)
            continue
        n = max(2, int(math.ceil(h / max_seg_ft)))
        for k in range(1, n):
            t = k / n
            out.append((a_lat + t * (b_lat - a_lat), a_lon + t * (b_lon - a_lon)))
            if out_elev is not None:
                out_elev.append(elev_a + t * (elev_b - elev_a))
        out.append((b_lat, b_lon))
        if out_elev is not None:
            out_elev.append(elev_b)
    if out_elev is not None:
        return out, out_elev
    return out


# -------- PROJECT POINT ONTO LINE SEGMENTS --------


def project_to_line(lat, lon, line, min_seg_index=0, min_t=0.0):
    """Snap (lat, lon) to closest point on centerline. Optional min_seg_index/min_t restrict to
    'forward' along the polyline (legacy); use defaults for unconstrained closest point."""
    best_dist = float("inf")
    best_index = 0
    best_t = 0.0

    for i in range(len(line) - 1):
        if i < min_seg_index:
            continue
        if i == min_seg_index and min_t >= 1.0:
            continue
        a_lat, a_lon = line[i]
        b_lat, b_lon = line[i + 1]
        ref_lat = math.radians((a_lat + b_lat) / 2.0)

        A = _to_local_xy_ft(a_lat, a_lon, ref_lat)
        B = _to_local_xy_ft(b_lat, b_lon, ref_lat)
        P = _to_local_xy_ft(lat, lon, ref_lat)

        AB = B - A
        denom = float(np.dot(AB, AB))
        if denom == 0:
            t = 0.0
        else:
            t = float(np.dot(P - A, AB) / denom)
            t = max(0.0, min(1.0, t))
        if i == min_seg_index and t < min_t - 1e-9:
            continue

        proj_xy = A + t * AB
        proj_lat, proj_lon = _from_local_xy_ft(proj_xy, ref_lat)
        d = haversine_ft(lat, lon, proj_lat, proj_lon)

        if d < best_dist:
            best_dist = d
            best_index = i
            best_t = t

    return best_index, best_t


def point_on_line(line, proj):
    """Return (lat, lon) at segment index and t in [0,1]."""
    idx = max(0, min(int(proj[0]), len(line) - 2))
    t = max(0.0, min(1.0, float(proj[1])))
    a, b = line[idx], line[idx + 1]
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


# -------- STATIONING (3D TERRAIN PATH — Google Earth style) --------


def compute_stationing_3d(line, elevations):
    """Cumulative **slope** distance: each segment = sqrt(horizontal² + Δelev²).

    Horizontal leg uses ellipsoidal great-circle distance (Haversine on WGS84-like sphere) in feet;
    that is the ground distance between vertices, not a separate “Haversine output.”
    """
    seg_len_3d = []
    for i in range(len(line) - 1):
        h = haversine_ft(*line[i], *line[i + 1])
        v = elevations[i + 1] - elevations[i]
        seg_len_3d.append(math.sqrt(h * h + v * v))

    cum = [0.0]
    for L in seg_len_3d:
        cum.append(cum[-1] + L)
    return seg_len_3d, cum


def station_at_3d(proj, line, elevations, cum_vertex):
    """3D chainage to projected point. Uses horizontal arc length along each segment (Haversine A→P),
    not planar projection parameter × segment length — that mismatch inflated segment gaps vs Google Earth."""
    idx, _t = proj
    n = len(line)
    if n < 2:
        return 0.0
    idx = int(idx)
    idx = max(0, min(idx, n - 2))
    a = line[idx]
    b = line[idx + 1]
    za = elevations[idx]
    zb = elevations[idx + 1]
    p = point_on_line(line, proj)
    H = haversine_ft(a[0], a[1], b[0], b[1])
    if H < 1e-6:
        return cum_vertex[idx]
    h_ap = haversine_ft(a[0], a[1], p[0], p[1])
    frac = min(1.0, max(0.0, h_ap / H))
    zp = za + frac * (zb - za)
    partial_3d = math.sqrt(h_ap * h_ap + (zp - za) * (zp - za))
    return cum_vertex[idx] + partial_3d


def station_at_horizontal(proj, line, cum_vertex_2d):
    """Horizontal chainage to projected point (for comparison column)."""
    idx, _t = proj
    n = len(line)
    if n < 2:
        return 0.0
    idx = int(idx)
    idx = max(0, min(idx, n - 2))
    a = line[idx]
    b = line[idx + 1]
    p = point_on_line(line, proj)
    H = haversine_ft(a[0], a[1], b[0], b[1])
    if H < 1e-6:
        return cum_vertex_2d[idx]
    h_ap = haversine_ft(a[0], a[1], p[0], p[1])
    return cum_vertex_2d[idx] + min(H, max(0.0, h_ap))


def compute_stationing_2d_cum(line):
    """Cumulative horizontal distance to each vertex (for station_at_horizontal)."""
    seg = [haversine_ft(*line[i], *line[i + 1]) for i in range(len(line) - 1)]
    cum = [0.0]
    for L in seg:
        cum.append(cum[-1] + L)
    return cum


def path_length_along_centerline(proj_a, proj_b, seg_len_3d, total_length):
    """3D distance along the polyline between two projected points (shorter of two ways on a loop)."""
    idx_a, t_a = int(proj_a[0]), float(proj_a[1])
    idx_b, t_b = int(proj_b[0]), float(proj_b[1])
    idx_a = max(0, min(idx_a, len(seg_len_3d) - 1))
    idx_b = max(0, min(idx_b, len(seg_len_3d) - 1))
    t_a = max(0.0, min(1.0, t_a))
    t_b = max(0.0, min(1.0, t_b))

    if idx_a == idx_b:
        same_seg = abs(t_b - t_a) * seg_len_3d[idx_a]
        other = total_length - same_seg
        return min(same_seg, other)

    if idx_a < idx_b:
        forward = (1.0 - t_a) * seg_len_3d[idx_a]
        for i in range(idx_a + 1, idx_b):
            forward += seg_len_3d[i]
        forward += t_b * seg_len_3d[idx_b]
        backward = total_length - forward
        return min(forward, backward)
    else:
        backward = (1.0 - t_b) * seg_len_3d[idx_b]
        for i in range(idx_b + 1, idx_a):
            backward += seg_len_3d[i]
        backward += t_a * seg_len_3d[idx_a]
        forward = total_length - backward
        return min(forward, backward)


def _smooth_elevations_1d(elev: list[float], window: int) -> list[float]:
    """Moving average (odd window) to reduce DEM speckle before 3D length accumulation."""
    if window < 3 or len(elev) < window or (window % 2) == 0:
        return elev
    half = window // 2
    out = []
    n = len(elev)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(float(sum(elev[lo:hi]) / (hi - lo)))
    return out


# ---------------- MAPBOX TERRAIN ----------------


tile_cache: dict[tuple[int, int, int], Image.Image] = {}


def _world_pixel_xy(lat, lon, z):
    n = 2**z
    x = (lon + 180.0) / 360.0 * n * TILE_SIZE
    lat_rad = math.radians(lat)
    y = (
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
        * TILE_SIZE
    )
    return x, y


def tile_xy(lat, lon, z):
    wx, wy = _world_pixel_xy(lat, lon, z)
    x = int(wx // TILE_SIZE)
    y = int(wy // TILE_SIZE)
    return x, y


def get_tile(z, x, y, token: str):
    key = (z, x, y)
    if key in tile_cache:
        return tile_cache[key]

    url = f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw?access_token={token}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGBA")
    tile_cache[key] = img
    return img


def _tile_keys_for_line(line, z=TILE_ZOOM):
    """Set of (z, x, y) tile keys needed for all points on the line."""
    keys = set()
    for lat, lon in line:
        wx, wy = _world_pixel_xy(lat, lon, z)
        tx, ty = int(wx // TILE_SIZE), int(wy // TILE_SIZE)
        keys.add((z, tx, ty))
    return keys


def _fetch_tile(args):
    z, x, y, token = args
    get_tile(z, x, y, token)
    return (z, x, y)


def _rgb_to_elevation_ft(r, g, b) -> float:
    meters = -10000.0 + (float(r) * 256.0 * 256.0 + float(g) * 256.0 + float(b)) * 0.1
    return meters * 3.28084


def _elevation_from_img_bilinear(img: Image.Image, px: float, py: float) -> float:
    """Bilinear sample in tile pixel space (reduces nearest-neighbor DEM chatter vs Google Earth)."""
    arr = np.asarray(img, dtype=np.float64)
    h, w = arr.shape[0], arr.shape[1]
    px = max(0.0, min(float(px), w - 1.001))
    py = max(0.0, min(float(py), h - 1.001))
    x0 = int(math.floor(px))
    y0 = int(math.floor(py))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = px - x0
    fy = py - y0

    def el(ix, iy):
        r, g, b = arr[iy, ix, 0], arr[iy, ix, 1], arr[iy, ix, 2]
        return _rgb_to_elevation_ft(r, g, b)

    e00 = el(x0, y0)
    e10 = el(x1, y0)
    e01 = el(x0, y1)
    e11 = el(x1, y1)
    e0 = e00 * (1.0 - fx) + e10 * fx
    e1 = e01 * (1.0 - fx) + e11 * fx
    return e0 * (1.0 - fy) + e1 * fy


def _elevation_from_tile(lat, lon):
    """Elevation (ft) from cached tile, bilinear. No network."""
    wx, wy = _world_pixel_xy(lat, lon, TILE_ZOOM)
    tx, ty = int(wx // TILE_SIZE), int(wy // TILE_SIZE)
    key = (TILE_ZOOM, tx, ty)
    if key not in tile_cache:
        return None
    img = tile_cache[key]
    lx = wx - tx * TILE_SIZE
    ly = wy - ty * TILE_SIZE
    return _elevation_from_img_bilinear(img, lx, ly)


def elevation_ft(lat, lon, token: str):
    wx, wy = _world_pixel_xy(lat, lon, TILE_ZOOM)
    tx = int(wx // TILE_SIZE)
    ty = int(wy // TILE_SIZE)
    img = get_tile(TILE_ZOOM, tx, ty, token)
    lx = wx - tx * TILE_SIZE
    ly = wy - ty * TILE_SIZE
    return _elevation_from_img_bilinear(img, lx, ly)


# ---------------- KML LOAD ----------------


def load_kml(upload):
    if upload.name.lower().endswith(".kmz"):
        z = zipfile.ZipFile(upload)
        kml_name = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
        if not kml_name:
            raise ValueError("KMZ does not contain a KML file")
        kml = z.read(kml_name)
    else:
        kml = upload.read()
    return ET.fromstring(kml)


def _get_style_colors(root, ns):
    """Build map style_id -> LineStyle color (aabbggrr hex)."""
    colors = {}
    for style in root.findall(".//k:Style", ns):
        sid = style.get("id")
        if not sid:
            continue
        line_style = style.find("k:LineStyle", ns)
        if line_style is not None:
            color_el = line_style.find("k:color", ns)
            if color_el is not None and color_el.text:
                colors[sid] = color_el.text.strip().lower()
    return colors


def _get_stylemap_normal_style_url(root, ns):
    """Build map stylemap_id -> styleUrl for normal state (so we resolve StyleMap -> Style)."""
    normal = {}
    for stylemap in root.findall(".//k:StyleMap", ns):
        smid = stylemap.get("id")
        if not smid:
            continue
        for pair in stylemap.findall("k:Pair", ns):
            key = pair.find("k:key", ns)
            if key is not None and key.text and key.text.strip().lower() == "normal":
                url_el = pair.find("k:styleUrl", ns)
                if url_el is not None and url_el.text:
                    normal[smid] = url_el.text.strip()
                break
    return normal


def _resolve_line_color(style_url, style_colors, stylemap_normal):
    """Resolve styleUrl to LineStyle color string (aabbggrr). Returns None if not found."""
    if not style_url or not style_url.strip():
        return None
    url = style_url.strip()
    if "#" in url:
        style_id = url.split("#")[-1].strip()
    else:
        style_id = url.split("/")[-1].strip() if "/" in url else url
    if not style_id:
        return None
    if style_id in style_colors:
        return style_colors[style_id]
    if style_id in stylemap_normal:
        return _resolve_line_color(stylemap_normal[style_id], style_colors, stylemap_normal)
    return None


def _is_red_line_color(color_str):
    """KML color is aabbggrr (hex). Red = high R, zero G, zero B."""
    if not color_str or len(color_str) < 8:
        return False
    color_str = color_str.lower().replace(" ", "")[:8]
    if len(color_str) != 8:
        return False
    r = color_str[6:8]
    g = color_str[4:6]
    b = color_str[2:4]
    return r == "ff" and g == "00" and b == "00"


def _include_agm(name: str) -> bool:
    stripped = name.strip()
    if stripped.startswith(SP_PREFIX):
        return False
    lower = stripped.lower()
    if any(k in lower for k in NEGATIVE_KEYWORDS):
        return False
    return True


def _is_anchor_agm(name: str) -> bool:
    lower = name.strip().lower()
    if lower == "000":
        return True
    if "launch valve" in lower or "launcher valve" in lower:
        return True
    if lower == "launcher":
        return True
    return False


def _coords_from_linestring(ls, ns):
    coord_node = ls.find("k:coordinates", ns)
    if coord_node is None or not coord_node.text:
        return None
    pts = []
    for c in coord_node.text.split():
        lon, lat, *_rest = map(float, c.split(","))
        pts.append((lat, lon))
    return pts if len(pts) >= 2 else None


def _placemark_style_url(pm, folder, ns):
    el = pm.find("k:styleUrl", ns)
    if el is not None and el.text and el.text.strip():
        return el.text.strip()
    if folder is not None:
        el = folder.find("k:styleUrl", ns)
        if el is not None and el.text and el.text.strip():
            return el.text.strip()
    return None


def _dedupe_polylines(segments: list[list[tuple[float, float]]], tol_ft: float = 2.0):
    """Drop duplicate polylines (same vertex count and vertices within tol)."""
    uniq = []
    for seg in segments:
        dup = False
        for u in uniq:
            if len(u) != len(seg):
                continue
            if all(haversine_ft(a[0], a[1], b[0], b[1]) < tol_ft for a, b in zip(u, seg)):
                dup = True
                break
        if not dup:
            uniq.append(seg)
    return uniq


def _append_fwd(chain: list, pts: list) -> None:
    """chain[-1] meets pts[0]; extend with pts[1:] (drop duplicate joint)."""
    if haversine_ft(chain[-1][0], chain[-1][1], pts[0][0], pts[0][1]) < 1.0:
        chain.extend(pts[1:])
    else:
        chain.extend(pts)


def _append_rev(chain: list, pts: list) -> None:
    """chain[-1] meets pts[-1]; extend backward along pts."""
    rp = list(reversed(pts))
    _append_fwd(chain, rp)


def _prepend_fwd(chain: list, pts: list) -> None:
    """pts[-1] meets chain[0]; prepend pts[0]..pts[-2] in order."""
    if haversine_ft(chain[0][0], chain[0][1], pts[-1][0], pts[-1][1]) < 1.0:
        for k in range(len(pts) - 2, -1, -1):
            chain.insert(0, pts[k])
    else:
        for k in range(len(pts) - 1, -1, -1):
            chain.insert(0, pts[k])


def _prepend_rev(chain: list, pts: list) -> None:
    """pts[0] meets chain[0]; prepend pts[1], pts[2], ... before chain."""
    if haversine_ft(chain[0][0], chain[0][1], pts[0][0], pts[0][1]) < 1.0:
        for k in range(1, len(pts)):
            chain.insert(k - 1, pts[k])
    else:
        for k in range(len(pts) - 1, -1, -1):
            chain.insert(0, pts[k])


def merge_centerline_segments(
    segments: list[list[tuple[float, float]]],
    anchor_lat: float,
    anchor_lon: float,
    tol_ft: float = MERGE_ENDPOINT_TOL_FT,
) -> list[tuple[float, float]]:
    """
    Order and stitch multiple LineStrings into one continuous centerline.
    Extends from both ends of the growing chain so stubs and mainline order correctly.
    """
    segs = [list(s) for s in segments if s and len(s) >= 2]
    segs = _dedupe_polylines(segs)
    if not segs:
        return []
    if len(segs) == 1:
        seg = segs[0]
        d0 = haversine_ft(anchor_lat, anchor_lon, seg[0][0], seg[0][1])
        d1 = haversine_ft(anchor_lat, anchor_lon, seg[-1][0], seg[-1][1])
        return list(seg) if d0 <= d1 else list(reversed(seg))

    def dist_to_line(pts):
        proj = project_to_line(anchor_lat, anchor_lon, pts)
        pt = point_on_line(pts, proj)
        return haversine_ft(anchor_lat, anchor_lon, pt[0], pt[1])

    def length_ft(pts):
        return sum(
            haversine_ft(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]) for i in range(len(pts) - 1)
        )

    near = [pts for pts in segs if dist_to_line(pts) <= max(100.0, tol_ft * 3)]
    start_seg = max(near, key=length_ft) if near else min(segs, key=dist_to_line)
    d0 = haversine_ft(anchor_lat, anchor_lon, start_seg[0][0], start_seg[0][1])
    d1 = haversine_ft(anchor_lat, anchor_lon, start_seg[-1][0], start_seg[-1][1])
    chain = list(start_seg) if d0 <= d1 else list(reversed(start_seg))
    remaining = [p for p in segs if p is not start_seg]

    def best_connection(chain_pts, cand, limit: float | None):
        best_d = float("inf")
        best_how = None
        opts = [
            (haversine_ft(chain_pts[-1][0], chain_pts[-1][1], cand[0][0], cand[0][1]), "append_fwd"),
            (haversine_ft(chain_pts[-1][0], chain_pts[-1][1], cand[-1][0], cand[-1][1]), "append_rev"),
            (haversine_ft(chain_pts[0][0], chain_pts[0][1], cand[-1][0], cand[-1][1]), "prepend_fwd"),
            (haversine_ft(chain_pts[0][0], chain_pts[0][1], cand[0][0], cand[0][1]), "prepend_rev"),
        ]
        for d, how in opts:
            if limit is not None and d > limit:
                continue
            if d < best_d:
                best_d, best_how = d, how
        return best_d, best_how

    while remaining:
        best_i = None
        best_d = float("inf")
        best_how = None
        for i, pts in enumerate(remaining):
            d, how = best_connection(chain, pts, tol_ft)
            if how is not None and d < best_d:
                best_d, best_i, best_how = d, i, how
        if best_i is None:
            break
        pts = remaining.pop(best_i)
        if best_how == "append_fwd":
            _append_fwd(chain, pts)
        elif best_how == "append_rev":
            _append_rev(chain, pts)
        elif best_how == "prepend_fwd":
            _prepend_fwd(chain, pts)
        else:
            _prepend_rev(chain, pts)

    if remaining:
        while remaining:
            best_i = None
            best_d = float("inf")
            best_how = None
            for i, pts in enumerate(remaining):
                d, how = best_connection(chain, pts, None)
                if how is not None and d < best_d:
                    best_d, best_i, best_how = d, i, how
            if best_i is None:
                break
            pts = remaining.pop(best_i)
            if best_how == "append_fwd":
                _append_fwd(chain, pts)
            elif best_how == "append_rev":
                _append_rev(chain, pts)
            elif best_how == "prepend_fwd":
                _prepend_fwd(chain, pts)
            else:
                _prepend_rev(chain, pts)

    return chain


def parse(root):
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    agms = []
    centerline_strings = []

    style_colors = _get_style_colors(root, ns)
    stylemap_normal = _get_stylemap_normal_style_url(root, ns)

    def try_add_red_line(pm, folder):
        ls = pm.find("k:LineString", ns)
        if ls is None:
            return
        style_url = _placemark_style_url(pm, folder, ns)
        line_color = _resolve_line_color(style_url, style_colors, stylemap_normal) if style_url else None
        if line_color is None or not _is_red_line_color(line_color):
            return
        pts = _coords_from_linestring(ls, ns)
        if pts:
            centerline_strings.append(pts)

    for folder in root.findall(".//k:Folder", ns):
        fname_el = folder.find("k:name", ns)
        fname = fname_el.text.lower() if fname_el is not None and fname_el.text else ""

        for pm in folder.findall(".//k:Placemark", ns):
            try_add_red_line(pm, folder)

        if "agm" in fname:
            for p in folder.findall(".//k:Placemark", ns):
                pname = p.find("k:name", ns)
                coords = p.find(".//k:coordinates", ns)
                if pname is None or coords is None or not pname.text or not coords.text:
                    continue
                if p.find("k:LineString", ns) is not None:
                    continue
                agm_name = pname.text.strip()
                if not _include_agm(agm_name):
                    continue
                lon, lat, *_rest = map(float, coords.text.split(","))
                agms.append((agm_name, lat, lon))

    # Document-level Placemarks (not always inside a Folder)
    doc = root.find("k:Document", ns)
    if doc is not None:
        for pm in doc.findall("k:Placemark", ns):
            try_add_red_line(pm, None)

    # Fallback: CENTERLINE folder — any LineString (color may not be red in some exports)
    if not centerline_strings:
        for folder in root.findall(".//k:Folder", ns):
            name = folder.find("k:name", ns)
            if name is None or not name.text or "center" not in name.text.lower():
                continue
            for ls in folder.findall(".//k:LineString", ns):
                pts = _coords_from_linestring(ls, ns)
                if pts:
                    centerline_strings.append(pts)

    center: list[tuple[float, float]] = []
    if centerline_strings and agms:
        anchor = _pick_anchor_agm(agms)
        if anchor:
            _name, alat, alon = anchor
            center = merge_centerline_segments(centerline_strings, alat, alon)
        else:
            merged = merge_centerline_segments(centerline_strings, centerline_strings[0][0][0], centerline_strings[0][0][1])
            center = merged if merged else list(centerline_strings[0])
    elif centerline_strings:
        merged = merge_centerline_segments(centerline_strings, centerline_strings[0][0][0], centerline_strings[0][0][1])
        center = merged if merged else list(centerline_strings[0])

    return agms, center


def _pick_anchor_agm(agms):
    ranked = []
    for n, lat, lon in agms:
        lower = n.strip().lower()
        if "launch valve" in lower or "launcher valve" in lower:
            priority = 0
        elif lower == "000":
            priority = 1
        elif lower == "launcher":
            priority = 2
        else:
            continue
        ranked.append((priority, n, lat, lon))

    if not ranked:
        return None

    ranked.sort(key=lambda x: x[0])
    _prio, n, lat, lon = ranked[0]
    return n, lat, lon


def _orient_centerline(center, agms):
    anchor = _pick_anchor_agm(agms)
    if not anchor or len(center) < 2:
        return center

    _name, alat, alon = anchor
    d_start = haversine_ft(alat, alon, *center[0])
    d_end = haversine_ft(alat, alon, *center[-1])
    return list(reversed(center)) if d_end < d_start else center


# ---------------- STREAMLIT ----------------


def _prepare_centerline(agms, center):
    """Orient, dedupe, trim at launcher — shared by Streamlit and CLI validation."""
    center = _orient_centerline(center, agms)
    center = dedupe_centerline(center, min_sep_ft=1.0)
    anchor = _pick_anchor_agm(agms)
    if anchor and len(center) >= 2:
        _name, alat, alon = anchor
        proj = project_to_line(alat, alon, center)
        idx, t = int(proj[0]), float(proj[1])
        if idx > 0 or t > 0.001:
            pt = point_on_line(center, proj)
            center = [pt] + center[idx + 1 :]
    return center


def _terrain_sample_and_elevations(center: list, token: str) -> list[float]:
    """Sample Mapbox terrain-RGB along centerline; return elevation (ft) at each vertex (same len as center)."""
    cum2d = [0.0]
    for i in range(len(center) - 1):
        cum2d.append(cum2d[-1] + haversine_ft(*center[i], *center[i + 1]))
    total2d = cum2d[-1]
    sample_dists = []
    sample_pts = []
    d = 0.0
    while d <= total2d:
        sample_dists.append(d)
        for i in range(len(center) - 1):
            seg_len = cum2d[i + 1] - cum2d[i]
            if seg_len <= 0:
                continue
            if d <= cum2d[i + 1]:
                t = (d - cum2d[i]) / seg_len if seg_len > 0 else 0.0
                t = max(0.0, min(1.0, t))
                lat = center[i][0] + t * (center[i + 1][0] - center[i][0])
                lon = center[i][1] + t * (center[i + 1][1] - center[i][1])
                sample_pts.append((lat, lon))
                break
        else:
            sample_pts.append(center[-1])
        d += ELEVATION_SAMPLE_INTERVAL_FT
    if not sample_pts:
        sample_pts = [center[0], center[-1]]
        sample_dists = [0.0, total2d]

    tile_keys = _tile_keys_for_line(sample_pts)
    need = [(*k, token) for k in tile_keys if k not in tile_cache]
    if need:
        with ThreadPoolExecutor(max_workers=TERRAIN_MAX_WORKERS) as ex:
            list(ex.map(_fetch_tile, need))
    sample_elevs = [_elevation_from_tile(lat, lon) for lat, lon in sample_pts]
    if None in sample_elevs:
        sample_elevs = [elevation_ft(lat, lon, token) for lat, lon in sample_pts]

    def interp_elev(cum_d):
        if cum_d <= sample_dists[0]:
            return sample_elevs[0]
        if cum_d >= sample_dists[-1]:
            return sample_elevs[-1]
        for j in range(len(sample_dists) - 1):
            if sample_dists[j] <= cum_d <= sample_dists[j + 1]:
                span = sample_dists[j + 1] - sample_dists[j]
                t = (cum_d - sample_dists[j]) / span if span > 0 else 0.0
                return sample_elevs[j] + t * (sample_elevs[j + 1] - sample_elevs[j])
        return sample_elevs[-1]

    raw_elevations = [interp_elev(cum2d[i]) for i in range(len(center))]
    if ELEVATION_SMOOTH_WINDOW >= 3:
        raw_elevations = _smooth_elevations_1d(raw_elevations, ELEVATION_SMOOTH_WINDOW)
    return raw_elevations


def _densify_with_elevations(center: list, raw_elevations: list) -> tuple[list, list]:
    res = densify_centerline(center, MAX_CENTERLINE_SEGMENT_FT, raw_elevations)
    if isinstance(res, tuple):
        return res[0], res[1]
    return res, None


def _build_distance_table_terrain(agms, center_densified: list, elevations: list):
    _seg_len_3d, cum3 = compute_stationing_3d(center_densified, elevations)
    cum2 = compute_stationing_2d_cum(center_densified)

    station_rows = []
    for i, (name, lat, lon) in enumerate(agms):
        proj = project_to_line(lat, lon, center_densified)
        stn = station_at_3d(proj, center_densified, elevations, cum3)
        station_rows.append((stn, name, i, lat, lon))
    station_rows.sort(key=lambda r: (r[0], r[2]))
    ordered_agms = [(name, lat, lon) for _st, name, _i, lat, lon in station_rows]

    projected = []
    for name, lat, lon in ordered_agms:
        proj = project_to_line(lat, lon, center_densified)
        st3 = station_at_3d(proj, center_densified, elevations, cum3)
        st2 = station_at_horizontal(proj, center_densified, cum2)
        projected.append((name, proj, st3, st2))

    agm_pos = {name: (lat, lon) for name, lat, lon in ordered_agms}
    rows = []
    cumulative = 0.0
    cum_h = 0.0
    for i in range(len(projected) - 1):
        stn3_a = projected[i][2]
        stn3_b = projected[i + 1][2]
        stn2_a = projected[i][3]
        stn2_b = projected[i + 1][3]
        d = abs(stn3_b - stn3_a)
        dh = abs(stn2_b - stn2_a)
        if d < 1.0 and projected[i][0] in agm_pos and projected[i + 1][0] in agm_pos:
            a, b = projected[i][0], projected[i + 1][0]
            d = max(d, haversine_ft(agm_pos[a][0], agm_pos[a][1], agm_pos[b][0], agm_pos[b][1]))
        cumulative += d
        cum_h += dh
        rows.append(
            {
                "From": projected[i][0],
                "To": projected[i + 1][0],
                "Segment Feet": round(d, 2),
                "Cumulative Feet": round(cumulative, 2),
                "Segment Miles": round(d / 5280, 4),
                "Cumulative Miles": round(cumulative / 5280, 4),
                "Horiz Seg Miles": round(dh / 5280, 4),
                "Horiz Cum Miles": round(cum_h / 5280, 4),
            }
        )
    return pd.DataFrame(rows), cum3[-1]


def validate_kmz_on_disk(path: str, token: str) -> None:
    """CLI: print every segment (3D terrain path). Set AGM_VALIDATE_KMZ and MAPBOX_TOKEN."""
    with open(path, "rb") as f:
        raw = f.read()

    class _U:
        name = "file.kmz" if path.lower().endswith(".kmz") else "file.kml"

        def read(self):
            return raw

    root = load_kml(_U())
    agms, center = parse(root)
    if not agms or not center:
        print("Missing AGMs or Centerline")
        return
    center = _prepare_centerline(agms, center)
    if len(center) < 2:
        print("Centerline too short after trim")
        return
    if not token:
        print("Set MAPBOX_TOKEN in the environment for terrain validation.")
        return
    raw_elev = _terrain_sample_and_elevations(center, token)
    center_d, elevs = _densify_with_elevations(center, raw_elev)
    if elevs is None:
        print("Densify failed to return elevations")
        return
    df, total_ft = _build_distance_table_terrain(agms, center_d, elevs)
    print(f"File: {path}")
    print(f"Total 3D terrain path length: {total_ft:.2f} ft ({total_ft / 5280:.4f} mi)")
    print()
    for _, row in df.iterrows():
        print(
            f"{row['From']:>5} -> {row['To']:<5}  {row['Segment Miles']:>8.4f} mi  ({row['Segment Feet']:>10.2f} ft)"
        )


def main():
    st.set_page_config(page_title="AGM Terrain Distances", layout="wide")
    st.title("Terrain-Aware AGM Distance Checker")
    st.caption(
        "3D segment miles use **arc-length** along each segment (Haversine A→P + elevation linear in that "
        "fraction), not planar projection × segment length. **Horiz** columns are ground-only for comparison."
    )

    token = _get_mapbox_token()
    if not token:
        token = st.text_input("Mapbox token (required for terrain-RGB elevation)", type="password", key="token").strip() or None

    upload = st.file_uploader("Upload KMZ/KML", ["kmz", "kml"], key="upload")

    if not upload:
        st.info("Upload a KMZ or KML to compute 3D terrain-aware distances between AGMs along the centerline.")
        return

    try:
        root = load_kml(upload)
        agms, center = parse(root)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return

    if not agms or not center:
        st.error("Missing AGMs or Centerline")
        return

    center = _prepare_centerline(agms, center)
    if len(center) < 2:
        st.error("Centerline too short after trim")
        return

    if not token:
        st.error("Mapbox token missing. Add it to Streamlit secrets or paste it above.")
        return

    with st.spinner("Fetching terrain tiles and computing 3D distances..."):
        raw_elev = _terrain_sample_and_elevations(center, token)
        center_d, elevs = _densify_with_elevations(center, raw_elev)
        if elevs is None:
            st.error("Internal error: elevations missing after densify")
            return
        df, total_ft = _build_distance_table_terrain(agms, center_d, elevs)

    st.metric("Total 3D path length along centerline (mi)", f"{total_ft / 5280:.4f}")
    st.dataframe(df, width="stretch")

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode(),
        "AGM_Terrain_Distances.csv",
        "text/csv",
    )


if __name__ == "__main__":
    v = os.environ.get("AGM_VALIDATE_KMZ")
    if v:
        try:
            validate_kmz_on_disk(v, os.environ.get("MAPBOX_TOKEN", "").strip())
        except Exception as e:
            print(f"Validation error: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
        sys.exit(0)
    try:
        main()
    except Exception as e:
        st.error(f"App error: {e}")
        import traceback

        st.code(traceback.format_exc(), language="text")
