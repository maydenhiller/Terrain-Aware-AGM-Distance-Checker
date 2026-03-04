import io
import math
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
# Sample elevation every this many ft (higher = fewer Mapbox tiles, faster)
ELEVATION_SAMPLE_INTERVAL_FT = 2000.0
TERRAIN_MAX_WORKERS = 10


ANCHOR_AGM_NAMES = {"000", "launcher", "launcher valve"}
SP_PREFIX = "SP"

POSITIVE_KEYWORDS = ["agm", "valve", "tap", "mlv"]
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
    """Snap (lat, lon) to closest point on centerline. If min_seg_index/min_t given, only consider
    points at or after that position (so AGMs project forward along the path)."""
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


# -------- STATIONING (TERRAIN-AWARE) --------


def compute_stationing(line, elevations):
    seg_len_3d = []
    for i in range(len(line) - 1):
        h = haversine_ft(*line[i], *line[i + 1])
        v = elevations[i + 1] - elevations[i]
        seg_len_3d.append(math.sqrt(h * h + v * v))

    cum = [0.0]
    for L in seg_len_3d:
        cum.append(cum[-1] + L)
    return seg_len_3d, cum


def station_at(proj, seg_len_3d, cum):
    idx, t = proj
    idx = int(idx)
    t = float(t)
    if idx < 0:
        idx = 0
        t = 0.0
    if idx >= len(seg_len_3d):
        idx = len(seg_len_3d) - 1
        t = 1.0
    return cum[idx] + seg_len_3d[idx] * t


def path_length_along_centerline(proj_a, proj_b, seg_len_3d, total_length):
    """
    Distance along the centerline (every bend, terrain) from snapped point A to snapped point B.
    Sums 3D segment lengths between the two projected points. Uses the shorter of the two directions.
    """
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


def _elevation_from_tile(lat, lon):
    """Read elevation at (lat, lon) from already-cached tile. No network."""
    wx, wy = _world_pixel_xy(lat, lon, TILE_ZOOM)
    tx, ty = int(wx // TILE_SIZE), int(wy // TILE_SIZE)
    key = (TILE_ZOOM, tx, ty)
    if key not in tile_cache:
        return None
    img = tile_cache[key]
    px = img.load()
    w, h = img.size
    fx = max(0, min(int(wx - tx * TILE_SIZE), w - 1))
    fy = max(0, min(int(wy - ty * TILE_SIZE), h - 1))
    R, G, B, _ = px[fx, fy]
    meters = -10000 + (R * 256 * 256 + G * 256 + B) * 0.1
    return meters * 3.28084


def elevation_ft(lat, lon, token: str):
    wx, wy = _world_pixel_xy(lat, lon, TILE_ZOOM)
    tx = int(wx // TILE_SIZE)
    ty = int(wy // TILE_SIZE)
    img = get_tile(TILE_ZOOM, tx, ty, token)

    px = img.load()
    w, h = img.size
    fx = int(wx - tx * TILE_SIZE)
    fy = int(wy - ty * TILE_SIZE)
    fx = max(0, min(w - 1, fx))
    fy = max(0, min(h - 1, fy))

    R, G, B, _A = px[fx, fy]
    meters = -10000 + (R * 256 * 256 + G * 256 + B) * 0.1
    return meters * 3.28084


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


def parse(root):
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    agms = []
    centerline_strings = []

    style_colors = _get_style_colors(root, ns)
    stylemap_normal = _get_stylemap_normal_style_url(root, ns)

    for folder in root.findall(".//k:Folder", ns):
        name = folder.find("k:name", ns)
        if name is None or not name.text:
            continue

        fname = name.text.lower()

        # Red LineString = actual pipeline (check every folder; CENTERLINE folder may hold access/blue)
        for pm in folder.findall(".//k:Placemark", ns):
            ls = pm.find("k:LineString", ns)
            if ls is None:
                continue
            style_url_el = pm.find("k:styleUrl", ns)
            if style_url_el is None or not style_url_el.text:
                style_url_el = folder.find("k:styleUrl", ns)
            style_url = style_url_el.text.strip() if style_url_el is not None and style_url_el.text else None
            line_color = _resolve_line_color(style_url, style_colors, stylemap_normal) if style_url else None
            if line_color is None or not _is_red_line_color(line_color):
                continue
            coord_node = ls.find("k:coordinates", ns)
            if coord_node is None or not coord_node.text:
                continue
            pts = []
            for c in coord_node.text.split():
                lon, lat, *_rest = map(float, c.split(","))
                pts.append((lat, lon))
            if len(pts) >= 2:
                centerline_strings.append(pts)

        if "agm" in fname:
            for p in folder.findall(".//k:Placemark", ns):
                pname = p.find("k:name", ns)
                coords = p.find(".//k:coordinates", ns)
                if pname is None or coords is None or not pname.text or not coords.text:
                    continue

                agm_name = pname.text.strip()
                if not _include_agm(agm_name):
                    continue

                lon, lat, *_rest = map(float, coords.text.split(","))
                agms.append((agm_name, lat, lon))

    # Fallback: if no red line found, use CENTERLINE folder (may be access/blue if mislabeled)
    if not centerline_strings:
        for folder in root.findall(".//k:Folder", ns):
            name = folder.find("k:name", ns)
            if name is None or not name.text or "center" not in name.text.lower():
                continue
            for ls in folder.findall(".//k:LineString", ns):
                coord_node = ls.find("k:coordinates", ns)
                if coord_node is None or not coord_node.text:
                    continue
                pts = []
                for c in coord_node.text.split():
                    lon, lat, *_rest = map(float, c.split(","))
                    pts.append((lat, lon))
                if len(pts) >= 2:
                    centerline_strings.append(pts)

    center = []
    if centerline_strings and agms:
        anchor = _pick_anchor_agm(agms)
        if anchor:
            _name, alat, alon = anchor
            # Pick the segment the launcher lies on. If several are close, use the longest (main line not stub).
            def dist_to_line(pts):
                proj = project_to_line(alat, alon, pts)
                pt = point_on_line(pts, proj)
                return haversine_ft(alat, alon, pt[0], pt[1])
            def length_ft(pts):
                return sum(haversine_ft(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1]) for i in range(len(pts)-1))
            near = [pts for pts in centerline_strings if dist_to_line(pts) <= 100.0]
            start_line = max(near, key=length_ft) if near else min(centerline_strings, key=dist_to_line)
            d0 = haversine_ft(alat, alon, start_line[0][0], start_line[0][1])
            d1 = haversine_ft(alat, alon, start_line[-1][0], start_line[-1][1])
            center = list(start_line) if d0 <= d1 else list(reversed(start_line))
            remaining = [p for p in centerline_strings if p is not start_line]
            tol_ft = 150.0  # connect segments that share a junction (stub to main line)
            anchor_ft = 200.0
            while remaining:
                end_pt = center[-1]
                best = None
                best_dist = tol_ft
                append_forward = True
                for pts in remaining:
                    d_start = haversine_ft(end_pt[0], end_pt[1], pts[0][0], pts[0][1])
                    d_end = haversine_ft(end_pt[0], end_pt[1], pts[-1][0], pts[-1][1])
                    if d_start < best_dist:
                        if haversine_ft(alat, alon, pts[-1][0], pts[-1][1]) > anchor_ft:
                            best_dist, best, append_forward = d_start, pts, True
                    if d_end < best_dist:
                        if haversine_ft(alat, alon, pts[0][0], pts[0][1]) > anchor_ft:
                            best_dist, best, append_forward = d_end, pts, False
                if best is None:
                    break
                if append_forward:
                    center.extend(best[1:])
                else:
                    center.extend(list(reversed(best))[1:])
                remaining = [p for p in remaining if p is not best]
        else:
            center = list(centerline_strings[0])
    elif centerline_strings:
        center = list(centerline_strings[0])

    return agms, center


def _is_numeric_label(name: str) -> bool:
    return name.strip().isdigit()


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

st.set_page_config(page_title="AGM Terrain Distances", layout="wide")

def main():
    st.title("Terrain-Aware AGM Distance Checker")

    token = _get_mapbox_token()
    if not token:
        token = st.text_input("Mapbox token (required for terrain)", type="password", key="token").strip() or None

    upload = st.file_uploader("Upload KMZ/KML", ["kmz", "kml"], key="upload")

    if not upload:
        st.info("Upload a KMZ or KML file to compute terrain-aware distances between AGMs along the centerline.")
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

    center = _orient_centerline(center, agms)
    center = dedupe_centerline(center, min_sep_ft=1.0)
    # Trim centerline to start at launcher (measure from there, no pre-000).
    anchor = _pick_anchor_agm(agms)
    if anchor and len(center) >= 2:
        _name, alat, alon = anchor
        proj = project_to_line(alat, alon, center)
        idx, t = int(proj[0]), float(proj[1])
        if idx > 0 or t > 0.001:
            pt = point_on_line(center, proj)
            center = [pt] + center[idx + 1:]
    if len(center) < 2:
        st.error("Centerline too short after trim")
        return

    if not token:
        st.error("Mapbox token missing. Add it to Streamlit secrets or paste it above.")
        return

    # Terrain: sample elevation every ELEVATION_SAMPLE_INTERVAL_FT, fetch tiles in parallel
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
    n_tiles = len(need)
    if need:
        with st.spinner(f"Fetching {n_tiles} terrain tiles ({TERRAIN_MAX_WORKERS} parallel)..."):
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

    with st.spinner("Building centerline and computing distances..."):
        center, elevations = densify_centerline(center, MAX_CENTERLINE_SEGMENT_FT, raw_elevations)
        seg_len_3d, cum = compute_stationing(center, elevations)
        total_length = cum[-1]

        all_numeric = all(_is_numeric_label(name) for name, _, _ in agms)
        if all_numeric:
            ordered_agms = sorted(agms, key=lambda x: int(x[0].strip()))
        else:
            pre = [(n, station_at(project_to_line(lat, lon, center), seg_len_3d, cum)) for n, lat, lon in agms]
            ordered_agms = sorted(agms, key=lambda a: next(s for n, s in pre if n == a[0]))

        projected = []
        min_idx, min_t = 0, 0.0
        for name, lat, lon in ordered_agms:
            proj = project_to_line(lat, lon, center, min_seg_index=min_idx, min_t=min_t)
            stn = station_at(proj, seg_len_3d, cum)
            projected.append((name, proj, stn))
            min_idx, min_t = int(proj[0]), float(proj[1])

        agm_pos = {name: (lat, lon) for name, lat, lon in ordered_agms}
        rows = []
        cumulative = 0.0
        for i in range(len(projected) - 1):
            stn_a = projected[i][2]
            stn_b = projected[i + 1][2]
            d = abs(stn_b - stn_a)
            if d < 1.0 and projected[i][0] in agm_pos and projected[i + 1][0] in agm_pos:
                a, b = projected[i][0], projected[i + 1][0]
                d = max(d, haversine_ft(agm_pos[a][0], agm_pos[a][1], agm_pos[b][0], agm_pos[b][1]))
            cumulative += d

            rows.append(
                {
                    "From": projected[i][0],
                    "To": projected[i + 1][0],
                    "Segment Feet": round(d, 2),
                    "Cumulative Feet": round(cumulative, 2),
                    "Segment Miles": round(d / 5280, 4),
                    "Cumulative Miles": round(cumulative / 5280, 4),
                }
            )

        df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode(),
        "AGM_Terrain_Distances.csv",
        "text/csv",
    )


# Streamlit: run main() only. No self-test on load.
try:
    main()
except Exception as e:
    st.error(f"App error: {e}")
    import traceback
    st.code(traceback.format_exc(), language="text")
