import io
import math
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image


EARTH_R_FT = 20925524.9
TILE_ZOOM = 14
TILE_SIZE = 256

# Sub-meter densification: 1 meter ≈ 3.28 ft. No segment longer than this so path follows bends closely.
MAX_CENTERLINE_SEGMENT_FT = 3.28


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


def densify_centerline(line, max_seg_ft):
    """Insert points so no segment exceeds max_seg_ft; path follows bends."""
    if not line or len(line) < 2 or max_seg_ft <= 0:
        return line
    out = [line[0]]
    for i in range(len(line) - 1):
        a_lat, a_lon = line[i]
        b_lat, b_lon = line[i + 1]
        h = haversine_ft(a_lat, a_lon, b_lat, b_lon)
        if h <= max_seg_ft:
            out.append((b_lat, b_lon))
            continue
        n = max(2, int(math.ceil(h / max_seg_ft)))
        for k in range(1, n):
            t = k / n
            lat = a_lat + t * (b_lat - a_lat)
            lon = a_lon + t * (b_lon - a_lon)
            out.append((lat, lon))
        out.append((b_lat, b_lon))
    return out


# -------- PROJECT POINT ONTO LINE SEGMENTS --------


def project_to_line(lat, lon, line):
    """Snap (lat, lon) to the closest point on the centerline. Returns (segment_index, t)."""
    best_dist = float("inf")
    best_index = 0
    best_t = 0.0

    for i in range(len(line) - 1):
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

        proj_xy = A + t * AB
        proj_lat, proj_lon = _from_local_xy_ft(proj_xy, ref_lat)
        d = haversine_ft(lat, lon, proj_lat, proj_lon)

        if d < best_dist:
            best_dist = d
            best_index = i
            best_t = t

    return best_index, best_t


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

        if "center" in fname:
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
            # Pick the LineString whose FIRST vertex is closest to the launcher.
            def dist_to_start(pts):
                return haversine_ft(alat, alon, pts[0][0], pts[0][1])
            start_line = min(centerline_strings, key=dist_to_start)
            center = list(start_line)
            remaining = [pts for pts in centerline_strings if pts is not start_line]
            tol_ft = 100.0
            while remaining:
                end_pt = center[-1]
                best = None
                best_dist = tol_ft
                append_forward = True
                for pts in remaining:
                    d_start = haversine_ft(end_pt[0], end_pt[1], pts[0][0], pts[0][1])
                    d_end = haversine_ft(end_pt[0], end_pt[1], pts[-1][0], pts[-1][1])
                    if d_start < best_dist:
                        best_dist = d_start
                        best = pts
                        append_forward = True
                    if d_end < best_dist:
                        best_dist = d_end
                        best = pts
                        append_forward = False
                if best is None:
                    break
                if append_forward:
                    center.extend(best[1:])
                else:
                    center.extend(list(reversed(best))[1:])
                remaining = [p for p in remaining if p is not best]
        else:
            center = max(centerline_strings, key=lambda pts: sum(haversine_ft(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]) for i in range(len(pts) - 1)))
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


st.title("Terrain-Aware AGM Distance Checker")

token = _get_mapbox_token()
if not token:
    token = st.text_input("Mapbox token (required for terrain)", type="password").strip() or None

upload = st.file_uploader("Upload KMZ/KML", ["kmz", "kml"])

if upload:
    try:
        root = load_kml(upload)
        agms, center = parse(root)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    if not agms or not center:
        st.error("Missing AGMs or Centerline")
        st.stop()

    center = _orient_centerline(center, agms)
    center = densify_centerline(center, MAX_CENTERLINE_SEGMENT_FT)

    if not token:
        st.error("Mapbox token missing. Add it to Streamlit secrets or paste it above.")
        st.stop()

    with st.spinner("Sampling terrain elevations along centerline..."):
        elevations = [elevation_ft(lat, lon, token) for lat, lon in center]

    seg_len_3d, cum = compute_stationing(center, elevations)
    total_length = cum[-1]

    projected = []
    for name, lat, lon in agms:
        proj = project_to_line(lat, lon, center)
        stn = station_at(proj, seg_len_3d, cum)
        projected.append((name, proj, stn))

    all_numeric = all(_is_numeric_label(name) for name, _, _ in agms)
    if all_numeric:
        projected.sort(key=lambda x: int(x[0].strip()))
    else:
        projected.sort(key=lambda x: x[2])

    rows = []
    cumulative = 0.0
    for i in range(len(projected) - 1):
        # Linear pipeline: distance = station difference (direction of travel 000 -> 010 -> ...).
        # Do NOT use "shorter direction" — that assumes a loop and breaks.
        stn_a = projected[i][2]
        stn_b = projected[i + 1][2]
        d = abs(stn_b - stn_a)
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
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode(),
        "AGM_Terrain_Distances.csv",
        "text/csv",
    )


def _self_test():
    """Verify: snap two points to centerline, distance = path length along line (3D)."""
    line = [(30.0, -90.0), (30.001, -90.0), (30.002, -90.0)]
    elevations = [0.0, 0.0, 0.0]
    seg_len_3d, cum = compute_stationing(line, elevations)
    total = cum[-1]
    proj1 = project_to_line(30.0, -90.0, line)
    proj2 = project_to_line(30.002, -90.0, line)
    d = path_length_along_centerline(proj1, proj2, seg_len_3d, total)
    expected_ft = haversine_ft(30.0, -90.0, 30.002, -90.0)
    assert abs(d - expected_ft) < 1.0, f"expected ~{expected_ft}, got {d}"
    assert d > 0, "segment distance must be positive"
    print("Self-test passed: path length along centerline is correct.")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _self_test()
