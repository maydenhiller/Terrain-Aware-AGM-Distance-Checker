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


ANCHOR_AGM_NAMES = {"000", "launcher", "launcher valve"}  # case-insensitive (exact or substring match for launcher/valve)
SP_PREFIX = "SP"  # case-sensitive: ignore only "SP*", not "Sp*"

POSITIVE_KEYWORDS = ["agm", "valve", "tap", "mlv"]  # case-insensitive
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
    # Equirectangular-ish local mapping in feet (good enough for projection math)
    x = math.radians(lon) * EARTH_R_FT * math.cos(ref_lat_rad)
    y = math.radians(lat) * EARTH_R_FT
    return np.array([x, y], dtype=float)


def _from_local_xy_ft(xy, ref_lat_rad):
    x, y = float(xy[0]), float(xy[1])
    lat = math.degrees(y / EARTH_R_FT)
    denom = EARTH_R_FT * math.cos(ref_lat_rad)
    lon = math.degrees(x / denom) if denom != 0 else 0.0
    return lat, lon


# -------- PROJECT POINT ONTO LINE SEGMENTS --------


def project_to_line(lat, lon, line):
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


def _include_agm(name: str) -> bool:
    """
    Decide whether a placemark in the AGMs folder should be treated as an AGM to be measured.

    Updated rules:
    - Always skip names that start with exact 'SP' (case-sensitive), e.g. 'SP01', 'SP10'.
    - Skip names that contain obvious non-measurement terms like welds, risers, doors, blowoffs, etc.
    - Everything else in the AGMs folder is treated as a valid AGM, including names like 'S-001'.
    """
    stripped = name.strip()

    # 1. Hard skip SP* (exact, case-sensitive)
    if stripped.startswith(SP_PREFIX):
        return False

    lower = stripped.lower()

    # 2. Skip obvious non-measurement features (welds, doors, risers, etc.)
    if any(k in lower for k in NEGATIVE_KEYWORDS):
        return False

    # 3. Otherwise keep it as an AGM
    return True


def _is_anchor_agm(name: str) -> bool:
    """
    Detect the AGM that should define the starting end of the line.
    We treat anything containing 'launch valve' or 'launcher valve' as anchor,
    as well as names equal to 'launcher' or '000' (case-insensitive).
    """
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
    center = []

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
            for ls in folder.findall(".//k:LineString", ns):
                coord_node = ls.find("k:coordinates", ns)
                if coord_node is None or not coord_node.text:
                    continue
                for c in coord_node.text.split():
                    lon, lat, *_rest = map(float, c.split(","))
                    center.append((lat, lon))

    return agms, center


def _is_numeric_label(name: str) -> bool:
    return name.strip().isdigit()


def _pick_anchor_agm(agms):
    # Prefer "launch valve" (or variants), then "000", then "launcher"
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
    # Ensure stationing starts from the end that has the anchor AGM.
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

    if not token:
        st.error("Mapbox token missing. Add it to Streamlit secrets or paste it above.")
        st.stop()

    with st.spinner("Sampling terrain elevations along centerline..."):
        elevations = [elevation_ft(lat, lon, token) for lat, lon in center]

    seg_len_3d, cum = compute_stationing(center, elevations)

    projected = []
    all_numeric = True
    for name, lat, lon in agms:
        proj = project_to_line(lat, lon, center)
        stn = station_at(proj, seg_len_3d, cum)
        projected.append((name, proj, stn))
        if not _is_numeric_label(name):
            all_numeric = False

    # Ordering rule:
    # - If ALL AGM names are purely numeric (e.g., 000, 010, 015, ...),
    #   assume that numeric label order represents the intended pipeline order.
    #   In that case, sort by numeric value of the name.
    # - Otherwise, fall back to geometric stationing along the centerline
    #   starting from the launcher/000 end chosen in _orient_centerline.
    if all_numeric:
        projected.sort(key=lambda x: int(x[0].strip()))
    else:
        projected.sort(key=lambda x: x[2])

    rows = []
    cumulative = 0.0
    for i in range(len(projected) - 1):
        d = projected[i + 1][2] - projected[i][2]
        if d < 0:
            d = abs(d)
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
