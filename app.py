import streamlit as st
import zipfile, io, math, os
import xml.etree.ElementTree as ET
import numpy as np
import requests
from PIL import Image
import pandas as pd

# ======================
# CONFIG
# ======================
DENSIFY_FEET = 30
EARTH_R_FT = 20925524.9
TILE_ZOOM = 14
TILE_SIZE = 256

def get_mapbox_token():
    try:
        return str(st.secrets["mapbox"]["token"]).strip()
    except Exception:
        return os.getenv("MAPBOX_TOKEN", "").strip()

MAPBOX_TOKEN = get_mapbox_token()

# ======================
# GEO HELPERS
# ======================
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

def densify(line):
    if len(line) < 2:
        return line
    out = [line[0]]
    for i in range(len(line) - 1):
        a, b = line[i], line[i + 1]
        d = haversine_ft(a[0], a[1], b[0], b[1])
        n = max(1, int(d // DENSIFY_FEET))
        for j in range(1, n + 1):
            f = j / n
            out.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
    return out

# ======================
# CENTERLINE STITCHING
# ======================
def stitch_centerline_parts(parts):
    parts = [p for p in parts if p and len(p) >= 2]
    if not parts:
        return [], []

    used = [False] * len(parts)
    start_idx = int(np.argmax([len(p) for p in parts]))
    poly = parts[start_idx][:]
    used[start_idx] = True

    gaps = []

    def dist(a, b):
        return haversine_ft(a[0], a[1], b[0], b[1])

    while not all(used):
        best = None
        end_pt = poly[-1]

        for i, p in enumerate(parts):
            if used[i]:
                continue
            gap1 = dist(end_pt, p[0])
            gap2 = dist(end_pt, p[-1])

            if best is None or gap1 < best[0]:
                best = (gap1, i, False)
            if gap2 < best[0]:
                best = (gap2, i, True)

        gap, i, rev = best
        p = parts[i][:]
        if rev:
            p.reverse()

        if dist(poly[-1], p[0]) < 5:
            poly.extend(p[1:])
        else:
            poly.extend(p)

        used[i] = True
        gaps.append(gap)

    return poly, gaps

# ======================
# MAPBOX TERRAIN
# ======================
tile_cache = {}

def latlon_to_world_float(lat, lon, z):
    n = 2 ** z
    x_float = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y_float = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x_float, y_float

def get_tile(z, x, y):
    key = (z, x, y)
    if key in tile_cache:
        return tile_cache[key]

    url = f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw?access_token={MAPBOX_TOKEN}"
    r = requests.get(url, timeout=25)
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    tile_cache[key] = img
    return img

def elevation_ft(lat, lon):
    x_float, y_float = latlon_to_world_float(lat, lon, TILE_ZOOM)
    x_tile = int(math.floor(x_float))
    y_tile = int(math.floor(y_float))

    px = int((x_float - x_tile) * TILE_SIZE)
    py = int((y_float - y_tile) * TILE_SIZE)

    img = get_tile(TILE_ZOOM, x_tile, y_tile)
    R, G, B = img.getpixel((px, py))
    meters = -10000 + (R * 256 * 256 + G * 256 + B) * 0.1
    return meters * 3.28084

# ======================
# KML / KMZ PARSE
# ======================
def load_kml(upload):
    if upload.name.lower().endswith(".kmz"):
        z = zipfile.ZipFile(upload)
        kml_name = next(n for n in z.namelist() if n.lower().endswith(".kml"))
        kml = z.read(kml_name)
    else:
        kml = upload.read()
    return ET.fromstring(kml)

def parse(root):
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    agms = []
    center_parts = []

    for f in root.findall(".//k:Folder", ns):
        name_el = f.find("k:name", ns)
        if name_el is None:
            continue
        fname = name_el.text.strip().upper()

        if fname == "AGMS":
            for p in f.findall(".//k:Placemark", ns):
                nm = p.find("k:name", ns).text.strip()
                coord = p.find(".//k:coordinates", ns).text.strip()
                lon, lat, *_ = map(float, coord.split(","))
                agms.append((nm, lat, lon))

        if fname == "CENTERLINE":
            for ls in f.findall(".//k:LineString", ns):
                coords = ls.find("k:coordinates", ns).text.split()
                part = []
                for c in coords:
                    lon, lat, *_ = map(float, c.split(","))
                    part.append((lat, lon))
                center_parts.append(part)

    return agms, center_parts

# ======================
# STREAMLIT APP
# ======================
st.title("Terrain-Aware AGM Distance Checker")

upload = st.file_uploader("Drag & drop KML or KMZ", ["kml", "kmz"])

if upload:
    root = load_kml(upload)
    agms, center_parts = parse(root)

    center, _ = stitch_centerline_parts(center_parts)

    center = densify(center)
    elevations = np.array([elevation_ft(lat, lon) for lat, lon in center])

    # SNAP AGMs
    snapped = []
    for name, lat, lon in agms:
        dists = [haversine_ft(lat, lon, p[0], p[1]) for p in center]
        snapped.append((name, int(np.argmin(dists))))

    # ===== NEW: FORCE START POINT =====
    start_names = {"000", "launcher", "launcher valve", "launch valve"}

    # sort by centerline order first
    snapped.sort(key=lambda x: x[1])

    for i, (n, idx) in enumerate(snapped):
        if n.lower() in start_names:
            snapped = snapped[i:] + snapped[:i]
            break

    # ===== DISTANCES =====
    rows = []
    cumulative_mi = 0.0

    for i in range(len(snapped) - 1):
        a_name, a_idx = snapped[i]
        b_name, b_idx = snapped[i + 1]

        if b_idx < a_idx:
            a_idx, b_idx = b_idx, a_idx

        dist_ft = 0.0
        for j in range(a_idx, b_idx):
            h = haversine_ft(center[j][0], center[j][1], center[j + 1][0], center[j + 1][1])
            v = elevations[j + 1] - elevations[j]
            dist_ft += math.sqrt(h * h + v * v)

        dist_mi = dist_ft / 5280
        cumulative_mi += dist_mi

        rows.append({
            "From AGM": a_name,
            "To AGM": b_name,
            "Distance (ft)": round(dist_ft, 2),
            "Distance (mi)": round(dist_mi, 6),
            "Cumulative (mi)": round(cumulative_mi, 6),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False), "terrain_agm_distances.csv")
