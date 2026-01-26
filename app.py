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
DENSIFY_FEET = 30        # ~10 m
EARTH_R_FT = 20925524.9
TILE_ZOOM = 14
TILE_SIZE = 256

def get_mapbox_token():
    # Your secrets format:
    # [mapbox]
    # token = "pk...."
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
# CENTERLINE STITCHING (fixes multi-LineString teleport jumps)
# ======================
def stitch_centerline_parts(parts, gap_warn_ft=200.0):
    """
    Given multiple line parts (each a list[(lat,lon)]), build one continuous polyline
    by attaching parts to the current end that minimizes endpoint gap, reversing parts if needed.
    """
    parts = [p for p in parts if p and len(p) >= 2]
    if not parts:
        return [], []

    used = [False] * len(parts)
    # start with the longest part (more stable)
    start_idx = int(np.argmax([len(p) for p in parts]))
    poly = parts[start_idx][:]
    used[start_idx] = True

    gaps = []

    def dist(a, b):
        return haversine_ft(a[0], a[1], b[0], b[1])

    while not all(used):
        best = None  # (gap, idx, mode, reversed?)
        end_pt = poly[-1]

        for i, p in enumerate(parts):
            if used[i]:
                continue
            p_start, p_end = p[0], p[-1]
            # attach p to end of poly:
            gap_end_to_start = dist(end_pt, p_start)
            gap_end_to_end   = dist(end_pt, p_end)

            # choose smaller gap; reverse part if needed
            if best is None or gap_end_to_start < best[0]:
                best = (gap_end_to_start, i, "append", False)  # p as-is
            if gap_end_to_end < best[0]:
                best = (gap_end_to_end, i, "append", True)     # p reversed

        gap, i, mode, rev = best
        p = parts[i][:]
        if rev:
            p.reverse()

        # append, dropping duplicate join point if essentially same
        if dist(poly[-1], p[0]) < 5.0:
            poly.extend(p[1:])
        else:
            poly.extend(p)

        used[i] = True
        gaps.append(gap)

    # warn if any big seam remains
    return poly, gaps

# ======================
# MAPBOX TERRAIN (tile cached + correct pixel-in-tile)
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
    headers = {
        "User-Agent": "terrain-aware-agm-distance-checker/1.0",
        "Accept": "image/png,image/*;q=0.9,*/*;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=25)

    ct = (r.headers.get("Content-Type") or "").lower()
    if r.status_code != 200 or ("image" not in ct):
        snippet = (r.text or "")[:400]
        raise RuntimeError(
            f"Mapbox tile error status={r.status_code}, content-type={ct}, response={snippet!r}"
        )

    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    tile_cache[key] = img
    return img

def elevation_ft(lat, lon):
    x_float, y_float = latlon_to_world_float(lat, lon, TILE_ZOOM)
    x_tile = int(math.floor(x_float))
    y_tile = int(math.floor(y_float))

    x_frac = x_float - math.floor(x_float)
    y_frac = y_float - math.floor(y_float)
    px = min(TILE_SIZE - 1, max(0, int(x_frac * TILE_SIZE)))
    py = min(TILE_SIZE - 1, max(0, int(y_frac * TILE_SIZE)))

    img = get_tile(TILE_ZOOM, x_tile, y_tile)
    R, G, B = img.getpixel((px, py))

    meters = -10000.0 + (R * 256.0 * 256.0 + G * 256.0 + B) * 0.1
    return meters * 3.28084  # feet

# ======================
# KML / KMZ PARSE
# ======================
def load_kml(upload):
    if upload.name.lower().endswith(".kmz"):
        z = zipfile.ZipFile(upload)
        kml_name = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
        if not kml_name:
            raise ValueError("KMZ contains no .kml file")
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
        if name_el is None or not name_el.text:
            continue
        fname = name_el.text.strip().upper()

        if fname == "AGMS":
            for p in f.findall(".//k:Placemark", ns):
                nm_el = p.find("k:name", ns)
                coord_el = p.find(".//k:Point/k:coordinates", ns) or p.find(".//k:coordinates", ns)
                if nm_el is None or coord_el is None or not coord_el.text:
                    continue
                n = nm_el.text.strip()
                lon, lat, *_ = map(float, coord_el.text.strip().split(","))
                agms.append((n, lat, lon))

        if fname == "CENTERLINE":
            for ls in f.findall(".//k:LineString", ns):
                coord_el = ls.find("k:coordinates", ns)
                if coord_el is None or not coord_el.text:
                    continue
                part = []
                for c in coord_el.text.split():
                    lon, lat, *_ = map(float, c.split(","))
                    part.append((lat, lon))
                if len(part) >= 2:
                    center_parts.append(part)

    return agms, center_parts

# ======================
# STREAMLIT APP
# ======================
st.title("Terrain-Aware AGM Distance Checker")

if not MAPBOX_TOKEN:
    st.error('Mapbox token not found in secrets. Expecting:\n\n[mapbox]\n token = "pk...."\n')
    st.stop()

upload = st.file_uploader("Drag & drop KML or KMZ", ["kml", "kmz"])

if upload:
    try:
        root = load_kml(upload)
        agms, center_parts = parse(root)
    except Exception as e:
        st.error(f"Failed to read/parse file: {e}")
        st.stop()

    # stitch multiple line parts (THIS fixes the 060->070 seam issue)
    center, seam_gaps = stitch_centerline_parts(center_parts)

    st.write(f"{len(agms)} AGMs | {sum(len(p) for p in center_parts)} raw centerline pts | {len(center_parts)} line part(s)")

    if seam_gaps:
        worst = max(seam_gaps)
        if worst > 200:
            st.warning(f"Centerline has a large seam gap (~{worst:,.1f} ft). That can break a span. (We stitched the best we could.)")

    if not agms or not center:
        st.error("Need both AGMS and CENTERLINE folders (and CENTERLINE must contain LineString coordinates).")
        st.stop()

    # Start at 000 by AGM name
    try:
        agms.sort(key=lambda x: int(x[0]))
    except Exception:
        st.error("AGM names must be numeric (e.g., 000, 005, 010...)")
        st.stop()

    # densify stitched centerline
    center = densify(center)

    with st.spinner("Sampling terrain elevations (cached tiles)..."):
        try:
            elevations = np.array([elevation_ft(lat, lon) for lat, lon in center], dtype=float)
        except Exception as e:
            st.error(f"Elevation sampling failed: {e}")
            st.stop()

    # snap AGMs to densified centerline vertices
    snapped = []
    for name, lat, lon in agms:
        dists = [haversine_ft(lat, lon, p[0], p[1]) for p in center]
        snapped.append((name, int(np.argmin(dists))))

    # reverse traversal if centerline digitized opposite stationing
    if snapped[0][1] > snapped[-1][1]:
        center.reverse()
        elevations = elevations[::-1]
        snapped = [(n, len(center) - 1 - i) for n, i in snapped]

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

        dist_mi = dist_ft / 5280.0
        cumulative_mi += dist_mi

        rows.append(
            {
                "From AGM": a_name,
                "To AGM": b_name,
                "Distance (ft)": round(dist_ft, 2),
                "Distance (mi)": round(dist_mi, 6),
                "Cumulative (mi)": round(cumulative_mi, 6),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="terrain_agm_distances.csv",
        mime="text/csv",
    )
