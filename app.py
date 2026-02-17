import streamlit as st
import zipfile, io, math
import xml.etree.ElementTree as ET
import numpy as np
import requests
from PIL import Image
import pandas as pd

MAPBOX_TOKEN = st.secrets["mapbox"]["token"]

EARTH_R_FT = 20925524.9
TILE_ZOOM = 14

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


# -------- PROJECT POINT ONTO LINE SEGMENTS --------

def project_to_line(lat, lon, line):

    best_dist = float("inf")
    best_index = 0
    best_t = 0

    for i in range(len(line) - 1):
        A = np.array(line[i])
        B = np.array(line[i + 1])
        P = np.array([lat, lon])

        AB = B - A
        t = np.dot(P - A, AB) / np.dot(AB, AB)
        t = max(0, min(1, t))

        proj = A + t * AB
        d = haversine_ft(lat, lon, proj[0], proj[1])

        if d < best_dist:
            best_dist = d
            best_index = i
            best_t = t

    return best_index, best_t


# -------- DISTANCE ALONG CENTERLINE --------

def distance_between(p1, p2, line, elevations):

    idx1, t1 = p1
    idx2, t2 = p2

    # Ensure p1 comes first along line
    if (idx1, t1) > (idx2, t2):
        idx1, t1, idx2, t2 = idx2, t2, idx1, t1

    # ---------- SAME SEGMENT CASE ----------
    if idx1 == idx2:
        A = line[idx1]
        B = line[idx1 + 1]

        seg_len = haversine_ft(*A, *B)

        horiz = seg_len * abs(t2 - t1)

        # approximate vertical difference
        v1 = elevations[idx1] + (elevations[idx1+1] - elevations[idx1]) * t1
        v2 = elevations[idx1] + (elevations[idx1+1] - elevations[idx1]) * t2

        vert = v2 - v1

        return math.sqrt(horiz*horiz + vert*vert)

    # ---------- DIFFERENT SEGMENTS ----------

    total = 0.0

    # partial first segment
    A = line[idx1]
    B = line[idx1 + 1]
    seg_len = haversine_ft(*A, *B)
    total += seg_len * (1 - t1)

    # full segments between
    for i in range(idx1 + 1, idx2):
        h = haversine_ft(*line[i], *line[i + 1])
        v = elevations[i + 1] - elevations[i]
        total += math.sqrt(h*h + v*v)

    # partial last segment
    A = line[idx2]
    B = line[idx2 + 1]
    seg_len = haversine_ft(*A, *B)
    total += seg_len * t2

    return total


# ---------------- MAPBOX TERRAIN ----------------

tile_cache = {}

def tile_xy(lat, lon, z):
    lat = math.radians(lat)
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(lat) + 1 / math.cos(lat)) / math.pi) / 2 * n)
    return x, y


def get_tile(z, x, y):
    key = (z, x, y)
    if key in tile_cache:
        return tile_cache[key]

    url = f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw?access_token={MAPBOX_TOKEN}"
    r = requests.get(url)
    img = Image.open(io.BytesIO(r.content))
    tile_cache[key] = img
    return img


def elevation_ft(lat, lon):
    x, y = tile_xy(lat, lon, TILE_ZOOM)
    img = get_tile(TILE_ZOOM, x, y)
    px = img.load()
    w, h = img.size

    fx = int((lon + 180) / 360 * w) % w
    fy = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * h) % h

    R, G, B = px[fx, fy][:3]
    meters = -10000 + (R * 256 * 256 + G * 256 + B) * 0.1
    return meters * 3.28084


# ---------------- KML LOAD ----------------

def load_kml(upload):
    if upload.name.lower().endswith(".kmz"):
        z = zipfile.ZipFile(upload)
        kml = z.read([n for n in z.namelist() if n.endswith(".kml")][0])
    else:
        kml = upload.read()
    return ET.fromstring(kml)


def parse(root):
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    agms = []
    center = []

    for folder in root.findall(".//k:Folder", ns):
        name = folder.find("k:name", ns)
        if name is None:
            continue

        fname = name.text.lower()

        if "agm" in fname:
            for p in folder.findall(".//k:Placemark", ns):
                pname = p.find("k:name", ns)
                coords = p.find(".//k:coordinates", ns)
                if pname is None or coords is None:
                    continue
                lon, lat, _ = map(float, coords.text.split(","))
                agms.append((pname.text.strip(), lat, lon))

        if "center" in fname:
            for ls in folder.findall(".//k:LineString", ns):
                for c in ls.find("k:coordinates", ns).text.split():
                    lon, lat, _ = map(float, c.split(","))
                    center.append((lat, lon))

    return agms, center


# ---------------- STREAMLIT ----------------

st.title("Terrain-Aware AGM Distance Checker")

upload = st.file_uploader("Upload KMZ/KML", ["kmz", "kml"])

if upload:

    root = load_kml(upload)
    agms, center = parse(root)

    if not agms or not center:
        st.error("Missing AGMs or Centerline")
        st.stop()

    elevations = [elevation_ft(lat, lon) for lat, lon in center]

    projected = []
    for name, lat, lon in agms:
        projected.append((name, project_to_line(lat, lon, center)))

    projected.sort(key=lambda x: x[1])

    rows = []
    cumulative = 0

    for i in range(len(projected) - 1):
        d = distance_between(
            projected[i][1],
            projected[i + 1][1],
            center,
            elevations,
        )

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
