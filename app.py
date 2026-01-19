import streamlit as st
import zipfile, io, math
import xml.etree.ElementTree as ET
import numpy as np
import requests
from PIL import Image
import pandas as pd

# ======================
# CONFIG
# ======================
MAPBOX_TOKEN = "PUT_YOUR_TOKEN_HERE"
DENSIFY_FEET = 30        # sampling resolution (~10 m)
EARTH_R_FT = 20925524.9  # Earth radius in feet
TILE_ZOOM = 14

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
    out = [line[0]]
    for i in range(len(line) - 1):
        a, b = line[i], line[i + 1]
        d = haversine_ft(*a, *b)
        n = max(1, int(d // DENSIFY_FEET))
        for j in range(1, n + 1):
            f = j / n
            out.append(
                (
                    a[0] + f * (b[0] - a[0]),
                    a[1] + f * (b[1] - a[1]),
                )
            )
    return out

# ======================
# MAPBOX TERRAIN
# ======================
tile_cache = {}

def tile_xy(lat, lon, z):
    lat = math.radians(lat)
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int(
        (1 - math.log(math.tan(lat) + 1 / math.cos(lat)) / math.pi)
        / 2
        * n
    )
    return x, y

def get_tile(z, x, y):
    key = (z, x, y)
    if key in tile_cache:
        return tile_cache[key]
    url = (
        f"https://api.mapbox.com/v4/mapbox.terrain-rgb/"
        f"{z}/{x}/{y}.pngraw?access_token={MAPBOX_TOKEN}"
    )
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
    fy = int(
        (
            1
            - math.log(
                math.tan(math.radians(lat))
                + 1 / math.cos(math.radians(lat))
            )
            / math.pi
        )
        / 2
        * h
    ) % h

    R, G, B = px[fx, fy][:3]
    meters = -10000 + (R * 256 * 256 + G * 256 + B) * 0.1
    return meters * 3.28084  # convert to feet

# ======================
# KML / KMZ
# ======================
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

    for f in root.findall(".//k:Folder", ns):
        name = f.find("k:name", ns)
        if name is None:
            continue

        if name.text.upper() == "AGMS":
            for p in f.findall(".//k:Placemark", ns):
                n = p.find("k:name", ns).text.strip()
                lon, lat, _ = map(
                    float, p.find(".//k:coordinates", ns).text.split(",")
                )
                agms.append((n, lat, lon))

        if name.text.upper() == "CENTERLINE":
            for ls in f.findall(".//k:LineString", ns):
                for c in ls.find("k:coordinates", ns).text.split():
                    lon, lat, _ = map(float, c.split(","))
                    center.append((lat, lon))

    return agms, center

# ======================
# STREAMLIT APP
# ======================
st.title("Terrain-Aware AGM Distance Checker")

upload = st.file_uploader("Drag & drop KML or KMZ", ["kml", "kmz"])

if upload:
    root = load_kml(upload)
    agms, center = parse(root)

    st.write(f"{len(agms)} AGMs | {len(center)} centerline pts")

    if not agms or not center:
        st.error("Need both AGMs and CENTERLINE folders")
        st.stop()

    agms.sort(key=lambda x: int(x[0]))
    center = densify(center)

    elevations = np.array([elevation_ft(lat, lon) for lat, lon in center])

    snapped = []
    for name, lat, lon in agms:
        dists = [haversine_ft(lat, lon, p[0], p[1]) for p in center]
        snapped.append((name, int(np.argmin(dists))))

    if snapped[0][1] > snapped[-1][1]:
        center.reverse()
        elevations = elevations[::-1]
        snapped = [(n, len(center) - 1 - i) for n, i in snapped]

    rows = []
    cumulative_mi = 0.0

    for i in range(len(snapped) - 1):
        a_name, a_idx = snapped[i]
        b_name, b_idx = snapped[i + 1]

        dist_ft = 0.0
        for j in range(a_idx, b_idx):
            h = haversine_ft(*center[j], *center[j + 1])
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
        df.to_csv(index=False).encode(),
        file_name="terrain_agm_distances.csv",
        mime="text/csv",
    )
