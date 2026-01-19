import streamlit as st
import zipfile
import io
import xml.etree.ElementTree as ET
import numpy as np
import requests
from math import sin, cos, sqrt, atan2, radians

# ==============================
# CONFIG
# ==============================
MAPBOX_TOKEN = "pk.eyJ1IjoibWF5ZGVuaGlsbGVyIiwiYSI6ImNtZ2ljMnN5ejA3amwyam9tNWZnYnZibWwifQ.GXoTyHdvCYtr7GvKIW9LPA"
DENSIFY_M = 10
EARTH_RADIUS_M = 6371000

# ==============================
# GEO UTILS
# ==============================
def haversine_m(lat1, lon1, lat2, lon2):
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * EARTH_RADIUS_M * atan2(sqrt(a), sqrt(1-a))

def densify(coords, step_m):
    out = [coords[0]]
    for i in range(len(coords)-1):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[i+1]
        d = haversine_m(lat1, lon1, lat2, lon2)
        n = max(1, int(d // step_m))
        for j in range(1, n+1):
            f = j / n
            out.append((
                lat1 + f*(lat2-lat1),
                lon1 + f*(lon2-lon1)
            ))
    return out

# ==============================
# MAPBOX TERRAIN SAMPLING
# ==============================
def tile_xy(lat, lon, z=14):
    lat_rad = radians(lat)
    n = 2 ** z
    xtile = int((lon + 180) / 360 * n)
    ytile = int((1 - np.log(np.tan(lat_rad) + 1 / cos(lat_rad)) / np.pi) / 2 * n)
    return xtile, ytile, z

def elevation_mapbox(lat, lon):
    x, y, z = tile_xy(lat, lon)
    url = f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw?access_token={MAPBOX_TOKEN}"
    r = requests.get(url)
    if r.status_code != 200:
        return 0
    from PIL import Image
    img = Image.open(io.BytesIO(r.content))
    px = img.load()
    n = img.size[0]
    fx = int((lon + 180) / 360 * n) % n
    fy = int((1 - np.log(np.tan(radians(lat)) + 1/cos(radians(lat))) / np.pi) / 2 * n) % n
    R, G, B = px[fx, fy][:3]
    return -10000 + (R*256*256 + G*256 + B) * 0.1

# ==============================
# KML / KMZ PARSER
# ==============================
def load_kml(upload):
    if upload.name.lower().endswith(".kmz"):
        z = zipfile.ZipFile(upload)
        kml = z.read([n for n in z.namelist() if n.endswith(".kml")][0])
    else:
        kml = upload.read()
    return ET.fromstring(kml)

def parse(root):
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    agms = []
    centerline = []

    for folder in root.findall(".//kml:Folder", ns):
        name = folder.find("kml:name", ns)
        if name is None:
            continue

        if name.text.strip().upper() == "AGMS":
            for pm in folder.findall(".//kml:Placemark", ns):
                n = pm.find("kml:name", ns).text
                coords = pm.find(".//kml:coordinates", ns)
                lon, lat, *_ = map(float, coords.text.strip().split(","))
                agms.append((n, lat, lon))

        if name.text.strip().upper() == "CENTERLINE":
            for ls in folder.findall(".//kml:LineString", ns):
                pts = []
                for c in ls.find("kml:coordinates", ns).text.strip().split():
                    lon, lat, *_ = map(float, c.split(","))
                    pts.append((lat, lon))
                centerline.extend(pts)

    return agms, centerline

# ==============================
# DISTANCE ENGINE
# ==============================
def terrain_distance(coords):
    coords = densify(coords, DENSIFY_M)
    elev = [elevation_mapbox(lat, lon) for lat, lon in coords]

    dist = 0
    for i in range(len(coords)-1):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[i+1]
        dz = elev[i+1] - elev[i]
        dxy = haversine_m(lat1, lon1, lat2, lon2)
        dist += sqrt(dxy**2 + dz**2)
    return dist

# ==============================
# STREAMLIT UI
# ==============================
st.title("Terrain-Aware AGM Distance Checker")

f = st.file_uploader("Drop KML or KMZ", type=["kml", "kmz"])

if f:
    root = load_kml(f)
    agms, centerline = parse(root)

    st.write(f"{len(agms)} AGMs | {len(centerline)} centerline pts")

    if agms and centerline:
        total_m = terrain_distance(centerline)
        st.success(f"Total terrain-aware distance: {total_m/1609.34:.3f} miles")
    else:
        st.error("Need both AGMs and CENTERLINE")
