import math
import io
import zipfile
import requests
import xml.etree.ElementTree as ET

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Point, LineString
from shapely.ops import substring
from pyproj import Transformer

# --- CONFIG ---
MAPBOX_TOKEN = st.secrets["mapbox"]["token"]
# Terrain-RGB tileset endpoint (PNG raw for precise RGB)
TERRAIN_TILE_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"

# Use Web Mercator for horizontal distance (meters)
transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)

# --- HELPERS ---

def agm_sort_key(name_geom):
    name = name_geom[0]
    base_digits = ''.join(filter(str.isdigit, name))
    base = int(base_digits) if base_digits else -1
    suffix = ''.join(filter(str.isalpha, name))
    return (base, suffix)

def parse_kml_kmz(uploaded_file):
    # Load KML from KMZ or KML
    if uploaded_file.name.endswith(".kmz"):
        with zipfile.ZipFile(uploaded_file) as zf:
            kml_file = next((f for f in zf.namelist() if f.endswith(".kml")), None)
            if not kml_file:
                return [], None
            with zf.open(kml_file) as f:
                kml_data = f.read()
    else:
        kml_data = uploaded_file.read()

    root = ET.fromstring(kml_data)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    agms = []
    centerline = None

    for folder in root.findall(".//kml:Folder", ns):
        name_el = folder.find("kml:name", ns)
        if name_el is None or not name_el.text:
            continue
        folder_name = name_el.text.strip().lower()

        if folder_name == "agms":
            for placemark in folder.findall("kml:Placemark", ns):
                pname = placemark.find("kml:name", ns)
                coords = placemark.find(".//kml:coordinates", ns)
                if pname is None or coords is None:
                    continue
                try:
                    lon, lat, *_ = map(float, coords.text.strip().split(","))
                    agms.append((pname.text.strip(), Point(lon, lat)))
                except Exception:
                    continue

        elif folder_name == "centerline":
            for placemark in folder.findall("kml:Placemark", ns):
                coords = placemark.find(".//kml:coordinates", ns)
                if coords is None:
                    continue
                try:
                    pts = []
                    for pair in coords.text.strip().split():
                        lon, lat, *_ = map(float, pair.split(","))
                        pts.append((lon, lat))
                    if len(pts) >= 2:
                        centerline = LineString(pts)
                except Exception:
                    continue

    agms.sort(key=agm_sort_key)
    return agms, centerline

def slice_centerline(centerline, p1, p2):
    # Project AGM points onto the centerline; use distances along the line
    d1 = centerline.project(p1)
    d2 = centerline.project(p2)
    if d1 == d2:
        return None
    start, end = (d1, d2) if d1 < d2 else (d2, d1)
    # Robust geometric slicing along the path
    seg = substring(centerline, start, end, normalized=False)
    if seg is None or seg.length == 0.0 or len(seg.coords) < 2:
        return None
    return seg

def interpolate_line(line, spacing_m=1.0):
    total_length = line.length
    steps = max(int(total_length / spacing_m), 1)
    # Ensure endpoints included
    points = [line.interpolate(i * spacing_m) for i in range(steps)]
    points.append(line.interpolate(total_length))
    return points

# --- Mapbox Terrain-RGB elevation sampling ---

def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return int(x), int(y), x, y

def pixel_in_tile(lon, lat, z, x_tile, y_tile, x_float, y_float):
    # Convert fractional tile position to pixel coordinate (256x256)
    x_pix = int((x_float - x_tile) * 256.0)
    y_pix = int((y_float - y_tile) * 256.0)
    # Clamp to tile bounds
    x_pix = max(0, min(255, x_pix))
    y_pix = max(0, min(255, y_pix))
    return x_pix, y_pix

def decode_terrain_rgb(r, g, b):
    # Elevation in meters: E = -10000 + (R*256^2 + G*256 + B) * 0.1
    return -10000.0 + (r * 256.0 * 256.0 + g * 256.0 + b) * 0.1

class TerrainTileCache:
    def __init__(self, token, zoom=15):
        self.token = token
        self.zoom = zoom
        self.cache = {}  # (z,x,y) -> PIL Image

    def get_tile_image(self, z, x, y):
        key = (z, x, y)
        img = self.cache.get(key)
        if img is not None:
            return img
        url = TERRAIN_TILE_URL.format(z=z, x=x, y=y)
        resp = requests.get(url, params={"access_token": self.token}, timeout=20)
        if resp.status_code != 200:
            return None
        try:
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception:
            return None
        self.cache[key] = img
        return img

    def elevation_at(self, lon, lat):
        z = self.zoom
        x_tile, y_tile, x_float, y_float = lonlat_to_tile(lon, lat, z)
        x_pix, y_pix = pixel_in_tile(lon, lat, z, x_tile, y_tile, x_float, y_float)
        img = self.get_tile_image(z, x_tile, y_tile)
        if img is None:
            return None
        r, g, b = img.getpixel((x_pix, y_pix))
        return decode_terrain_rgb(r, g, b)

def get_elevations(points, cache: TerrainTileCache):
    elevations = []
    for p in points:
        elev = cache.elevation_at(p.x, p.y)
        if elev is None or not np.isfinite(elev):
            elevations.append(0.0)  # fallback
        else:
            elevations.append(float(elev))
    return elevations

def distance_3d(p1, p2, e1, e2):
    # Horizontal in meters via EPSG:3857 + vertical component
    x1, y1 = transformer.transform(p1.x, p1.y)
    x2, y2 = transformer.transform(p2.x, p2.y)
    dx = x2 - x1
    dy = y2 - y1
    dz = e2 - e1
    return math.sqrt(dx * dx + dy * dy + dz * dz)

# --- STREAMLIT UI ---

st.title("Terrain-Aware AGM Distance Calculator (Mapbox Terrain-RGB)")

uploaded_file = st.file_uploader("Upload KML or KMZ file", type=["kml", "kmz"])
if uploaded_file:
    agms, centerline = parse_kml_kmz(uploaded_file)

    st.subheader("📌 AGM summary")
    st.text(f"Total AGMs found: {len(agms)}")
    st.subheader("📈 CENTERLINE status")
    st.text("CENTERLINE found" if centerline else "CENTERLINE missing")

    if not centerline or len(agms) < 2:
        st.warning("Missing CENTERLINE or insufficient AGM points.")
    else:
        rows = []
        cumulative_miles = 0.0
        skipped = 0

        # Initialize Mapbox tile cache at max precision zoom
        tile_cache = TerrainTileCache(token=MAPBOX_TOKEN, zoom=15)

        for i in range(len(agms) - 1):
            name1, pt1 = agms[i]
            name2, pt2 = agms[i + 1]

            segment = slice_centerline(centerline, pt1, pt2)
            if segment is None or segment.length == 0.0 or len(segment.coords) < 2:
                skipped += 1
                continue

            interp_points = interpolate_line(segment, spacing_m=1.0)
            if len(interp_points) < 2:
                skipped += 1
                continue

            elevations = get_elevations(interp_points, tile_cache)
            if len(elevations) != len(interp_points):
                skipped += 1
                continue

            # Sum 3D distance along the path at 1m spacing
            dist_m = 0.0
            for j in range(len(interp_points) - 1):
                dist_m += distance_3d(interp_points[j], interp_points[j + 1],
                                      elevations[j], elevations[j + 1])

            dist_ft = dist_m * 3.28084
            dist_mi = dist_ft / 5280.0
            cumulative_miles += dist_mi

            rows.append({
                "From AGM": name1,
                "To AGM": name2,
                "Distance (feet)": round(dist_ft, 2),
                "Distance (miles)": round(dist_mi, 6),
                "Cumulative Distance (miles)": round(cumulative_miles, 6)
            })

        st.subheader("📊 Distance table")
        df = pd.DataFrame(rows)
        st.dataframe(df)
        st.text(f"Skipped segments: {skipped}")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "terrain_distances.csv", "text/csv")
