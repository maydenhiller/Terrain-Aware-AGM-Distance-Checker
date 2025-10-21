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
TERRAIN_TILE_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"

# Coordinate transformers
to_merc = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)   # lon/lat -> meters
to_geo = Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)    # meters -> lon/lat

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
                        centerline = LineString(pts)  # lon/lat
                except Exception:
                    continue

    agms.sort(key=agm_sort_key)
    return agms, centerline

def to_merc_point(pt_ll):
    x, y = to_merc.transform(pt_ll.x, pt_ll.y)
    return Point(x, y)

def to_merc_line(line_ll):
    return LineString([to_merc.transform(x, y) for (x, y) in line_ll.coords])

def to_geo_point(pt_m):
    lon, lat = to_geo.transform(pt_m.x, pt_m.y)
    return Point(lon, lat)

def slice_centerline_merc(centerline_m, p1_m, p2_m):
    # Distances along centerline in meters
    d1 = centerline_m.project(p1_m)
    d2 = centerline_m.project(p2_m)
    if d1 == d2:
        return None
    start, end = (d1, d2) if d1 < d2 else (d2, d1)
    seg = substring(centerline_m, start, end, normalized=False)
    if seg is None or seg.length == 0.0 or len(seg.coords) < 2:
        return None
    return seg

def interpolate_line_merc(line_m, spacing_m=1.0):
    total_len = line_m.length
    steps = max(int(total_len / spacing_m), 1)
    points = [line_m.interpolate(i * spacing_m) for i in range(steps)]
    points.append(line_m.interpolate(total_len))
    return points  # Points in EPSG:3857 meters

# --- Mapbox Terrain-RGB elevation sampling ---

def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return int(x), int(y), x, y

def pixel_in_tile(x_tile, y_tile, x_float, y_float):
    x_pix = int((x_float - x_tile) * 256.0)
    y_pix = int((y_float - y_tile) * 256.0)
    x_pix = max(0, min(255, x_pix))
    y_pix = max(0, min(255, y_pix))
    return x_pix, y_pix

def decode_terrain_rgb(r, g, b):
    # Elevation in meters per Mapbox formula
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

    def elevation_at_lonlat(self, lon, lat):
        z = self.zoom
        x_tile, y_tile, x_float, y_float = lonlat_to_tile(lon, lat, z)
        x_pix, y_pix = pixel_in_tile(x_tile, y_tile, x_float, y_float)
        img = self.get_tile_image(z, x_tile, y_tile)
        if img is None:
            return None
        r, g, b = img.getpixel((x_pix, y_pix))
        return decode_terrain_rgb(r, g, b)

def get_elevations_for_merc_points(points_m, cache: TerrainTileCache):
    elevations = []
    for pm in points_m:
        # convert each meter-based point back to lon/lat for Mapbox sampling
        plonlat = to_geo_point(pm)
        elev = cache.elevation_at_lonlat(plonlat.x, plonlat.y)
        if elev is None or not np.isfinite(elev):
            elevations.append(0.0)
        else:
            elevations.append(float(elev))
    return elevations

def distance_3d_between_merc_points(p1_m, p2_m, e1, e2):
    dx = p2_m.x - p1_m.x
    dy = p2_m.y - p1_m.y
    dz = e2 - e1
    return math.sqrt(dx * dx + dy * dy + dz * dz)

# --- STREAMLIT UI ---

st.title("Terrain-Aware AGM Distance Calculator (Mapbox Terrain-RGB, meter-accurate)")

uploaded_file = st.file_uploader("Upload KML or KMZ file", type=["kml", "kmz"])
if uploaded_file:
    agms_ll, centerline_ll = parse_kml_kmz(uploaded_file)

    st.subheader("📌 AGM summary")
    st.text(f"Total AGMs found: {len(agms_ll)}")
    st.subheader("📈 CENTERLINE status")
    st.text("CENTERLINE found" if centerline_ll else "CENTERLINE missing")

    if not centerline_ll or len(agms_ll) < 2:
        st.warning("Missing CENTERLINE or insufficient AGM points.")
    else:
        # Reproject centerline and AGMs to EPSG:3857 meters for true meter-based slicing
        centerline_m = to_merc_line(centerline_ll)
        agms_m = [(name, to_merc_point(pt)) for (name, pt) in agms_ll]

        rows = []
        cumulative_miles = 0.0
        skipped = 0

        # Initialize Mapbox tile cache at max precision zoom
        tile_cache = TerrainTileCache(token=MAPBOX_TOKEN, zoom=15)

        for i in range(len(agms_m) - 1):
            name1, pt1_m = agms_m[i]
            name2, pt2_m = agms_m[i + 1]

            segment_m = slice_centerline_merc(centerline_m, pt1_m, pt2_m)
            if segment_m is None or segment_m.length <= 0.0 or len(segment_m.coords) < 2:
                skipped += 1
                continue

            interp_points_m = interpolate_line_merc(segment_m, spacing_m=1.0)
            if len(interp_points_m) < 2:
                skipped += 1
                continue

            elevations = get_elevations_for_merc_points(interp_points_m, tile_cache)
            if len(elevations) != len(interp_points_m):
                skipped += 1
                continue

            # Sum 3D distance along the meter-based path at 1m spacing
            dist_m = 0.0
            for j in range(len(interp_points_m) - 1):
                dist_m += distance_3d_between_merc_points(
                    interp_points_m[j], interp_points_m[j + 1],
                    elevations[j], elevations[j + 1]
                )

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
