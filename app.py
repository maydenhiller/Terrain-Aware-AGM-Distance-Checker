import io
import math
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

# Coordinate transformers (lon/lat <-> Web Mercator meters)
to_merc = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
to_geo = Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)

# --- KML PARSING ---
def parse_kml_kmz(uploaded_file):
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
        fname = name_el.text.strip().lower()

        if fname == "agms":
            for pm in folder.findall("kml:Placemark", ns):
                pname = pm.find("kml:name", ns)
                coords = pm.find(".//kml:coordinates", ns)
                if pname is None or coords is None:
                    continue
                try:
                    lon, lat, *_ = map(float, coords.text.strip().split(","))
                    agms.append((pname.text.strip(), Point(lon, lat)))
                except Exception:
                    continue

        elif fname == "centerline":
            for pm in folder.findall("kml:Placemark", ns):
                coords = pm.find(".//kml:coordinates", ns)
                if coords is None:
                    continue
                pts = []
                for pair in coords.text.strip().split():
                    lon, lat, *_ = map(float, pair.split(","))
                    pts.append((lon, lat))
                if len(pts) >= 2:
                    centerline = LineString(pts)

    return agms, centerline

# --- PROJECTION HELPERS ---
def to_merc_point(pt_ll):
    x, y = to_merc.transform(pt_ll.x, pt_ll.y)
    return Point(x, y)

def to_merc_line(line_ll):
    return LineString([to_merc.transform(x, y) for (x, y) in line_ll.coords])

def to_geo_point(pt_m):
    lon, lat = to_geo.transform(pt_m.x, pt_m.y)
    return Point(lon, lat)

# --- SLICING & INTERPOLATION ---
def slice_centerline(line_m, p1_m, p2_m):
    d1 = line_m.project(p1_m)
    d2 = line_m.project(p2_m)
    if d1 == d2:
        return None
    start, end = (d1, d2) if d1 < d2 else (d2, d1)
    seg = substring(line_m, start, end, normalized=False)
    if not seg or seg.length <= 0.0 or len(seg.coords) < 2:
        return None
    return seg

def interpolate_line(line_m, spacing=1.0):
    total = line_m.length
    steps = max(int(total / spacing), 1)
    pts = [line_m.interpolate(i * spacing) for i in range(steps)]
    pts.append(line_m.interpolate(total))
    return pts

# --- Mapbox Terrain-RGB sampling ---
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
    return -10000.0 + (r * 256.0 * 256.0 + g * 256.0 + b) * 0.1

class TerrainTileCache:
    def __init__(self, token, zoom=15):
        self.token = token
        self.zoom = zoom
        self.cache = {}

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
            return 0.0
        r, g, b = img.getpixel((x_pix, y_pix))
        return decode_terrain_rgb(r, g, b)

def get_elevations_for_merc_points(points_m, cache: TerrainTileCache):
    elevs = []
    for pm in points_m:
        lonlat = to_geo_point(pm)
        elev = cache.elevation_at_lonlat(lonlat.x, lonlat.y)
        elevs.append(float(elev))
    return elevs

# --- Distance calculations ---
def dist3d_between_merc_points(p1_m, p2_m, e1, e2):
    dx = p2_m.x - p1_m.x
    dy = p2_m.y - p1_m.y
    dz = e2 - e1
    return math.sqrt(dx * dx + dy * dy + dz * dz)

# --- STREAMLIT UI ---
st.title("Terrain-Aware AGM Distance Calculator (Spatial-order corrected)")

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
        # Project centerline and AGMs to meters
        centerline_m = to_merc_line(centerline_ll)
        agms_m = [(name, to_merc_point(pt)) for (name, pt) in agms_ll]

        # Compute along-centerline measure (in meters) for each AGM and sort by that, not by name
        agm_measures = []
        for name, pt_m in agms_m:
            s = centerline_m.project(pt_m)
            agm_measures.append({"name": name, "pt_m": pt_m, "s_m": s})
        agm_measures.sort(key=lambda r: r["s_m"])

        # Optional: show reordering diagnostics
        diag = pd.DataFrame({
            "AGM name": [r["name"] for r in agm_measures],
            "Along-line meters": [round(r["s_m"], 2) for r in agm_measures]
        })
        st.subheader("🔎 AGM spatial order along centerline")
        st.dataframe(diag, use_container_width=True)

        # Build distances in spatial order
        rows = []
        cumulative_miles = 0.0
        skipped = 0
        tile_cache = TerrainTileCache(token=MAPBOX_TOKEN, zoom=15)

        for i in range(len(agm_measures) - 1):
            n1, p1_m = agm_measures[i]["name"], agm_measures[i]["pt_m"]
            n2, p2_m = agm_measures[i + 1]["name"], agm_measures[i + 1]["pt_m"]

            segment_m = slice_centerline(centerline_m, p1_m, p2_m)
            if segment_m is None:
                skipped += 1
                continue

            # 1-meter path points
            pts_m = interpolate_line(segment_m, spacing=1.0)
            if len(pts_m) < 2:
                skipped += 1
                continue

            elevs = get_elevations_for_merc_points(pts_m, tile_cache)
            if len(elevs) != len(pts_m):
                skipped += 1
                continue

            # 2D and 3D distances
            d2d_m = segment_m.length
            d3d_m = sum(
                dist3d_between_merc_points(pts_m[j], pts_m[j + 1], elevs[j], elevs[j + 1])
                for j in range(len(pts_m) - 1)
            )

            d2d_mi = d2d_m / 1609.34
            d3d_mi = d3d_m / 1609.34
            cumulative_miles += d3d_mi

            rows.append({
                "From AGM": n1,
                "To AGM": n2,
                "2D miles": round(d2d_mi, 6),
                "3D miles": round(d3d_mi, 6),
                "Ratio 3D/2D": round(d3d_mi / d2d_mi if d2d_mi > 0 else 0, 3),
                "Cumulative 3D miles": round(cumulative_miles, 6)
            })

        st.subheader("📊 Distance table (spatial order)")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.text(f"Skipped segments: {skipped}")

        csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "terrain_distances_spatial_order.csv", "text/csv")
