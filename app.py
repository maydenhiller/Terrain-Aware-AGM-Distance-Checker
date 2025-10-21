# app.py
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
from pyproj import Geod, CRS, Transformer

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="Terrain-Aware AGM Distance Calculator", layout="wide")
st.title("Terrain-Aware AGM Distance Calculator (Snap-to-Centerline, Geodesic 3D)")

MAPBOX_TOKEN = st.secrets["mapbox"]["token"]
TERRAIN_TILE_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"
GEOD = Geod(ellps="WGS84")

with st.sidebar:
    st.header("Settings")
    mapbox_zoom = st.slider("Terrain tile zoom", 15, 17, 17)
    interp_spacing_m = st.slider("Sampling spacing along path (m)", 0.5, 5.0, 1.0, 0.5)
    smooth_window = st.slider("Elevation smoothing window (samples)", 1, 21, 5, 2)
    st.caption("Set zoom=17 and spacing=1 m for highest accuracy.")

# =========================
# HELPERS: parsing
# =========================

def agm_sort_key(name_geom):
    name = name_geom[0]
    base_digits = ''.join(filter(str.isdigit, name))
    base = int(base_digits) if base_digits else -1
    suffix = ''.join(filter(str.isalpha, name)).upper()
    return (base, suffix)

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
                pts = []
                for pair in coords.text.strip().split():
                    lon, lat, *_ = map(float, pair.split(","))
                    pts.append((lon, lat))
                if len(pts) >= 2:
                    centerline = LineString(pts)

    agms.sort(key=agm_sort_key)
    return agms, centerline

# =========================
# HELPERS: CRS & transforms
# =========================

def get_local_utm_crs(line_ll: LineString) -> CRS:
    xs = [c[0] for c in line_ll.coords]
    ys = [c[1] for c in line_ll.coords]
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    zone = int((cx + 180.0) / 6.0) + 1
    is_north = cy >= 0.0
    epsg = 32600 + zone if is_north else 32700 + zone
    return CRS.from_epsg(epsg)

def transformer_ll_to(crs: CRS) -> Transformer:
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)

def transformer_to_ll(crs: CRS) -> Transformer:
    return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

def transform_linestring(ls: LineString, xf: Transformer) -> LineString:
    xs, ys = zip(*ls.coords)
    X, Y = xf.transform(xs, ys)
    return LineString(list(zip(X, Y)))

def transform_point(pt: Point, xf: Transformer) -> Point:
    x, y = xf.transform(pt.x, pt.y)
    return Point(x, y)

# =========================
# AGM SNAP to CENTERLINE (true nearest point)
# =========================

def snap_point_to_centerline_metric(pt_ll: Point, line_ll: LineString, crs_metric: CRS):
    xf_ll_to_m = transformer_ll_to(crs_metric)
    xf_m_to_ll = transformer_to_ll(crs_metric)

    line_m = transform_linestring(line_ll, xf_ll_to_m)
    pt_m = transform_point(pt_ll, xf_ll_to_m)

    s = line_m.project(pt_m)
    snapped_m = line_m.interpolate(s)

    lon, lat = xf_m_to_ll.transform(snapped_m.x, snapped_m.y)
    return Point(lon, lat), s, line_m  # return also s and metric line for reuse

# =========================
# GEODESIC LENGTH
# =========================

def geodesic_length_ll(coords_ll):
    total = 0.0
    for i in range(len(coords_ll) - 1):
        lon1, lat1 = coords_ll[i]
        lon2, lat2 = coords_ll[i + 1]
        _, _, d = GEOD.inv(lon1, lat1, lon2, lat2)
        total += d
    return total

# =========================
# MAPBOX TERRAIN-RGB (bilinear)
# =========================

def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return int(x), int(y), x, y

def decode_terrain_rgb(r, g, b):
    return -10000.0 + (r * 256.0 * 256.0 + g * 256.0 + b) * 0.1

class TerrainTileCache:
    def __init__(self, token, zoom=17):
        self.token = token
        self.zoom = zoom
        self.cache = {}

    def get_tile_array(self, z, x, y):
        key = (z, x, y)
        arr = self.cache.get(key)
        if arr is not None:
            return arr
        url = TERRAIN_TILE_URL.format(z=z, x=x, y=y)
        resp = requests.get(url, params={"access_token": self.token}, timeout=20)
        if resp.status_code != 200:
            return None
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        self.cache[key] = arr
        return arr

    def elevation_at_bilinear(self, lon, lat):
        z = self.zoom
        x_tile, y_tile, x_f, y_f = lonlat_to_tile(lon, lat, z)
        x_pix_f = (x_f - x_tile) * 256.0
        y_pix_f = (y_f - y_tile) * 256.0
        x0, y0 = int(math.floor(x_pix_f)), int(math.floor(y_pix_f))
        dx, dy = x_pix_f - x0, y_pix_f - y0
        x0 = max(0, min(255, x0))
        y0 = max(0, min(255, y0))
        x1 = min(x0 + 1, 255)
        y1 = min(y0 + 1, 255)
        arr = self.get_tile_array(z, x_tile, y_tile)
        if arr is None:
            return None
        p00 = decode_terrain_rgb(*arr[y0, x0])
        p10 = decode_terrain_rgb(*arr[y0, x1])
        p01 = decode_terrain_rgb(*arr[y1, x0])
        p11 = decode_terrain_rgb(*arr[y1, x1])
        elev = (
            p00 * (1 - dx) * (1 - dy)
            + p10 * dx * (1 - dy)
            + p01 * (1 - dx) * dy
            + p11 * dx * dy
        )
        return float(elev)

def get_elevations_ll(points_ll, cache: TerrainTileCache):
    elevs = []
    for lon, lat in points_ll:
        e = cache.elevation_at_bilinear(lon, lat)
        elevs.append(0.0 if e is None else e)
    return elevs

def smooth_elevations(elevs, window):
    if window <= 1:
        return elevs
    kernel = np.ones(window) / window
    return np.convolve(elevs, kernel, mode="same").tolist()

# =========================
# MAIN APP
# =========================

uploaded_file = st.file_uploader("Upload KML or KMZ file", type=["kml", "kmz"])

if uploaded_file:
    agms, centerline_ll = parse_kml_kmz(uploaded_file)

    st.subheader("📌 AGM summary")
    st.text(f"Total AGMs found: {len(agms)}")
    st.subheader("📈 CENTERLINE status")
    st.text("CENTERLINE found" if centerline_ll else "CENTERLINE missing")

    if not centerline_ll or len(agms) < 2:
        st.warning("Missing CENTERLINE or insufficient AGM points.")
    else:
        # Build local UTM for precise snapping/slicing
        try:
            crs_metric = get_local_utm_crs(centerline_ll)
        except Exception:
            # Fallback: CONUS Albers (good enough if UTM fails)
            crs_metric = CRS.from_epsg(5070)
        xf_ll_to_m = transformer_ll_to(crs_metric)
        xf_m_to_ll = transformer_to_ll(crs_metric)

        centerline_m = transform_linestring(centerline_ll, xf_ll_to_m)

        # Snap AGMs to centerline (true nearest point), get their measures s along metric centerline
        snapped = []
        for name, pt_ll in agms:
            pt_m = transform_point(pt_ll, xf_ll_to_m)
            s = centerline_m.project(pt_m)
            snap_m = centerline_m.interpolate(s)
            lon_s, lat_s = xf_m_to_ll.transform(snap_m.x, snap_m.y)
            snapped.append((name, Point(lon_s, lat_s), s))

        # Prepare elevation cache
        tile_cache = TerrainTileCache(MAPBOX_TOKEN, zoom=mapbox_zoom)

        rows = []
        skipped = 0
        cumulative_miles = 0.0

        for i in range(len(snapped) - 1):
            name1, pt1_ll_snap, s1 = snapped[i]
            name2, pt2_ll_snap, s2 = snapped[i + 1]

            if np.isclose(s1, s2):
                skipped += 1
                continue

            start, end = (s1, s2) if s1 < s2 else (s2, s1)

            # Slice the metric centerline exactly between projected distances
            seg_m = substring(centerline_m, start, end, normalized=False)
            if seg_m is None or seg_m.length <= 0 or len(seg_m.coords) < 2:
                skipped += 1
                continue

            # Interpolate along the metric segment every interp_spacing_m
            L = seg_m.length
            steps = max(int(L / interp_spacing_m), 1)
            dists = [k * interp_spacing_m for k in range(steps)]
            pts_m = [seg_m.interpolate(d) for d in dists]
            pts_m.append(seg_m.interpolate(L))

            # Transform sampled points back to lon/lat for elevation + geodesic dxy
            xs = [p.x for p in pts_m]
            ys = [p.y for p in pts_m]
            lons, lats = xf_m_to_ll.transform(xs, ys)
            interp_pts_ll = list(zip(lons, lats))

            # Elevation series
            elevations = smooth_elevations(get_elevations_ll(interp_pts_ll, tile_cache), smooth_window)

            # Accumulate 3D distance using geodesic horizontal component
            dist_m = 0.0
            for j in range(len(interp_pts_ll) - 1):
                lon1, lat1 = interp_pts_ll[j]
                lon2, lat2 = interp_pts_ll[j + 1]
                _, _, dxy = GEOD.inv(lon1, lat1, lon2, lat2)  # meters
                dz = elevations[j + 1] - elevations[j]
                dist_m += math.sqrt(dxy * dxy + dz * dz)

            dist_ft = dist_m * 3.28084
            dist_mi = dist_ft / 5280.0
            cumulative_miles += dist_mi

            rows.append({
                "From AGM": name1,
                "To AGM": name2,
                "Distance (ft)": round(dist_ft, 2),
                "Distance (mi)": round(dist_mi, 6),
                "Cumulative (mi)": round(cumulative_miles, 6)
            })

        st.subheader("📊 Distance table")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.text(f"Skipped segments: {skipped}")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "terrain_distances.csv", "text/csv")
