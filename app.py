import os
import math
import re
import io
import zipfile
import requests
import pandas as pd
import streamlit as st
import xml.etree.ElementTree as ET
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
from pyproj import CRS, Transformer
import srtm  # pip install srtm.py

# ---------------------------- Config ----------------------------
FT_PER_M = 3.28084
MI_PER_FT = 1 / 5280
DEFAULT_STEP_M = 5.0  # densification step along centerline in meters
EPQS_URL = "https://nationalmap.gov/epqs/pqs.php"
OPENTOPO_URL = "https://portal.opentopography.org/API/point"  # requires key
OPENTOPO_KEY = os.getenv("OPENTOPO_KEY")  # set in Streamlit secrets or env
OPENTOPO_DEMTYPES = ["USGS3DEP1m", "USGSNED10m", "SRTMGL1"]  # priority
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"

# ---------------------------- Streamlit setup ----------------------------
st.set_page_config(page_title="Terrain-aware AGM distances", layout="wide")
st.title("📏 Terrain-aware AGM distances (KML/KMZ)")

with st.expander("Advanced options", expanded=False):
    step_m = st.number_input("Densification step (meters)", min_value=1.0, max_value=50.0, value=DEFAULT_STEP_M, step=1.0, help="Smaller is more accurate but slower.")
    debug = st.checkbox("Enable debug mode", value=False)

uploaded = st.file_uploader("Upload KML or KMZ with ‘AGMs’ (Points) and ‘Centerline’ (LineString)", type=["kml", "kmz"])

# ---------------------------- Utility functions ----------------------------
def parse_station(label: str) -> tuple[int, str]:
    m = re.match(r"^(\d+)([A-Za-z]*)$", label.strip())
    if not m:
        return 0, ""
    return int(m.group(1)), m.group(2)

def extract_kml_from_kmz(data: bytes) -> bytes | None:
    with zipfile.ZipFile(io.BytesIO(data), 'r') as zf:
        for name in zf.namelist():
            if name.lower().endswith(".kml"):
                return zf.read(name)
    return None

def parse_kml_kmz(file) -> tuple[list[tuple[str, float, float]], list[tuple[float, float]]]:
    raw = file.read()
    if file.name.lower().endswith(".kmz"):
        raw = extract_kml_from_kmz(raw) or b""

    if not raw:
        return [], []

    root = ET.fromstring(raw)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    # Extract AGMs: Points under Folder named “AGMs”
    agms: list[tuple[str, float, float]] = []
    for fld in root.findall(".//kml:Folder", ns):
        nm = fld.find("kml:name", ns)
        if nm is not None and (nm.text or "").strip().lower() == "agms":
            for pm in fld.findall(".//kml:Placemark", ns):
                name_el = pm.find("kml:name", ns)
                coord_el = pm.find(".//kml:Point/kml:coordinates", ns)
                if name_el is None or coord_el is None:
                    continue
                label = (name_el.text or "").strip()
                ctext = (coord_el.text or "").strip()
                if not ctext:
                    continue
                lon, lat, *_ = ctext.split(",")
                agms.append((label, float(lat), float(lon)))

    # Extract Centerline: LineString(s) under Folder named “Centerline”
    lines_wgs84: list[list[tuple[float, float]]] = []
    for fld in root.findall(".//kml:Folder", ns):
        nm = fld.find("kml:name", ns)
        if nm is not None and (nm.text or "").strip().lower() == "centerline":
            for ls in fld.findall(".//kml:LineString", ns):
                coords_el = ls.find("kml:coordinates", ns)
                if coords_el is None or not (coords_el.text or "").strip():
                    continue
                coords = []
                for pair in coords_el.text.strip().split():
                    lon, lat, *_ = pair.split(",")
                    coords.append((float(lon), float(lat)))
                if len(coords) >= 2:
                    lines_wgs84.append(coords)

    # Fallback: also capture any Placemark named exactly “Centerline”
    if not lines_wgs84:
        for pm in root.findall(".//kml:Placemark", ns):
            name_el = pm.find("kml:name", ns)
            if name_el is None or (name_el.text or "").strip().lower() != "centerline":
                continue
            for ls in pm.findall(".//kml:LineString", ns):
                coords_el = ls.find("kml:coordinates", ns)
                if coords_el is None or not (coords_el.text or "").strip():
                    continue
                coords = []
                for pair in coords_el.text.strip().split():
                    lon, lat, *_ = pair.split(",")
                    coords.append((float(lon), float(lat)))
                if len(coords) >= 2:
                    lines_wgs84.append(coords)

    # Flatten to a single path of (lat, lon) tuples for legacy compatibility
    centerline_ll: list[tuple[float, float]] = [(lat, lon) for line in lines_wgs84 for (lon, lat) in line]
    return agms, centerline_ll

def utm_crs_for(lats: list[float], lons: list[float]) -> CRS:
    lat_mean = sum(lats) / len(lats)
    lon_mean = sum(lons) / len(lons)
    zone = int((lon_mean + 180) // 6) + 1
    if lat_mean >= 0:
        epsg = 32600 + zone  # WGS84 / UTM Northern
    else:
        epsg = 32700 + zone  # WGS84 / UTM Southern
    return CRS.from_epsg(epsg)

@st.cache_resource(show_spinner=False)
def get_srtm():
    return srtm.get_data()

srtm_data = get_srtm()

@st.cache_data(show_spinner=False, max_entries=200000, ttl=24*3600)
def get_elevation(lat: float, lon: float) -> float:
    # 1) USGS EPQS (best available; can be 1 m LiDAR in many US areas)
    try:
        r = requests.get(
            EPQS_URL,
            params={"x": lon, "y": lat, "units": "Meters", "output": "json"},
            timeout=6,
        )
        r.raise_for_status()
        e = r.json()["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]["Elevation"]
        if e is not None:
            return float(e)
    except Exception:
        pass

    # 2) OpenTopography (if key provided) with preferred DEMs
    if OPENTOPO_KEY:
        for dem in OPENTOPO_DEMTYPES:
            try:
                r = requests.get(
                    OPENTOPO_URL,
                    params={"x": lon, "y": lat, "demtype": dem, "outputFormat": "JSON", "key": OPENTOPO_KEY},
                    timeout=6,
                )
                r.raise_for_status()
                j = r.json()
                if "data" in j and "elevation" in j["data"]:
                    return float(j["data"]["elevation"])
            except Exception:
                continue

    # 3) Local SRTM tiles as a robust fallback (≈30 m)
    try:
        e = srtm_data.get_elevation(lat, lon)
        if e is not None:
            return float(e)
    except Exception:
        pass

    # 4) Open-Elevation final fallback
    try:
        r = requests.get(OPEN_ELEVATION_URL, params={"locations": f"{lat:.6f},{lon:.6f}"}, timeout=6)
        r.raise_for_status()
        return float(r.json()["results"][0]["elevation"])
    except Exception:
        return 0.0

def build_centerline(lines_wgs84: list[list[tuple[float, float]]]) -> LineString:
    """Given list(s) of (lon,lat) sequences, merge to one LineString in projected meters."""
    if not lines_wgs84:
        return LineString()

    # Pick UTM based on all centerline vertices
    all_lons = [lon for line in lines_wgs84 for (lon, _) in line]
    all_lats = [lat for line in lines_wgs84 for (_, lat) in line]
    crs_utm = utm_crs_for(all_lats, all_lons)
    to_utm = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
    # Transform each line to UTM and keep as shapely LineStrings
    utm_lines = []
    for line in lines_wgs84:
        xs, ys = to_utm.transform(*zip(*line))
        utm_lines.append(LineString(list(zip(xs, ys))))

    # Merge contiguous lines if possible
    merged = linemerge(utm_lines if len(utm_lines) > 1 else utm_lines[0])
    if isinstance(merged, LineString):
        return merged
    # If still MultiLineString, choose the longest and append others greedily by nearest endpoints
    parts = list(merged.geoms)  # type: ignore[attr-defined]
    parts.sort(key=lambda g: g.length, reverse=True)
    base = parts.pop(0)
    while parts:
        # Find the line whose end is closest to either end of base; reverse if needed
        best_i, best_dist, best_flip, attach_to_end = None, float("inf"), False, True
        bx0, by0 = base.coords[0]
        bx1, by1 = base.coords[-1]
        for i, seg in enumerate(parts):
            sx0, sy0 = seg.coords[0]
            sx1, sy1 = seg.coords[-1]
            d_end_start = math.hypot(bx1 - sx0, by1 - sy0)
            d_end_end = math.hypot(bx1 - sx1, by1 - sy1)
            d_start_start = math.hypot(bx0 - sx0, by0 - sy0)
            d_start_end = math.hypot(bx0 - sx1, by0 - sy1)
            # Prefer attaching to base end, else base start
            candidates = [
                ("end", False, d_end_start),
                ("end", True, d_end_end),
                ("start", True, d_start_start),
                ("start", False, d_start_end),
            ]
            side, flip, dist = min(candidates, key=lambda t: t[2])
            if dist < best_dist:
                best_i, best_dist = i, dist
                best_flip = flip
                attach_to_end = (side == "end")
        seg = parts.pop(best_i)  # type: ignore[arg-type]
        coords = list(seg.coords)
        if best_flip:
            coords = list(reversed(coords))
        if attach_to_end:
            base = LineString(list(base.coords) + coords)
        else:
            base = LineString(coords + list(base.coords))
    return base

def densified_points(line_utm: LineString, s0: float, s1: float, step: float) -> list[tuple[float, float]]:
    """Return list of projected (x,y) along line from s0 to s1 at given step, inclusive of endpoints."""
    a, b = (s0, s1) if s0 <= s1 else (s1, s0)
    if b - a < 1e-6:
        b = a + step  # ensure at least one step
    dists = [a + i * step for i in range(int(math.floor((b - a) / step)) + 1)]
    if dists[-1] < b:
        dists.append(b)
    pts = [line_utm.interpolate(s) for s in dists]
    return [(p.x, p.y) for p in pts]

def terrain_distance_m(line_utm: LineString, pts_xy: list[tuple[float, float]], to_wgs84: Transformer) -> float:
    """Sum 3D distances between consecutive densified points using elevations."""
    total = 0.0
    # Convert all projected points to lat/lon once, then fetch elevations with cache
    xs, ys = zip(*pts_xy)
    lons, lats = to_wgs84.transform(xs, ys)
    elevs = [get_elevation(lat, lon) for lat, lon in zip(lats, lons)]
    for i in range(len(pts_xy) - 1):
        x1, y1 = pts_xy[i]
        x2, y2 = pts_xy[i + 1]
        h = math.hypot(x2 - x1, y2 - y1)  # horizontal in meters (projected)
        v = elevs[i + 1] - elevs[i]       # vertical in meters
        total += math.hypot(h, v)
    return total

# ---------------------------- Main flow ----------------------------
if not uploaded:
    st.info("Please upload a KML or KMZ file.")
    st.stop()

agms, centerline_latlon = parse_kml_kmz(uploaded)
if not agms:
    st.error("No AGMs found (Points in a Folder named ‘AGMs’).")
    st.stop()

# Rebuild centerline from LineStrings, not from flattened legacy list
# We need original lon/lat sequences to build a geometric line; reparse for lists of LineStrings
uploaded.seek(0)
raw_again = uploaded.read()
if uploaded.name.lower().endswith(".kmz"):
    raw_again = extract_kml_from_kmz(raw_again) or b""
root = ET.fromstring(raw_again)
ns = {"kml": "http://www.opengis.net/kml/2.2"}
lines_ll: list[list[tuple[float, float]]] = []
for fld in root.findall(".//kml:Folder", ns):
    nm = fld.find("kml:name", ns)
    if nm is not None and (nm.text or "").strip().lower() == "centerline":
        for ls in fld.findall(".//kml:LineString", ns):
            coords_el = ls.find("kml:coordinates", ns)
            if coords_el is None or not (coords_el.text or "").strip():
                continue
            coords = []
            for pair in coords_el.text.strip().split():
                lon, lat, *_ = pair.split(",")
                coords.append((float(lon), float(lat)))
            if len(coords) >= 2:
                lines_ll.append(coords)
# fallback by Placemark name
if not lines_ll:
    for pm in root.findall(".//kml:Placemark", ns):
        name_el = pm.find("kml:name", ns)
        if name_el is None or (name_el.text or "").strip().lower() != "centerline":
            continue
        for ls in pm.findall(".//kml:LineString", ns):
            coords_el = ls.find("kml:coordinates", ns)
            if coords_el is None or not (coords_el.text or "").strip():
                continue
            coords = []
            for pair in coords_el.text.strip().split():
                lon, lat, *_ = pair.split(",")
                coords.append((float(lon), float(lat)))
            if len(coords) >= 2:
                lines_ll.append(coords)

if not lines_ll:
    st.error("No Centerline found (LineString(s) under Folder named ‘Centerline’).")
    st.stop()

# Build merged centerline in projected meters
line_utm = build_centerline(lines_ll)
if line_utm.is_empty or line_utm.length <= 0:
    st.error("Centerline geometry could not be constructed.")
    st.stop()

# Transformers
# Choose UTM previously; build both directions
all_lons = [lon for line in lines_ll for (lon, _) in line]
all_lats = [lat for line in lines_ll for (_, lat) in line]
crs_utm = utm_crs_for(all_lats, all_lons)
to_utm = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
to_wgs84 = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)

# Sort AGMs by station label (numeric then suffix), then project each to chainage
agms_sorted = sorted(agms, key=lambda x: parse_station(x[0]))
agm_chain = []
for label, lat, lon in agms_sorted:
    x, y = to_utm.transform(lon, lat)
    d = line_utm.project(Point(x, y))  # meters along line
    agm_chain.append((label, lat, lon, d))

if debug:
    st.subheader("AGM projection to centerline (chainage in meters)")
    st.dataframe(pd.DataFrame(
        [{"AGM": lab, "lat": la, "lon": lo, "chainage_m": ch} for lab, la, lo, ch in agm_chain]
    ), use_container_width=True)
    st.write(f"Centerline total length (m): {line_utm.length:,.2f}")

# Compute 3D terrain-aware distances between consecutive AGMs
rows = []
total_ft = 0.0
for i in range(len(agm_chain) - 1):
    lab1, la1, lo1, s0 = agm_chain[i]
    lab2, la2, lo2, s1 = agm_chain[i + 1]
    pts_xy = densified_points(line_utm, s0, s1, step_m)
    d_m = terrain_distance_m(line_utm, pts_xy, to_wgs84)
    d_ft = d_m * FT_PER_M
    d_mi = d_ft * MI_PER_FT
    total_ft += d_ft
    rows.append({
        "Segment": f"Distance from {lab1} to {lab2}:",
        "Distance (ft)": f"{d_ft:,.2f}",
        "Distance (mi)": f"{d_mi:.4f}"
    })

df = pd.DataFrame(rows)
st.subheader("AGM segment distances (terrain-aware)")
st.dataframe(df, use_container_width=True)

tot_mi = total_ft * MI_PER_FT
st.markdown(f"**Total:** {total_ft:,.2f} ft  |  **{tot_mi:.4f} mi**")

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV", data=csv, file_name="agm_distances.csv", mime="text/csv")
