import streamlit as st
from fastkml import kml
from shapely.geometry import LineString, Point
import zipfile
import xml.etree.ElementTree as ET
import math
import requests
import folium
from streamlit_folium import st_folium
import io
import base64
import csv

st.set_page_config(page_title="Terrain-Aware AGM Distance Checker", layout="wide")
st.title("🗺️ Terrain-Aware AGM Distance Checker")
st.write("Upload a KMZ or KML file with a **red centerline** (`#ff0000`) and **numbered AGMs**.")

# ─────────────────────────────────────────────────────────────────────────────

def extract_kml_content(uploaded_file):
    if uploaded_file.name.endswith('.kmz'):
        with zipfile.ZipFile(uploaded_file, 'r') as zf:
            for name in zf.namelist():
                if name.endswith('.kml'):
                    return zf.read(name)  # Return raw bytes
    elif uploaded_file.name.endswith('.kml'):
        return uploaded_file.read()  # Return raw bytes
    return None

# ─────────────────────────────────────────────────────────────────────────────

def parse_kml(kml_bytes):
    k = kml.KML()
    k.from_string(kml_bytes)

    centerline = None
    agms = []

    def recurse_features(features):
        nonlocal centerline, agms
        for f in features:
            if hasattr(f, 'geometry') and isinstance(f.geometry, LineString):
                style = getattr(f, 'styleUrl', '')
                if 'ff0000' in style.lower() or 'red' in style.lower():
                    centerline = f.geometry
            elif hasattr(f, 'geometry') and isinstance(f.geometry, Point):
                name = f.name.strip()
                if name.isdigit() and not name.startswith("SP"):
                    agms.append((int(name), f.geometry))
            elif hasattr(f, 'features'):
                recurse_features(f.features())

    recurse_features(k.features())
    agms.sort(key=lambda x: x[0])
    agm_points = [p for _, p in agms]
    return centerline, agm_points

# ─────────────────────────────────────────────────────────────────────────────

def haversine_2d(p1, p2):
    R = 6371000  # Earth radius in meters
    lat1, lon1 = math.radians(p1[1]), math.radians(p1[0])
    lat2, lon2 = math.radians(p2[1]), math.radians(p2[0])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def interpolate_line(line, spacing=30):
    coords = list(line.coords)
    interpolated = []
    for i in range(len(coords) - 1):
        start, end = coords[i], coords[i + 1]
        dist = haversine_2d(start, end)
        steps = max(1, int(dist / spacing))
        for j in range(steps):
            lat = start[1] + (end[1] - start[1]) * j / steps
            lon = start[0] + (end[0] - start[0]) * j / steps
            interpolated.append((lon, lat))
    interpolated.append((coords[-1][0], coords[-1][1]))
    return interpolated

def get_elevations(points):
    # Uses USGS Elevation Point Query Service (EPQS)
    base_url = "https://nationalmap.gov/epqs/pqs.php"
    elevations = []
    for lon, lat in points:
        response = requests.get(base_url, params={
            "x": lon,
            "y": lat,
            "units": "Feet",
            "output": "json"
        })
        if response.ok:
            try:
                data = response.json()
                elev = float(data['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation'])
                elevations.append(elev)
            except Exception:
                elevations.append(0)
        else:
            elevations.append(0)
    return elevations

# ─────────────────────────────────────────────────────────────────────────────

def compute_terrain_distances(centerline, agms):
    interpolated = interpolate_line(centerline)
    elevations = get_elevations(interpolated)
    dist_along = [0]
    for i in range(1, len(interpolated)):
        dx = haversine_2d(interpolated[i-1], interpolated[i])
        dz = elevations[i] - elevations[i-1]
        dist = math.sqrt(dx**2 + dz**2)
        dist_along.append(dist_along[-1] + dist)

    def nearest_index(pt):
        return min(range(len(interpolated)), key=lambda i: haversine_2d(interpolated[i], (pt.x, pt.y)))

    indices = [nearest_index(pt) for pt in agms]
    results = []
    for i in range(1, len(indices)):
        seg = dist_along[indices[i]] - dist_along[indices[i - 1]]
        cumulative = dist_along[indices[i]]
        results.append((i, seg, cumulative))
    return results

# ─────────────────────────────────────────────────────────────────────────────

def create_map(centerline, agms):
    center = agms[0].y, agms[0].x
    m = folium.Map(location=center, zoom_start=14)
    folium.PolyLine([(p[1], p[0]) for p in centerline.coords], color='red').add_to(m)
    for i, pt in enumerate(agms):
        folium.Marker(location=(pt.y, pt.x), popup=f"{i:03}").add_to(m)
    return m

# ─────────────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader("Upload KMZ or KML", type=["kmz", "kml"])
if uploaded_file:
    kml_data = extract_kml_content(uploaded_file)
    try:
        centerline, agms = parse_kml(kml_data)
        if not centerline or len(agms) < 2:
            st.error("Centerline or AGMs not found. Ensure the centerline is red (#ff0000) and points are purely numeric.")
        else:
            st.success("KMZ/KML parsed successfully.")

            with st.spinner("Computing terrain-aware distances..."):
                results = compute_terrain_distances(centerline, agms)

            st.subheader("Distance Table")
            rows = []
            for i, (seg_ft, cum_ft) in enumerate([(r[1], r[2]) for r in results], 1):
                rows.append({
                    "From": f"{(i - 1) * 10:03}",
                    "To": f"{i * 10:03}",
                    "Segment (ft)": round(seg_ft, 2),
                    "Cumulative (ft)": round(cum_ft, 2),
                    "Cumulative (mi)": round(cum_ft / 5280, 3)
                })
            st.dataframe(rows, use_container_width=True)

            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
            b64 = base64.b64encode(output.getvalue().encode()).decode()
            st.download_button("Download CSV", data=output.getvalue(), file_name="terrain_distances.csv", mime="text/csv")

            st.subheader("Map Preview")
            folium_map = create_map(centerline, agms)
            st_folium(folium_map, height=500)
    except Exception as e:
        st.error(f"Failed to parse KMZ/KML: {e}")
