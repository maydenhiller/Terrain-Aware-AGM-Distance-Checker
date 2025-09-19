import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import requests
import xml.etree.ElementTree as ET
from shapely.geometry import Point, LineString
from pyproj import Transformer

API_KEY = "AIzaSyCd7sfheaJIbB8_J9Q9cxWb5jnv4U0K0LA"
ELEVATION_URL = "https://maps.googleapis.com/maps/api/elevation/json"
transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)

def agm_sort_key(name_geom):
    name = name_geom[0]
    base = ''.join(filter(str.isdigit, name))
    suffix = ''.join(filter(str.isalpha, name))
    return (int(base), suffix)

def parse_kml_kmz(uploaded_file):
    if uploaded_file.name.endswith(".kmz"):
        with zipfile.ZipFile(uploaded_file) as zf:
            kml_file = next((f for f in zf.namelist() if f.endswith(".kml")), None)
            with zf.open(kml_file) as f:
                kml_data = f.read()
    else:
        kml_data = uploaded_file.read()

    root = ET.fromstring(kml_data)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    agms = []
    centerline = None

    for folder in root.findall(".//kml:Folder", ns):
        name = folder.find("kml:name", ns)
        if name is None:
            continue
        folder_name = name.text.strip().lower()
        if folder_name == "agms":
            for placemark in folder.findall("kml:Placemark", ns):
                pname = placemark.find("kml:name", ns)
                coords = placemark.find(".//kml:coordinates", ns)
                if pname is not None and coords is not None:
                    try:
                        lon, lat, *_ = map(float, coords.text.strip().split(","))
                        agms.append((pname.text.strip(), Point(lon, lat)))
                    except:
                        continue
        elif folder_name == "centerline":
            for placemark in folder.findall("kml:Placemark", ns):
                coords = placemark.find(".//kml:coordinates", ns)
                if coords is not None:
                    try:
                        pts = []
                        for pair in coords.text.strip().split():
                            lon, lat, *_ = map(float, pair.split(","))
                            pts.append((lon, lat))
                        centerline = LineString(pts)
                    except:
                        continue

    agms.sort(key=agm_sort_key)
    return agms, centerline

def slice_centerline(centerline, p1, p2):
    coords = list(centerline.coords)
    idx1 = min(range(len(coords)), key=lambda i: Point(coords[i]).distance(p1))
    idx2 = min(range(len(coords)), key=lambda i: Point(coords[i]).distance(p2))
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    segment_coords = coords[idx1:idx2+1]
    segment_coords = [c for c in segment_coords if isinstance(c, tuple) and len(c) == 2 and all(np.isfinite(c))]
    if len(segment_coords) < 2:
        return None
    return LineString(segment_coords)

def interpolate_line(line, spacing_m=1.0):
    coords = list(line.coords)
    points = [Point(coords[0])]
    for i in range(1, len(coords)):
        seg = LineString([coords[i-1], coords[i]])
        dist = seg.length
        steps = max(int(dist / spacing_m), 1)
        for j in range(1, steps + 1):
            points.append(seg.interpolate(j * spacing_m))
    return points

def get_elevations(points):
    elevations = []
    batch_size = 512
    for i in range(0, len(points), batch_size):
        chunk = points[i:i+batch_size]
        locs = "|".join([f"{p.y},{p.x}" for p in chunk])
        response = requests.get(ELEVATION_URL, params={"locations": locs, "key": API_KEY})
        data = response.json()
        elevations.extend([r["elevation"] for r in data["results"]])
    return elevations

def distance_3d(p1, p2, e1, e2):
    x1, y1 = transformer.transform(p1.x, p1.y)
    x2, y2 = transformer.transform(p2.x, p2.y)
    dz = e2 - e1
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + dz**2)

st.title("Terrain-Aware AGM Distance Calculator")

uploaded_file = st.file_uploader("Upload KML or KMZ file", type=["kml", "kmz"])
if uploaded_file:
    agms, centerline = parse_kml_kmz(uploaded_file)

    st.subheader("📌 AGM Summary")
    st.text(f"Total AGMs found: {len(agms)}")
    st.subheader("📈 CENTERLINE Status")
    st.text("CENTERLINE found" if centerline else "CENTERLINE missing")

    if not centerline or len(agms) < 2:
        st.warning("Missing CENTERLINE or insufficient AGM points.")
    else:
        rows = []
        cumulative_miles = 0.0
        skipped = 0

        for i in range(len(agms) - 1):
            name1, pt1 = agms[i]
            name2, pt2 = agms[i + 1]
            segment = slice_centerline(centerline, pt1, pt2)
            if segment is None or len(segment.coords) < 2:
                skipped += 1
                continue
            interp_points = interpolate_line(segment, spacing_m=1.0)
            elevations = get_elevations(interp_points)

            dist_m = sum(distance_3d(interp_points[j], interp_points[j+1],
                                     elevations[j], elevations[j+1])
                         for j in range(len(interp_points)-1))
            dist_ft = dist_m * 3.28084
            dist_mi = dist_ft / 5280
            cumulative_miles += dist_mi

            rows.append({
                "From AGM": name1,
                "To AGM": name2,
                "Distance (feet)": round(dist_ft, 2),
                "Distance (miles)": round(dist_mi, 6),
                "Cumulative Distance (miles)": round(cumulative_miles, 6)
            })

        st.subheader("📊 Distance Table")
        df = pd.DataFrame(rows)
        st.dataframe(df)
        st.text(f"Skipped segments: {skipped}")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "terrain_distances.csv", "text/csv")
