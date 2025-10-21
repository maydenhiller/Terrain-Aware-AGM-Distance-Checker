import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import requests
import xml.etree.ElementTree as ET
from shapely.geometry import Point, LineString
from pyproj import Transformer

# --- CONFIG ---
MAPBOX_TOKEN = st.secrets["MAPBOX_TOKEN"]
TILEQUERY_URL = "https://api.mapbox.com/v4/mapbox.mapbox-terrain-v2/tilequery/{lon},{lat}.json"

transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)

# --- HELPERS ---
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

def project_onto_centerline(centerline, point):
    return centerline.interpolate(centerline.project(point))

def slice_centerline(centerline, p1, p2):
    d1 = centerline.project(p1)
    d2 = centerline.project(p2)
    if d1 > d2:
        d1, d2 = d2, d1
    coords = []
    for i in range(len(centerline.coords) - 1):
        seg = LineString([centerline.coords[i], centerline.coords[i+1]])
        seg_start = centerline.project(Point(centerline.coords[i]))
        seg_end = centerline.project(Point(centerline.coords[i+1]))
        if seg_end < d1 or seg_start > d2:
            continue
        seg_start = max(seg_start, d1)
        seg_end = min(seg_end, d2)
        steps = max(int(seg.length / 1.0), 1)
        for j in range(steps + 1):
            pt = seg.interpolate(j / steps, normalized=True)
            coords.append((pt.x, pt.y))
    return LineString(coords) if len(coords) >= 2 else None

def interpolate_line(line, spacing_m=1.0):
    total_length = line.length
    steps = max(int(total_length / spacing_m), 1)
    return [line.interpolate(i * spacing_m) for i in range(steps + 1)]

def get_elevations(points):
    elevations = []
    for p in points:
        url = TILEQUERY_URL.format(lon=p.x, lat=p.y)
        resp = requests.get(url, params={"layers": "contour", "limit": 1, "access_token": MAPBOX_TOKEN})
        if resp.status_code == 200:
            data = resp.json()
            if "features" in data and len(data["features"]) > 0:
                elev = data["features"][0]["properties"].get("ele")
                if elev is not None:
                    elevations.append(float(elev))
                    continue
        elevations.append(0.0)  # fallback if no data
    return elevations

def distance_3d(p1, p2, e1, e2):
    x1, y1 = transformer.transform(p1.x, p1.y)
    x2, y2 = transformer.transform(p2.x, p2.y)
    dz = e2 - e1
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + dz**2)

# --- STREAMLIT UI ---
st.title("Terrain-Aware AGM Distance Calculator (Mapbox)")

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

            if len(elevations) != len(interp_points):
                skipped += 1
                continue

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
