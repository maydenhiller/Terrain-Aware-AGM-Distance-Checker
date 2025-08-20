import streamlit as st
from fastkml import kml
from shapely.geometry import LineString, Point
import zipfile, io, requests
import xml.etree.ElementTree as ET
import math
import pyproj

# Elevation API (replace with preferred multi-source logic)
def get_elevation(lat, lon):
    # Stub: Replace with real elevation retrieval
    return 0.0

def terrain_distance(p1, p2):
    # p1, p2 = (lon, lat)
    elev1 = get_elevation(p1[1], p1[0])
    elev2 = get_elevation(p2[1], p2[0])
    geod = pyproj.Geod(ellps="WGS84")
    horiz_dist = geod.inv(p1[0], p1[1], p2[0], p2[1])[2]
    vert_diff = elev2 - elev1
    return math.sqrt(horiz_dist**2 + vert_diff**2)

def extract_kml(file):
    if file.name.lower().endswith(".kmz"):
        with zipfile.ZipFile(file) as kmz:
            with kmz.open([n for n in kmz.namelist() if n.endswith(".kml")][0]) as kml_file:
                return kml_file.read()
    return file.read()

st.title("AGM Terrain‑Aware Distance Calculator")

uploaded = st.file_uploader("Upload KML or KMZ", type=["kml", "kmz"])
if uploaded:
    kml_bytes = extract_kml(uploaded)
    root = ET.fromstring(kml_bytes)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    # Find the red centerline coordinates
    centerline_coords = []
    for placemark in root.findall(".//kml:Folder[kml:name='CENTERLINE']/kml:Placemark", ns):
        style_url = placemark.find("kml:styleUrl", ns)
        if style_url is not None and "2_0" in style_url.text:
            coords_text = placemark.find(".//kml:coordinates", ns).text.strip()
            for coord in coords_text.split():
                lon, lat, *_ = map(float, coord.split(","))
                centerline_coords.append((lon, lat))

    # Extract AGMs
    agms = []
    for pm in root.findall(".//kml:Folder[kml:name='AGMs']/kml:Placemark", ns):
        coord_text = pm.find(".//kml:coordinates", ns).text.strip()
        lon, lat, *_ = map(float, coord_text.split(","))
        name = pm.find("kml:name", ns).text
        agms.append((name, (lon, lat)))

    # Sort AGMs by order along centerline
    def proj_on_line(pt, line):
        return line.project(Point(pt), normalized=False)
    line_geom = LineString(centerline_coords)
    agms.sort(key=lambda x: proj_on_line(x[1], line_geom))

    # Distance calculations
    results = []
    total = 0
    for i in range(len(agms) - 1):
        seg_len = 0
        seg_line = []
        # Walk centerline between AGM[i] and AGM[i+1]
        start_m = proj_on_line(agms[i][1], line_geom)
        end_m = proj_on_line(agms[i+1][1], line_geom)
        segment = LineString(line_geom.interpolate(d) for d in [start_m, end_m])
        seg_coords = list(segment.coords)
        for a, b in zip(seg_coords[:-1], seg_coords[1:]):
            seg_len += terrain_distance(a, b)
        total += seg_len
        results.append((agms[i][0], agms[i+1][0], seg_len))

    st.subheader("AGM‑to‑AGM Distances")
    for a, b, d in results:
        st.write(f"{a} → {b}: {d:.2f} m")
    st.write(f"**Total Distance:** {total:.2f} m")
