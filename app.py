import streamlit as st
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os
import tempfile
import math
from fastkml import kml
from shapely.geometry import LineString, Point
from pykml import parser as pykml_parser
from xml.dom import minidom
import requests

def parse_coords(coord_string):
    coords = []
    for line in coord_string.strip().split():
        parts = line.strip().split(',')
        if len(parts) >= 2:
            lon, lat = float(parts[0]), float(parts[1])
            ele = float(parts[2]) if len(parts) == 3 else 0
            coords.append((lon, lat, ele))
    return coords

def extract_centerline_and_agms(file_path):
    with open(file_path, 'rb') as f:
        tree = ET.parse(f)
        root = tree.getroot()

    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    def find_folder(root, name):
        for folder in root.findall('.//kml:Folder', ns):
            folder_name = folder.find('kml:name', ns)
            if folder_name is not None and folder_name.text and folder_name.text.strip().upper() == name:
                return folder
        return None

    def extract_line(folder):
        for placemark in folder.findall('.//kml:Placemark', ns):
            color = placemark.find('.//kml:color', ns)
            if color is not None and color.text.strip().lower() == 'ff0000ff':
                coords_el = placemark.find('.//kml:coordinates', ns)
                if coords_el is not None:
                    coords = parse_coords(coords_el.text)
                    return LineString(coords)
        return None

    def extract_agms(folder):
        agms = []
        for placemark in folder.findall('.//kml:Placemark', ns):
            name_el = placemark.find('kml:name', ns)
            coords_el = placemark.find('.//kml:coordinates', ns)
            if name_el is not None and coords_el is not None:
                name = name_el.text.strip()
                if name.isnumeric():
                    lon, lat = map(float, coords_el.text.strip().split(',')[:2])
                    agms.append((name, Point(lon, lat)))
        agms.sort(key=lambda x: int(x[0]))
        return agms

    centerline_folder = find_folder(root, 'CENTERLINE')
    agms_folder = find_folder(root, 'AGMs')

    if not centerline_folder or not agms_folder:
        raise ValueError('CENTERLINE or AGMs folder not found.')

    line = extract_line(centerline_folder)
    agms = extract_agms(agms_folder)

    if not line or not agms:
        raise ValueError('Failed to extract centerline or AGMs.')

    return line, agms

def get_elevation(lat, lon, api_key):
    url = f'https://maps.googleapis.com/maps/api/elevation/json?locations={lat},{lon}&key={api_key}'
    r = requests.get(url)
    results = r.json().get('results')
    if results and 'elevation' in results[0]:
        return results[0]['elevation']
    return 0

def terrain_distance_3d(p1, p2, api_key):
    elev1 = get_elevation(p1.y, p1.x, api_key)
    elev2 = get_elevation(p2.y, p2.x, api_key)
    d2d = p1.distance(p2) * 111139  # Convert degrees to meters roughly
    elev_diff = elev2 - elev1
    return math.sqrt(d2d**2 + elev_diff**2)

def interpolate_point_on_line(line, point):
    return line.interpolate(line.project(point))

st.set_page_config(layout="wide")
st.title("📏 Terrain-Aware AGM Distance Checker")

uploaded_file = st.file_uploader("Upload KML or KMZ", type=['kml', 'kmz'])
api_key = st.text_input("Google Elevation API Key", type="password")

if uploaded_file and api_key:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, uploaded_file.name)
            with open(path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            if path.endswith('.kmz'):
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                    for fname in zip_ref.namelist():
                        if fname.endswith('.kml'):
                            kml_path = os.path.join(tmpdir, fname)
                            break
            else:
                kml_path = path

            centerline, agms = extract_centerline_and_agms(kml_path)

            projected_points = [(name, interpolate_point_on_line(centerline, pt)) for name, pt in agms]

            data = []
            total_meters = 0
            for i in range(1, len(projected_points)):
                prev_name, prev_pt = projected_points[i - 1]
                curr_name, curr_pt = projected_points[i]
                seg_meters = terrain_distance_3d(prev_pt, curr_pt, api_key)
                total_meters += seg_meters
                data.append({
                    'From': prev_name,
                    'To': curr_name,
                    'Segment Distance (ft)': seg_meters * 3.28084,
                    'Segment Distance (mi)': seg_meters * 0.000621371,
                    'Cumulative Distance (mi)': total_meters * 0.000621371
                })

            df = pd.DataFrame(data)
            st.dataframe(df)

            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "terrain_distances.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
