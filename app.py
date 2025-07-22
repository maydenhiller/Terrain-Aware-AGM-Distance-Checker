import streamlit as st
import zipfile
import simplekml
import xml.etree.ElementTree as ET
import tempfile
import os
import math
import requests
from io import BytesIO

API_KEY = "AIzaSyB9HxznAvlGb02e-K1rhld_CPeAm_wvPWU"
EARTH_RADIUS_FT = 20925524.9  # Feet

def parse_kmz_or_kml(uploaded_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        if uploaded_file.name.endswith('.kmz'):
            with zipfile.ZipFile(uploaded_file, 'r') as zf:
                zf.extractall(tmpdir)
                for name in zf.namelist():
                    if name.endswith('.kml'):
                        kml_path = os.path.join(tmpdir, name)
                        break
        else:
            kml_path = os.path.join(tmpdir, 'temp.kml')
            with open(kml_path, 'wb') as f:
                f.write(uploaded_file.read())

        tree = ET.parse(kml_path)
        root = tree.getroot()
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}

        def find_folder_by_name(name):
            return next((f for f in root.iter() if f.tag.endswith('Folder') and f.find('kml:name', ns) is not None and f.find('kml:name', ns).text == name), None)

        agm_folder = find_folder_by_name('AGMs')
        centerline_folder = find_folder_by_name('CENTERLINE')

        if not agm_folder or not centerline_folder:
            raise ValueError("CENTERLINE or AGMs folder not found.")

        def extract_points(folder):
            points = []
            for pm in folder.findall(".//kml:Placemark", ns):
                name_elem = pm.find("kml:name", ns)
                point_elem = pm.find(".//kml:Point/kml:coordinates", ns)
                if name_elem is not None and point_elem is not None:
                    name = name_elem.text.strip()
                    if name.isnumeric():
                        lon, lat, *_ = map(float, point_elem.text.strip().split(','))
                        points.append((name, lat, lon))
            return sorted(points, key=lambda x: int(x[0]))

        def extract_linestring_coords(folder):
            for pm in folder.findall(".//kml:Placemark", ns):
                style = pm.find("kml:styleUrl", ns)
                if style is not None and 'red' in style.text.lower():
                    coords_elem = pm.find(".//kml:LineString/kml:coordinates", ns)
                    if coords_elem is not None:
                        coord_pairs = []
                        for line in coords_elem.text.strip().split():
                            lon, lat, *_ = map(float, line.split(','))
                            coord_pairs.append((lat, lon))
                        return coord_pairs
            return None

        agms = extract_points(agm_folder)
        centerline = extract_linestring_coords(centerline_folder)

        if not centerline:
            raise ValueError("No red centerline found inside the CENTERLINE folder.")

        return agms, centerline

def get_elevation(lat, lon):
    url = (
        f"https://maps.googleapis.com/maps/api/elevation/json"
        f"?locations={lat},{lon}&key={API_KEY}"
    )
    resp = requests.get(url).json()
    return resp['results'][0]['elevation'] if 'results' in resp and resp['results'] else 0

def distance_3d(p1, p2):
    lat1, lon1, ele1 = p1
    lat2, lon2, ele2 = p2

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    horiz_dist = EARTH_RADIUS_FT * c
    elev_diff = ele2 - ele1
    return math.sqrt(horiz_dist**2 + elev_diff**2)

def find_closest_point(point, path):
    min_dist = float('inf')
    closest = None
    for lat, lon in path:
        d = math.sqrt((lat - point[0])**2 + (lon - point[1])**2)
        if d < min_dist:
            min_dist = d
            closest = (lat, lon)
    return closest

def main():
    st.title("📏 Terrain-Aware AGM Distance Checker")
    uploaded = st.file_uploader("Upload KML or KMZ", type=["kml", "kmz"])
    st.text_input("Google Elevation API Key", value=API_KEY, type="password", disabled=True)

    if uploaded:
        try:
            agms, centerline = parse_kmz_or_kml(uploaded)

            # Get elevations
            st.info("Fetching elevation data (this may take a few seconds)...")
            centerline_elev = [(lat, lon, get_elevation(lat, lon)) for lat, lon in centerline]
            agm_coords = []
            for name, lat, lon in agms:
                elev = get_elevation(lat, lon)
                agm_coords.append((name, lat, lon, elev))

            results = []
            total_dist_ft = 0

            for i in range(len(agm_coords) - 1):
                name1, lat1, lon1, ele1 = agm_coords[i]
                name2, lat2, lon2, ele2 = agm_coords[i + 1]

                # Projected points
                p1 = find_closest_point((lat1, lon1), [(lat, lon) for lat, lon, *_ in centerline_elev])
                p2 = find_closest_point((lat2, lon2), [(lat, lon) for lat, lon, *_ in centerline_elev])

                # Trace centerline between the projections
                try:
                    start_idx = centerline.index(p1)
                    end_idx = centerline.index(p2)
                except ValueError:
                    results.append((name1, name2, 0, 0))
                    continue

                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx

                segment = centerline_elev[start_idx:end_idx+1]
                seg_dist = sum(distance_3d(segment[i], segment[i+1]) for i in range(len(segment) - 1))
                total_dist_ft += seg_dist
                results.append((name1, name2, round(seg_dist, 2), round(total_dist_ft, 2)))

            st.success("✅ Distances calculated successfully.")

            # Display table
            st.dataframe(
                [{
                    "From": f,
                    "To": t,
                    "Segment (ft)": seg,
                    "Segment (mi)": round(seg / 5280, 3),
                    "Cumulative (mi)": round(cum / 5280, 3)
                } for f, t, seg, cum in results]
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
