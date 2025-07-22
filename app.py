import streamlit as st
import zipfile
import xml.etree.ElementTree as ET
import tempfile
import os
import math
import requests
from io import BytesIO

API_KEY = "AIzaSyB9HxznAvlGb02e-K1rhld_CPeAm_wvPWU"
EARTH_RADIUS_FT = 20925524.9  # Earth's radius in feet

def extract_kml_from_kmz_or_kml(uploaded_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        if uploaded_file.name.endswith('.kmz'):
            with zipfile.ZipFile(uploaded_file, 'r') as zf:
                zf.extractall(tmpdir)
                for name in zf.namelist():
                    if name.endswith('.kml'):
                        return os.path.join(tmpdir, name)
        else:
            path = os.path.join(tmpdir, 'temp.kml')
            with open(path, 'wb') as f:
                f.write(uploaded_file.read())
            return path

def recursive_find_folders_by_name(element, target_name, ns):
    found = []
    if element.tag.endswith("Folder"):
        name_elem = element.find("kml:name", ns)
        if name_elem is not None and name_elem.text.strip() == target_name:
            found.append(element)
    for child in element:
        found.extend(recursive_find_folders_by_name(child, target_name, ns))
    return found

def extract_agms(folder, ns):
    agms = []
    for placemark in folder.findall(".//kml:Placemark", ns):
        name_elem = placemark.find("kml:name", ns)
        coord_elem = placemark.find(".//kml:Point/kml:coordinates", ns)
        if name_elem is not None and coord_elem is not None:
            name = name_elem.text.strip()
            if name.isnumeric():
                lon, lat, *_ = map(float, coord_elem.text.strip().split(','))
                agms.append((name, lat, lon))
    return sorted(agms, key=lambda x: int(x[0]))

def extract_red_centerline(folder, ns):
    for placemark in folder.findall(".//kml:Placemark", ns):
        style = placemark.find("kml:styleUrl", ns)
        if style is not None and 'red' in style.text.lower():
            coords_elem = placemark.find(".//kml:LineString/kml:coordinates", ns)
            if coords_elem is not None:
                coords = []
                for line in coords_elem.text.strip().split():
                    lon, lat, *_ = map(float, line.split(','))
                    coords.append((lat, lon))
                return coords
    return None

def get_elevation(lat, lon):
    url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat},{lon}&key={API_KEY}"
    resp = requests.get(url).json()
    return resp['results'][0]['elevation'] if 'results' in resp and resp['results'] else 0

def distance_3d(p1, p2):
    lat1, lon1, ele1 = p1
    lat2, lon2, ele2 = p2
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    horiz = EARTH_RADIUS_FT * c
    elev_diff = ele2 - ele1
    return math.sqrt(horiz**2 + elev_diff**2)

def find_nearest(point, path):
    return min(path, key=lambda p: (p[0] - point[0])**2 + (p[1] - point[1])**2)

def main():
    st.title("📏 Terrain-Aware AGM Distance Checker")
    uploaded = st.file_uploader("Upload KML or KMZ file", type=['kml', 'kmz'])
    st.text_input("Google Elevation API Key", value=API_KEY, type="password", disabled=True)

    if uploaded:
        try:
            kml_path = extract_kml_from_kmz_or_kml(uploaded)
            tree = ET.parse(kml_path)
            root = tree.getroot()
            ns = {'kml': 'http://www.opengis.net/kml/2.2'}

            agm_folders = recursive_find_folders_by_name(root, 'AGMs', ns)
            cl_folders = recursive_find_folders_by_name(root, 'CENTERLINE', ns)

            if not agm_folders or not cl_folders:
                st.error("❌ AGMs or CENTERLINE folder not found.")
                return

            agms = extract_agms(agm_folders[0], ns)
            centerline = extract_red_centerline(cl_folders[0], ns)
            if not centerline:
                st.error("❌ Red centerline not found.")
                return

            # Fetch elevations
            st.info("Fetching elevation data. Please wait...")
            cl_with_elev = [(lat, lon, get_elevation(lat, lon)) for lat, lon in centerline]
            agm_with_elev = [(name, lat, lon, get_elevation(lat, lon)) for name, lat, lon in agms]

            # Calculate distances
            results = []
            total_ft = 0
            for i in range(len(agm_with_elev) - 1):
                name1, lat1, lon1, ele1 = agm_with_elev[i]
                name2, lat2, lon2, ele2 = agm_with_elev[i+1]

                nearest1 = find_nearest((lat1, lon1), [(lat, lon) for lat, lon, *_ in cl_with_elev])
                nearest2 = find_nearest((lat2, lon2), [(lat, lon) for lat, lon, *_ in cl_with_elev])

                try:
                    idx1 = centerline.index(nearest1)
                    idx2 = centerline.index(nearest2)
                except ValueError:
                    results.append((name1, name2, 0, 0))
                    continue

                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1

                segment = cl_with_elev[idx1:idx2+1]
                dist = sum(distance_3d(segment[j], segment[j+1]) for j in range(len(segment) - 1))
                total_ft += dist
                results.append((name1, name2, round(dist, 2), round(total_ft, 2)))

            # Display
            st.success("✅ Distances calculated successfully.")
            st.dataframe([
                {
                    "From": r[0],
                    "To": r[1],
                    "Segment (ft)": r[2],
                    "Segment (mi)": round(r[2] / 5280, 3),
                    "Cumulative (mi)": round(r[3] / 5280, 3)
                } for r in results
            ])
        except Exception as e:
            st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
