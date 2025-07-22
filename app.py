import streamlit as st
import zipfile
import xml.etree.ElementTree as ET
import tempfile
import os
import math
import requests
from io import BytesIO

API_KEY = "AIzaSyB9HxznAvlGb02e-K1rhld_CPeAm_wvPWU"
EARTH_RADIUS_FT = 20925524.9

def extract_kml(uploaded_file):
    content = uploaded_file.read()
    with tempfile.TemporaryDirectory() as tmpdir:
        if uploaded_file.name.endswith(".kmz"):
            with zipfile.ZipFile(BytesIO(content), 'r') as kmz:
                kmz.extractall(tmpdir)
                for f in kmz.namelist():
                    if f.endswith(".kml"):
                        return os.path.join(tmpdir, f)
        else:
            path = os.path.join(tmpdir, "temp.kml")
            with open(path, "wb") as f:
                f.write(content)
            return path
    return None

def recursive_find_folders_by_name(elem, target_name, ns):
    found = []
    if elem.tag.endswith("Folder"):
        name_elem = elem.find("kml:name", ns)
        if name_elem is not None and name_elem.text.strip() == target_name:
            found.append(elem)
    for child in elem:
        found.extend(recursive_find_folders_by_name(child, target_name, ns))
    return found

def extract_agms(folder, ns):
    agms = []
    for placemark in folder.findall(".//kml:Placemark", ns):
        name_elem = placemark.find("kml:name", ns)
        coord_elem = placemark.find(".//kml:Point/kml:coordinates", ns)
        if name_elem is not None and coord_elem is not None:
            name = name_elem.text.strip()
            if name.isdigit():
                lon, lat, *_ = map(float, coord_elem.text.strip().split(','))
                agms.append((name, lat, lon))
    return sorted(agms, key=lambda x: int(x[0]))

def extract_red_centerline(folder, ns):
    for placemark in folder.findall(".//kml:Placemark", ns):
        style_elem = placemark.find("kml:styleUrl", ns)
        coords_elem = placemark.find(".//kml:LineString/kml:coordinates", ns)
        if coords_elem is not None and (style_elem is None or 'red' in style_elem.text.lower()):
            coords = []
            for line in coords_elem.text.strip().split():
                lon, lat, *_ = map(float, line.split(','))
                coords.append((lat, lon))
            return coords
    return None

def get_elevation(lat, lon):
    url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat},{lon}&key={API_KEY}"
    response = requests.get(url).json()
    if "results" in response and response["results"]:
        return response["results"][0]["elevation"]
    return 0

def distance_3d(p1, p2):
    lat1, lon1, ele1 = p1
    lat2, lon2, ele2 = p2
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    horiz = EARTH_RADIUS_FT * c
    elev_diff = ele2 - ele1
    return math.sqrt(horiz**2 + elev_diff**2)

def find_nearest_index(point, line_coords):
    return min(range(len(line_coords)), key=lambda i: (point[0] - line_coords[i][0])**2 + (point[1] - line_coords[i][1])**2)

def main():
    st.title("📏 Terrain-Aware AGM Distance Checker")
    uploaded = st.file_uploader("Upload KMZ or KML", type=["kmz", "kml"])
    st.text_input("Google Elevation API Key", value=API_KEY, type="password", disabled=True)

    if uploaded:
        try:
            kml_path = extract_kml(uploaded)
            if not kml_path:
                st.error("❌ Failed to extract KML from uploaded file.")
                return

            tree = ET.parse(kml_path)
            root = tree.getroot()
            ns = {'kml': 'http://www.opengis.net/kml/2.2'}

            agm_folders = recursive_find_folders_by_name(root, 'AGMs', ns)
            cl_folders = recursive_find_folders_by_name(root, 'CENTERLINE', ns)

            if not agm_folders or not cl_folders:
                st.error("❌ Error: CENTERLINE or AGMs folder not found.")
                return

            agms = extract_agms(agm_folders[0], ns)
            centerline = extract_red_centerline(cl_folders[0], ns)

            if not centerline or not agms:
                st.error("❌ No AGMs or red centerline found.")
                return

            st.info("Fetching elevation data...")
            cl_with_elev = [(lat, lon, get_elevation(lat, lon)) for lat, lon in centerline]
            agm_with_elev = [(name, lat, lon, get_elevation(lat, lon)) for name, lat, lon in agms]

            results = []
            total_ft = 0
            for i in range(len(agm_with_elev) - 1):
                name1, lat1, lon1, ele1 = agm_with_elev[i]
                name2, lat2, lon2, ele2 = agm_with_elev[i + 1]

                idx1 = find_nearest_index((lat1, lon1), centerline)
                idx2 = find_nearest_index((lat2, lon2), centerline)
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1

                segment = cl_with_elev[idx1:idx2 + 1]
                seg_dist = sum(distance_3d(segment[j], segment[j + 1]) for j in range(len(segment) - 1))
                total_ft += seg_dist
                results.append({
                    "From": name1,
                    "To": name2,
                    "Segment (ft)": round(seg_dist, 2),
                    "Segment (mi)": round(seg_dist / 5280, 3),
                    "Cumulative (mi)": round(total_ft / 5280, 3)
                })

            st.success("✅ Distances calculated!")
            st.dataframe(results)

        except Exception as e:
            st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
