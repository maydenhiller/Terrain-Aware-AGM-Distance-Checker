import streamlit as st
import zipfile
import xml.etree.ElementTree as ET
import tempfile
import os
import re
import pandas as pd
from fastkml import kml
from shapely.geometry import LineString, Point
import requests
import math

# Auto-fill API Key
DEFAULT_API_KEY = "AIzaSyB9HxznAvlGb02e-K1rhld_CPeAm_wvPWU"

# Streamlit UI
st.set_page_config(layout="wide")
st.title("📏 Terrain-Aware AGM Distance Checker")

uploaded_file = st.file_uploader("Upload KML or KMZ", type=["kml", "kmz"])
st.text_input("Google Elevation API Key", value=DEFAULT_API_KEY, type="password", key="api_key")

if uploaded_file:
    try:
        # Unzip KMZ if needed
        def extract_kml(file):
            if file.name.endswith('.kmz'):
                with zipfile.ZipFile(file, 'r') as z:
                    for name in z.namelist():
                        if name.endswith('.kml'):
                            return z.read(name).decode('utf-8')
            else:
                return file.read().decode('utf-8')

        raw_kml = extract_kml(uploaded_file)

        # Parse KML
        k = kml.KML()
        k.from_string(raw_kml.encode("utf-8"))

        def find_folder_by_name(obj, name):
            for feature in obj.features():
                if feature.name and feature.name.strip().upper() == name:
                    return feature
                elif hasattr(feature, 'features'):
                    found = find_folder_by_name(feature, name)
                    if found:
                        return found
            return None

        centerline_folder = find_folder_by_name(k, "CENTERLINE")
        agms_folder = find_folder_by_name(k, "AGMs")

        if not centerline_folder or not agms_folder:
            st.error("❌ Error: CENTERLINE or AGMs folder not found.")
            st.stop()

        # Find red line (styleUrl may be #red, #line-ff0000, or similar)
        def is_red_line(line):
            style = getattr(line, 'styleUrl', '')
            return style and 'ff0000' in style.lower()

        centerline = None
        for feat in centerline_folder.features():
            if isinstance(feat.geometry, LineString) and is_red_line(feat):
                centerline = feat.geometry
                break

        if not centerline:
            st.error("❌ Error: No red centerline found inside the CENTERLINE folder.")
            st.stop()

        # Extract numeric AGMs
        agms = []
        for feat in agms_folder.features():
            if re.fullmatch(r"\d+", feat.name):
                agms.append((int(feat.name), feat.geometry))

        agms.sort()
        if len(agms) < 2:
            st.error("❌ Error: Not enough valid AGM points.")
            st.stop()

        centerline_coords = list(centerline.coords)

        # Interpolate points along centerline
        def closest_point_on_line(point, line_coords):
            min_dist = float('inf')
            closest_proj = None
            for i in range(len(line_coords) - 1):
                seg_start = line_coords[i]
                seg_end = line_coords[i + 1]
                px, py = project_point_onto_segment(point, seg_start, seg_end)
                dist = distance_3d(point, (px, py, point[2]))
                if dist < min_dist:
                    min_dist = dist
                    closest_proj = (i, (px, py))
            return closest_proj

        def project_point_onto_segment(p, a, b):
            ax, ay = a[0], a[1]
            bx, by = b[0], b[1]
            px, py = p.x, p.y
            dx, dy = bx - ax, by - ay
            if dx == dy == 0:
                return ax, ay
            t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
            t = max(0, min(1, t))
            return ax + t * dx, ay + t * dy

        def distance_3d(a, b):
            lat1, lon1, ele1 = a
            lat2, lon2, ele2 = b
            dx = (lon2 - lon1) * 111320 * math.cos(math.radians((lat1 + lat2) / 2))
            dy = (lat2 - lat1) * 110540
            dz = ele2 - ele1
            return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # Get elevation for centerline points
        api_key = st.session_state.api_key

        def get_elevations(coords):
            locs = [f"{lat},{lon}" for lon, lat, *_ in coords]
            url = (
                f"https://maps.googleapis.com/maps/api/elevation/json?locations={'|'.join(locs)}"
                f"&key={api_key}"
            )
            r = requests.get(url)
            return [result['elevation'] for result in r.json()['results']]

        elevations = get_elevations(centerline_coords)
        elevated_path = [(*pt, ele) for pt, ele in zip(centerline_coords, elevations)]

        # Project AGMs onto centerline and record index
        projected_agms = []
        for idx, pt in agms:
            lon, lat, *_ = pt.coords[0]
            proj_idx, (px, py) = closest_point_on_line(Point(lon, lat), centerline_coords)
            ele = get_elevations([(px, py)])[0]
            projected_agms.append((idx, proj_idx, (py, px, ele)))

        projected_agms.sort()

        # Calculate distances
        results = []
        total_distance_ft = 0

        for i in range(len(projected_agms) - 1):
            name_from, idx_from, pt_from = projected_agms[i]
            name_to, idx_to, pt_to = projected_agms[i + 1]

            segment_path = elevated_path[min(idx_from, idx_to):max(idx_from, idx_to) + 1]
            segment_dist = sum(
                distance_3d(segment_path[j], segment_path[j + 1])
                for j in range(len(segment_path) - 1)
            )
            total_distance_ft += segment_dist * 3.28084
            results.append({
                "From": f"{name_from:03}",
                "To": f"{name_to:03}",
                "Segment Distance (ft)": round(segment_dist * 3.28084, 2),
                "Segment Distance (mi)": round((segment_dist * 3.28084) / 5280, 4),
                "Cumulative Distance (mi)": round(total_distance_ft / 5280, 4),
            })

        df = pd.DataFrame(results)
        st.success("✅ Distance calculation complete!")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="AGM_Distances.csv", mime='text/csv')

    except Exception as e:
        st.error(f"❌ Failed to parse KMZ/KML: {e}")
