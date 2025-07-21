import streamlit as st
from fastkml import kml
from shapely.geometry import LineString, Point
import zipfile
import io
import re
import pandas as pd
from xml.etree import ElementTree as ET

st.set_page_config(page_title="Terrain-Aware AGM Distance Checker", layout="wide")
st.title("📏 Terrain-Aware AGM Distance Checker")
st.write("Upload a `.kmz` or `.kml` file with a red centerline (from `CENTERLINE` folder) and numbered AGM placemarks (in `AGMs` folder).")

def extract_kml_from_kmz(file) -> str:
    with zipfile.ZipFile(file, 'r') as zf:
        for name in zf.namelist():
            if name.endswith('.kml'):
                return zf.read(name).decode('utf-8')
    return None

def parse_style_map(kml_root):
    style_map = {}
    for style in kml_root.findall(".//{http://www.opengis.net/kml/2.2}Style"):
        style_id = style.attrib.get('id')
        line_style = style.find('{http://www.opengis.net/kml/2.2}LineStyle')
        if style_id and line_style is not None:
            color = line_style.find('{http://www.opengis.net/kml/2.2}color')
            if color is not None:
                style_map[f"#{style_id}"] = color.text
    return style_map

def extract_geometry_from_folder(folder_elem, tag='Placemark'):
    return folder_elem.findall(f".//{{http://www.opengis.net/kml/2.2}}{tag}")

def parse_kml_file(kml_str):
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    root = ET.fromstring(kml_str.encode('utf-8'))
    document = root.find('kml:Document', ns)
    if document is None:
        raise ValueError("KML Document not found.")

    # Parse styles
    style_map = parse_style_map(document)

    centerline = None
    agms = []

    for folder in document.findall("kml:Folder", ns):
        folder_name = folder.find("kml:name", ns)
        if folder_name is not None and folder_name.text == "CENTERLINE":
            for placemark in extract_geometry_from_folder(folder):
                line = placemark.find(".//kml:LineString", ns)
                style_url = placemark.find("kml:styleUrl", ns)
                if line is not None and style_url is not None:
                    style_color = style_map.get(style_url.text, "")
                    if style_color.lower() == "ffff0000":  # red
                        coords_text = line.find("kml:coordinates", ns).text.strip()
                        coords = [(float(x), float(y)) for x, y, *_ in [c.split(",") for c in coords_text.split()]]
                        centerline = LineString(coords)
        elif folder_name is not None and folder_name.text == "AGMs":
            for placemark in extract_geometry_from_folder(folder):
                name_elem = placemark.find("kml:name", ns)
                point_elem = placemark.find(".//kml:Point", ns)
                if name_elem is not None and re.fullmatch(r"\d+", name_elem.text) and point_elem is not None:
                    coord_text = point_elem.find("kml:coordinates", ns).text.strip()
                    x, y, *_ = coord_text.split(",")
                    agms.append((name_elem.text, Point(float(x), float(y))))

    if centerline is None:
        raise ValueError("No red centerline found.")
    if not agms:
        raise ValueError("No valid AGMs found.")
    agms.sort(key=lambda a: a[0])  # Sort by name like "000", "010", etc.
    return centerline, agms

def project_points_to_line(line, points):
    projected = []
    for label, pt in points:
        dist = line.project(pt)
        projected.append((dist, label, pt))
    projected.sort()
    return [(label, pt) for _, label, pt in projected]

def compute_distances(centerline, agms):
    projected = project_points_to_line(centerline, agms)
    distances = []
    cumulative = 0.0

    for i in range(len(projected) - 1):
        pt1 = projected[i][1]
        pt2 = projected[i + 1][1]
        seg_line = centerline.interpolate(centerline.project(pt1)), centerline.interpolate(centerline.project(pt2))
        segment_distance = seg_line[0].distance(seg_line[1]) * 364000  # rough conversion degrees to feet
        cumulative += segment_distance
        distances.append({
            "From": projected[i][0],
            "To": projected[i + 1][0],
            "Segment_Distance_ft": round(segment_distance, 2),
            "Cumulative_Distance_ft": round(cumulative, 2),
            "Cumulative_Distance_mi": round(cumulative / 5280, 4)
        })
    return pd.DataFrame(distances)

uploaded_file = st.file_uploader("Upload KMZ or KML", type=["kmz", "kml"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".kmz"):
            kml_str = extract_kml_from_kmz(uploaded_file)
        else:
            kml_str = uploaded_file.read().decode("utf-8")

        centerline, agms = parse_kml_file(kml_str)
        df = compute_distances(centerline, agms)
        st.success("✅ Distance calculation complete.")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv, file_name="Terrain_Distances.csv", mime="text/csv")
    except Exception as e:
        st.error(f"⚠️ Failed to parse KMZ/KML: {e}")
