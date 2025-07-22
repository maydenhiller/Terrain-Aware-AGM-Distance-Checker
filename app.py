import streamlit as st
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
import folium
from folium import PolyLine, Marker
from streamlit_folium import st_folium
import io
import math

st.set_page_config(layout="wide")
st.title("📏 Terrain-Aware AGM Distance Calculator")

def haversine_3d(lat1, lon1, ele1, lat2, lon2, ele2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    flat_dist = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    elev_diff = ele2 - ele1
    return math.sqrt(flat_dist**2 + elev_diff**2)

def parse_kml_root(root):
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    centerline_coords = None
    agm_points = []

    for folder in root.findall(".//kml:Folder", ns):
        name_tag = folder.find("kml:name", ns)
        if name_tag is None:
            continue
        folder_name = name_tag.text.strip().upper()

        if folder_name == "CENTERLINE":
            for placemark in folder.findall("kml:Placemark", ns):
                coords = placemark.find(".//kml:LineString/kml:coordinates", ns)
                if coords is not None:
                    raw = coords.text.strip().split()
                    centerline_coords = [tuple(map(float, pt.split(',')[:3])) for pt in raw]
                    break

        elif folder_name == "AGMS":
            for placemark in folder.findall("kml:Placemark", ns):
                pname = placemark.find("kml:name", ns)
                if pname is None or not pname.text.strip().isdigit():
                    continue
                coords = placemark.find(".//kml:Point/kml:coordinates", ns)
                if coords is not None:
                    lon, lat, ele = map(float, coords.text.strip().split(',')[:3])
                    agm_points.append((pname.text.strip(), lat, lon, ele))

    return centerline_coords, sorted(agm_points, key=lambda x: int(x[0]))

def extract_kml_from_upload(uploaded_file):
    if uploaded_file.name.endswith(".kmz"):
        with zipfile.ZipFile(uploaded_file) as z:
            for name in z.namelist():
                if name.endswith(".kml"):
                    with z.open(name) as f:
                        return ET.parse(f).getroot()
    elif uploaded_file.name.endswith(".kml"):
        return ET.parse(uploaded_file).getroot()
    return None

uploaded_file = st.file_uploader("Upload KMZ or KML file", type=["kmz", "kml"])

if uploaded_file:
    try:
        root = extract_kml_from_upload(uploaded_file)
        if root is None:
            st.error("❌ Failed to read KMZ/KML file.")
        else:
            centerline, agms = parse_kml_root(root)

            if not centerline or len(agms) < 2:
                st.warning("⚠️ CENTERLINE or AGMs not found. Make sure they’re in folders named 'CENTERLINE' and 'AGMs'.")
            else:
                def nearest_on_path(lat, lon):
                    return min(centerline, key=lambda pt: math.hypot(pt[1] - lat, pt[0] - lon))

                ordered_agm_coords = []
                for agm in agms:
                    name, lat, lon, ele = agm
                    nearest = nearest_on_path(lat, lon)
                    ordered_agm_coords.append((name, nearest[1], nearest[0], nearest[2]))

                results = []
                total = 0
                for i in range(1, len(ordered_agm_coords)):
                    n1, lat1, lon1, ele1 = ordered_agm_coords[i - 1]
                    n2, lat2, lon2, ele2 = ordered_agm_coords[i]
                    dist_m = haversine_3d(lat1, lon1, ele1, lat2, lon2, ele2)
                    dist_ft = dist_m * 3.28084
                    dist_mi = dist_m * 0.000621371
                    total += dist_m
                    results.append({
                        "From": n1,
                        "To": n2,
                        "Segment (ft)": round(dist_ft, 2),
                        "Segment (mi)": round(dist_mi, 4),
                        "Cumulative (mi)": round(total * 0.000621371, 4)
                    })

                df = pd.DataFrame(results)
                st.dataframe(df)

                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download CSV", data=csv, file_name="terrain_distances.csv", mime="text/csv")

                m = folium.Map(location=[ordered_agm_coords[0][1], ordered_agm_coords[0][2]], zoom_start=13)
                PolyLine([(lat, lon) for _, lat, lon, _ in ordered_agm_coords], color="blue").add_to(m)
                for name, lat, lon, _ in ordered_agm_coords:
                    Marker([lat, lon], popup=name).add_to(m)
                st_folium(m, width=700, height=500)

    except Exception as e:
        st.error(f"❌ Failed to parse KMZ/KML: {str(e)}")
