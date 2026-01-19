import streamlit as st
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
from math import radians, sin, cos, sqrt, atan2

st.set_page_config(layout="wide")

# ------------------------
# Geometry utilities
# ------------------------

def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.7613  # Earth radius in miles
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def polyline_length(coords):
    dist = 0.0
    for i in range(len(coords) - 1):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[i + 1]
        dist += haversine_miles(lat1, lon1, lat2, lon2)
    return dist


# ------------------------
# KML / KMZ Parsing
# ------------------------

KML_NS = {
    "kml": "http://www.opengis.net/kml/2.2",
    "gx": "http://www.google.com/kml/ext/2.2"
}


def extract_kml(uploaded_file):
    if uploaded_file.name.lower().endswith(".kmz"):
        with zipfile.ZipFile(uploaded_file) as z:
            for name in z.namelist():
                if name.endswith(".kml"):
                    return z.read(name).decode("utf-8")
        raise ValueError("KMZ contains no KML")
    else:
        return uploaded_file.read().decode("utf-8")


def parse_kml_kmz(uploaded_file):
    text = extract_kml(uploaded_file)

    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        st.error(f"Failed to parse XML: {e}")
        return [], []

    agms = []
    centerlines = []

    # Find all Folders (namespace safe)
    for folder in root.findall(".//kml:Folder", KML_NS):
        name_el = folder.find("kml:name", KML_NS)
        if name_el is None:
            continue

        folder_name = name_el.text.strip().upper()

        # ------------------------
        # AGMs
        # ------------------------
        if folder_name == "AGMS":
            for pm in folder.findall(".//kml:Placemark", KML_NS):
                name = pm.findtext("kml:name", default="AGM", namespaces=KML_NS)
                coord_el = pm.find(".//kml:Point/kml:coordinates", KML_NS)
                if coord_el is None:
                    continue
                lon, lat, *_ = map(float, coord_el.text.strip().split(","))
                agms.append({
                    "name": name,
                    "lat": lat,
                    "lon": lon
                })

        # ------------------------
        # CENTERLINE
        # ------------------------
        elif folder_name == "CENTERLINE":
            for pm in folder.findall(".//kml:Placemark", KML_NS):

                coords = []

                # 1️⃣ gx:Track (MOST ACCURATE)
                track = pm.find(".//gx:Track", KML_NS)
                if track is not None:
                    for gxcoord in track.findall("gx:coord", KML_NS):
                        lon, lat, *_ = map(float, gxcoord.text.strip().split())
                        coords.append((lat, lon))

                # 2️⃣ Fallback LineString
                if not coords:
                    coord_el = pm.find(".//kml:LineString/kml:coordinates", KML_NS)
                    if coord_el is not None:
                        for c in coord_el.text.strip().split():
                            lon, lat, *_ = map(float, c.split(","))
                            coords.append((lat, lon))

                if coords:
                    centerlines.append(coords)

    return agms, centerlines


# ------------------------
# Streamlit UI
# ------------------------

st.title("Terrain-Aware AGM Distance Checker")

uploaded = st.file_uploader(
    "Drag and drop file here",
    type=["kml", "kmz"],
    accept_multiple_files=False
)

if uploaded:
    agms, centerlines = parse_kml_kmz(uploaded)

    st.write(f"**{len(agms)} AGMs | {len(centerlines)} centerline part(s)**")

    if not agms or not centerlines:
        st.error("Need both AGMs and CENTERLINE.")
        st.stop()

    centerline_coords = centerlines[0]
    total_miles = polyline_length(centerline_coords)

    st.success(f"Centerline length: **{total_miles:.3f} miles**")

    st.subheader("AGMs")
    st.dataframe(agms, use_container_width=True)
