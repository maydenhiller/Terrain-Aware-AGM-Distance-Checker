import streamlit as st
import zipfile
import io
import xml.etree.ElementTree as ET
from shapely.geometry import LineString, Point
import math

st.set_page_config(page_title="Terrain Distance Checker", layout="centered")
st.title("📏 Terrain-Aware AGM Distance Checker")

uploaded_file = st.file_uploader("Upload a KMZ or KML file with a red centerline and numbered AGMs", type=["kmz", "kml"])

def extract_kml_from_kmz(kmz_file):
    with zipfile.ZipFile(kmz_file, 'r') as z:
        for filename in z.namelist():
            if filename.endswith('.kml'):
                with z.open(filename) as kmlfile:
                    return kmlfile.read()
    return None

def parse_kml(kml_bytes):
    try:
        if isinstance(kml_bytes, bytes):
            root = ET.fromstring(kml_bytes)
        else:
            root = ET.fromstring(kml_bytes.encode("utf-8"))

        ns = {'kml': 'http://www.opengis.net/kml/2.2'}

        red_styles = set()
        for style in root.findall('.//kml:Style', ns):
            line_color = style.find('.//kml:LineStyle/kml:color', ns)
            if line_color is not None and line_color.text.strip().lower() == 'ff0000ff':  # ABGR red
                red_styles.add(style.attrib.get('id'))

        line = None
        agm_points = []

        for placemark in root.findall('.//kml:Placemark', ns):
            name = placemark.find('kml:name', ns)
            line_str = placemark.find('.//kml:LineString', ns)
            point = placemark.find('.//kml:Point', ns)

            if line_str is not None:
                # Check if red line
                style_url = placemark.find('kml:styleUrl', ns)
                if style_url is not None and style_url.text.startswith('#'):
                    style_id = style_url.text[1:]
                    if style_id in red_styles:
                        coords_text = line_str.find('kml:coordinates', ns).text.strip()
                        coords = [tuple(map(float, c.strip().split(',')[:2])) for c in coords_text.split()]
                        line = LineString(coords)

            elif point is not None and name is not None:
                if name.text.isdigit():
                    coords_text = point.find('kml:coordinates', ns).text.strip()
                    lon, lat = map(float, coords_text.split(',')[:2])
                    agm_points.append((name.text, Point(lon, lat)))

        if not line or not agm_points:
            raise ValueError("Centerline or AGMs not found.")

        agm_points.sort(key=lambda x: int(x[0]))
        return line, agm_points

    except Exception as e:
        raise ValueError(f"Failed to parse KMZ/KML: {e}")

def haversine(p1, p2):
    lon1, lat1 = p1.x, p1.y
    lon2, lat2 = p2.x, p2.y
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".kmz"):
            kml_data = extract_kml_from_kmz(uploaded_file)
        else:
            kml_data = uploaded_file.read()

        centerline, agms = parse_kml(kml_data)

        st.success("✅ Centerline and AGMs loaded successfully.")

        rows = []
        total = 0
        for i in range(1, len(agms)):
            prev_name, prev_point = agms[i - 1]
            curr_name, curr_point = agms[i]
            dist = haversine(prev_point, curr_point)
            total += dist
            rows.append({
                "From": prev_name,
                "To": curr_name,
                "Segment (ft)": round(dist * 3.28084, 1),
                "Cumulative (mi)": round(total * 0.000621371, 3)
            })

        st.write("### 📄 Distance Table")
        st.dataframe(rows)

        csv = "From,To,Segment (ft),Cumulative (mi)\n" + "\n".join(
            f"{r['From']},{r['To']},{r['Segment (ft)']},{r['Cumulative (mi)']}" for r in rows
        )

        st.download_button("📥 Download CSV", csv.encode(), "AGM_Distances.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ {str(e)}")
