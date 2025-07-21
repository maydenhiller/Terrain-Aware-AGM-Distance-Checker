import streamlit as st
from fastkml import kml
from shapely.geometry import LineString, Point
import zipfile
import io
import math
import pandas as pd

st.set_page_config(layout="wide")
st.title("📏 Terrain-Aware Distance Calculator")
st.markdown("Upload a KMZ or KML file with:")
st.markdown("- A red centerline under the `CENTERLINE` folder (style: `#ff0000`)")
st.markdown("- Numbered AGMs under the `AGMs` folder")

uploaded_file = st.file_uploader("Upload KMZ or KML", type=["kmz", "kml"])

def haversine_3d(p1, p2):
    R = 6371000  # meters
    lat1, lon1, ele1 = p1
    lat2, lon2, ele2 = p2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    horiz = 2 * R * math.asin(math.sqrt(a))
    elev_diff = ele2 - ele1
    return math.sqrt(horiz**2 + elev_diff**2)

def extract_centerline_and_agms(kml_data):
    k = kml.KML()
    k.from_string(kml_data)
    centerline = None
    agms = {}

    def extract_from_features(features):
        nonlocal centerline, agms
        for f in features:
            if hasattr(f, 'features'):
                extract_from_features(f.features())
            elif hasattr(f, 'geometry'):
                if f.name and f.name.strip().isnumeric():
                    agms[int(f.name.strip())] = f.geometry
                elif isinstance(f.geometry, LineString):
                    style = getattr(f, 'styleUrl', '')
                    desc = getattr(f, 'description', '')
                    if '#ff0000' in str(style).lower() or '#ff0000' in str(desc).lower():
                        centerline = f.geometry

    extract_from_features(k.features())
    return centerline, dict(sorted(agms.items()))

def interpolate_along_line(line, point):
    min_dist = float('inf')
    min_proj = 0
    total = 0
    coords = list(line.coords)
    for i in range(len(coords) - 1):
        segment = LineString([coords[i], coords[i + 1]])
        proj = segment.project(point)
        dist = point.distance(segment)
        if dist < min_dist:
            min_dist = dist
            min_proj = total + proj
        total += segment.length
    return min_proj

def get_elevation(lat, lon):
    # Placeholder for elevation (replace with real API if needed)
    return 0.0

if uploaded_file:
    try:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        if file_ext == "kmz":
            with zipfile.ZipFile(uploaded_file) as zf:
                kml_name = [name for name in zf.namelist() if name.endswith(".kml")][0]
                kml_data = zf.read(kml_name)
        else:
            kml_data = uploaded_file.read()

        centerline, agms = extract_centerline_and_agms(kml_data)

        if centerline is None or not agms:
            st.error("❌ Centerline or AGMs not found. They must be under folders named 'CENTERLINE' and 'AGMs'.")
        else:
            coords = []
            keys = sorted(agms.keys())
            cumulative_ft = 0
            for i in range(len(keys) - 1):
                p1 = agms[keys[i]]
                p2 = agms[keys[i + 1]]
                dist1 = interpolate_along_line(centerline, p1)
                dist2 = interpolate_along_line(centerline, p2)
                if dist2 < dist1:
                    dist1, dist2 = dist2, dist1
                segment = centerline.interpolate(dist1), centerline.interpolate(dist2)
                segment_line = LineString([segment[0], segment[1]])
                points = list(segment_line.coords)
                seg_dist = 0
                for j in range(len(points) - 1):
                    lat1, lon1 = points[j][1], points[j][0]
                    lat2, lon2 = points[j+1][1], points[j+1][0]
                    ele1 = get_elevation(lat1, lon1)
                    ele2 = get_elevation(lat2, lon2)
                    seg_dist += haversine_3d((lat1, lon1, ele1), (lat2, lon2, ele2))
                cumulative_ft += seg_dist * 3.28084
                coords.append({
                    "From": keys[i],
                    "To": keys[i + 1],
                    "Segment Distance (ft)": round(seg_dist * 3.28084, 2),
                    "Cumulative Distance (mi)": round(cumulative_ft / 5280, 2)
                })

            df = pd.DataFrame(coords)
            st.success("✅ Distances calculated.")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Download CSV", csv, file_name="terrain_distances.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Failed to parse KMZ/KML: {e}")
