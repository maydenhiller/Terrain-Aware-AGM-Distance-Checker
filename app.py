import streamlit as st
import requests
import math
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO

# ── Elevation Fetchers ─────────────────────────────────────────────────────────
def fetch_usgs_elev(lat, lon):
    url = "https://nationalmap.gov/epqs/pqs.php"
    params = {"x": lon, "y": lat, "units": "Meters", "output": "json"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return float(r.json()
                 ["USGS_Elevation_Point_Query_Service"]
                 ["Elevation_Query"]["Elevation"])

def fetch_open_elev(lat, lon):
    url = "https://api.open-elevation.com/api/v1/lookup"
    r = requests.get(url, params={"locations": f"{lat},{lon}"}, timeout=10)
    r.raise_for_status()
    return float(r.json()["results"][0]["elevation"])

def get_elevation(lat, lon):
    try:
        return fetch_usgs_elev(lat, lon), "USGS 3DEP"
    except Exception:
        return fetch_open_elev(lat, lon), "Open-Elevation"

# ── Distance Calc ─────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ── KML/KMZ Parser ────────────────────────────────────────────────────────────
def parse_centerline(file):
    data = file.read()
    if file.name.lower().endswith(".kmz"):
        z = zipfile.ZipFile(BytesIO(data))
        for name in z.namelist():
            if name.lower().endswith(".kml"):
                data = z.read(name)
                break
    root = ET.fromstring(data)
    ns = {"kml": root.tag.split("}")[0].strip("{")}
    coords = []
    for ls in root.findall(".//kml:LineString", ns):
        text = ls.find("kml:coordinates", ns).text.strip()
        for segment in text.split():
            lon, lat, *rest = segment.split(",")
            coords.append((float(lat), float(lon)))
    return coords

# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config("AGM Centerline Distance Checker", layout="wide")
st.title("🗺️ Terrain-Aware AGM Distances from KML/KMZ Centerline")

upload = st.file_uploader("Upload centerline (KML or KMZ)", type=["kml", "kmz"])
if upload:
    try:
        points = parse_centerline(upload)
    except Exception as e:
        st.error(f"Failed to parse KML/KMZ: {e}")
        points = []

    if points and st.button("▶️ Compute Distances"):
        total_2d = total_3d = 0.0
        rows = []

        for i in range(len(points) - 1):
            lat1, lon1 = points[i]
            lat2, lon2 = points[i + 1]

            d2d = haversine(lat1, lon1, lat2, lon2)
            e1, src1 = get_elevation(lat1, lon1)
            e2, src2 = get_elevation(lat2, lon2)
            d3d = math.sqrt(d2d**2 + (e2 - e1)**2)

            total_2d += d2d
            total_3d += d3d

            rows.append({
                "Segment": i + 1,
                "Start (lat,lon)": f"{lat1:.6f},{lon1:.6f}",
                "End (lat,lon)": f"{lat2:.6f},{lon2:.6f}",
                "2D Dist (m)": f"{d2d:.2f}",
                "Δ Elev (m)": f"{(e2-e1):.2f}",
                "3D Dist (m)": f"{d3d:.2f}",
                "Src Start": src1,
                "Src End": src2,
            })

        st.subheader("Segment-Wise Distances")
        st.table(rows)

        st.markdown(f"**Total 2D Distance:** {total_2d:.2f} m  ")
        st.markdown(f"**Total 3D (Terrain-Aware) Distance:** {total_3d:.2f} m")
    elif not points:
        st.warning("No LineString found in your KML/KMZ.")
