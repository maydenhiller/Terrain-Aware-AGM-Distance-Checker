import os
import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from math import radians, cos, sin, asin, sqrt

st.set_page_config(page_title="Terrain-Aware Distance Calculator")

# ---------------------------------------------
#  Config: Your OpenTopography API Key (opt.)
# ---------------------------------------------
OT_KEY = os.getenv("OPENTOPO_KEY")  # set in Streamlit Cloud Secrets

# ---------------------------------------------
#  Elevation Fetcher with Fallback & Caching
# ---------------------------------------------
@st.cache_data(show_spinner=False, max_entries=5000, ttl=24*3600)
def get_elevation(lat: float, lon: float) -> float:
    # 1) USGS EPQS (LiDAR / best DEM)
    try:
        r = requests.get(
            "https://nationalmap.gov/epqs/pqs.php",
            params={
                "x": lon, "y": lat,
                "units": "Meters", "output": "json"
            },
            timeout=8
        )
        r.raise_for_status()
        elev = r.json()["USGS_Elevation_Point_Query_Service"] \
                     ["Elevation_Query"]["Elevation"]
        if elev is not None:
            return elev
    except Exception:
        pass

    # 2) OpenTopography Point Query (if key provided)
    if OT_KEY:
        try:
            ot = requests.get(
                "https://portal.opentopography.org/API/point",
                params={
                    "demtype": "SRTMGL1",  # or your preferred DEM
                    "x": lon, "y": lat,
                    "outputFormat": "JSON",
                    "key": OT_KEY
                },
                timeout=8
            )
            ot.raise_for_status()
            jo = ot.json()
            return jo["data"]["elevation"]
        except Exception:
            pass

    # 3) Open-Elevation batch fallback
    try:
        payload = {"locations": [{"latitude": lat, "longitude": lon}]}
        r2 = requests.post(
            "https://api.open-elevation.com/api/v1/lookup",
            json=payload,
            timeout=8
        )
        r2.raise_for_status()
        return r2.json()["results"][0]["elevation"]
    except Exception:
        pass

    # 4) Last‐resort fallback
    return 0.0

# ---------------------------------------------
#  3D Haversine for Segment Length
# ---------------------------------------------
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * 6371000 * asin(sqrt(a))

def segment_length(p1, p2):
    e1, e2 = get_elevation(*p1), get_elevation(*p2)
    plan = haversine(p1[1], p1[0], p2[1], p2[0])
    vert = abs(e2 - e1)
    return sqrt(plan**2 + vert**2)

# ---------------------------------------------
#  UI: Upload + Compute
# ---------------------------------------------
st.title("Terrain-Aware Centerline Distance")

f = st.file_uploader("Upload your KML/KMZ", type=["kml","kmz"])
if not f:
    st.warning("Upload a KML or KMZ to begin.")
    st.stop()

# Parse coords
root = ET.fromstring(f.read())
coords = []
for node in root.findall(".//{*}coordinates"):
    for text in node.text.strip().split():
        lon, lat, *_ = map(float, text.split(","))
        coords.append((lat, lon))

segments = list(zip(coords, coords[1:]))
st.info(f"Calculating {len(segments)} segments…")

lengths_m = [segment_length(a, b) for a, b in segments]
total_m = sum(lengths_m)

df = pd.DataFrame({
    "Segment #": range(1, len(lengths_m)+1),
    "Length (ft)": [m * 3.28084 for m in lengths_m],
    "Length (mi)": [m * 0.000621371 for m in lengths_m]
})

st.dataframe(df, use_container_width=True)
st.markdown(f"**Total:** {total_m*0.000621371:.4f} mi | {total_m*3.28084:.1f} ft")

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name="distances.csv")

