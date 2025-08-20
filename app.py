import os
import io
import math
import zipfile
import xml.etree.ElementTree as ET

import requests
import streamlit as st
import pandas as pd
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import linemerge
from pyproj import CRS, Transformer
import srtm  # pip install srtm.py

# ---------------------------- Config ----------------------------
METERS_TO_FEET = 3.28084
FEET_PER_MILE = 5280
DEFAULT_STEP_M = 5.0  # densification step along centerline in meters

# Elevation sources
EPQS_URL = "https://nationalmap.gov/epqs/pqs.php"
OPENTOPO_URL = "https://portal.opentopography.org/API/point"
OPENTOPO_KEY = os.getenv("OPENTOPO_KEY") or st.secrets.get("OPENTOPO_KEY", None)
OPENTOPO_DEMTYPES = ["USGS3DEP1m", "USGSNED10m", "SRTMGL1"]
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"

# ---------------------------- UI ----------------------------
st.set_page_config(page_title="Terrain-aware AGM distances", layout="wide")
st.title("Terrain-aware AGM distances (KML/KMZ)")

with st.expander("Options", expanded=False):
    step_m = st.number_input(
        "Densification step (meters)",
        min_value=1.0,
        max_value=50.0,
        value=DEFAULT_STEP_M,
        step=1.0,
        help="Smaller steps increase accuracy but require more elevation queries."
    )
    show_debug = st.checkbox("Show debug info", value=False)

uploaded = st.file_uploader("Upload KML or KMZ containing AGMs and CENTERLINE", type=["kml", "kmz"])
if not uploaded:
    st.stop()

# ---------------------------- Helpers ----------------------------
def read_uploaded_bytes(file) -> bytes:
    file.seek(0)
    return file.read()

def extract_kml_from_kmz(data: bytes) -> bytes | None:
    with zipfile.ZipFile(io.BytesIO(data), 'r') as zf:
        for name in zf.namelist():
            if name.lower().endswith(".kml"):
                return zf.read(name)
    return None

def utm_crs_for(lats, lons) -> CRS:
    lat_mean = sum(lats) / len(lats)
    lon_mean = sum(lons) / len(lons)
    zone = int((lon_mean + 180) // 6) + 1
    epsg = 32600 + zone if lat_mean >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)

def parse_station
