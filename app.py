# app.py
import io
import math
import re
import zipfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Point, LineString
from shapely.ops import substring
from pyproj import Transformer, Geod

# ---------------- CONFIG ----------------
# Hard-coded Mapbox token (you said to hard-code). Replace with st.secrets if you prefer.
MAPBOX_TOKEN = "pk.eyJ1IjoibWF5ZGVuaGlsbGVyIiwiYSI6ImNtZ2ljMnN5ejA3amwyam9tNWZnYnZibWwifQ.GXoTyHdvCYtr7GvKIW9LPA"

# Sampling resolution (meters) — user chose 10 m
RESAMPLE_M = 10.0

# Centerline snap limit: if AGM is > this meters away from centerline, skip
MAX_SNAP_M = 200.0

# Mapbox terrain zoom (higher zoom -> more tiles but same per-pixel precision)
TERRAIN_Z = 15  # good balance for Terrain-RGB

# Parallel tile prefetch
MAX_WORKERS = 8

# Basic constants
FT_PER_M = 3.28084
MI_PER_FT = 1.0 / 5280.0
GEOD = Geod(ellps="WGS84")

st.set_page_config(page_title="Terrain-Aware AGM Distance Checker", layout="wide")
st.title("Terrain-Aware AGM Distance Checker — Full 3D (Mapbox Terrain-RGB)")

# ---------------- HELPERS: robust KML/KMZ parsing ----------------

def strip_prefixes_kml_text(kml_text: str) -> str:
    """
    Strip namespace prefixes (e.g. gx:, kml:, xsi:) from tags to avoid unbound prefix errors,
    while preserving all text content. Also remove xml:base/xsi declarations that break parsing.
    """
    # remove xml declaration weirdness (keep first <?xml ... ?> if present)
    # Remove xmlns:* declarations to avoid unused/unbound prefix problems
    # But keep default namespace declaration if present - not required if we strip prefixes.
    # Simple approach: remove prefixes in tags like <gx:LineString> -> <LineString>
    s = kml_text
    # remove namespace declarations: xmlns:something="..."
    s = re.sub(r'\s+xmlns:[a-zA-Z0-9_-]+="[^"]+"', '', s)
    # remove standalone xsi:schemaLocation or similar
    s = re.sub(r'\s+xsi:[a-zA-Z0-9_-]+="[^"]+"', '', s)
    # remove prefixes in element names like <gx:... or </gx:...
    s = re.sub(r'<(/?)[a-zA-Z0-9_]+:([a-zA-Z0-9_\-]+)', r'<\1\2', s)
    # remove prefixes in closing tags that might remain
    s = re.sub(r'</[a-zA-Z0-9_]+:([a-zA-Z0-9_\-]+)>', r'</\1>', s)
    return s

def parse_kml_kmz(uploaded_file):
    """
    Returns: agms: list of (name, shapely.Point(lon,lat))
             centerlines: list of shapely.LineString (lon, lat)
    Parsing rules (keeps your parser behavior):
    - Looks for <Folder><name>AGMs</name> ... Placemark/Point/coordinates
    - Ignores AGM names that start with 'SP' (case-insensitive)
    - Looks for <Folder><name>CENTERLINE</name> ... Placemark/LineString/coordinates
    - Robust to kmz with nested directories
    """
    data_bytes = None
    if uploaded_file.name.lower().endswith(".kmz"):
        # read the kmz and find the
