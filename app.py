# app.py

import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import math
import re
from io import BytesIO
import zipfile
import srtm  # local SRTM lookup

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
FT_PER_M = 3.28084
MI_PER_FT = 1 / 5280

# ── LOAD & CACHE LOCAL SRTM ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_srtm():
    return srtm.get_data()

elev_data = load_srtm()

# ── HELPER: PARSE ALPHANUMERIC STATION IDS ───────────────────────────────────────
def parse_station(label: str) -> tuple[int, str]:
    """
    Split a label like "240A" into (240, "A").
    If no letters follow, suffix is empty.
    """
    m = re.match(r"^(\d+)([A-Za-z]*)$", label)
    if not m:
        return 0, ""
    num = int(m.group(1))
    suffix = m.group(2)
    return num, suffix

# ── ELEVATION LOOKUP WITH FALLBACKS ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=10000, ttl=24*3600)
def get_elevation(lat: float, lon: float) -> float:
    # 1) LOCAL SRTM  
    elev = elev_data.get_elevation(lat, lon)
    if elev is not None:
        return elev

    # 2) USGS EPQS (LiDAR / best‐available DEM)
    try:
        r = requests.get(
            "https://nationalmap.gov/epqs/pqs.php",
            params={"x": lon, "y": lat, "units": "Meters", "output": "json"},
            timeout=5
        )
        r.raise_for_status()
        e = r.json()["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]["Elevation"]
        if e is not None:
            return float(e)
    except:
        pass

    # 3) OPEN‐ELEVATION fallback
    try:
        r = requests.get(
            "https://api.open-elevation.com/api/v1/lookup",
            params={"locations": f"{lat:.6f},{lon:.6f}"},
            timeout=5
        )
        r.raise_for_status()
        return float(r.json()["results"][0]["elevation"])
    except:
        return 0.0

# ── HAVERSINE IN METERS ─────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # radius in meters
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ── PARSE KMZ / KML ─────────────────────────────────────────────────────────────
def parse_kml_kmz(file) -> tuple[list[tuple[str,float,float]], list[tuple[float,float]]]:
    raw = file.read()
    if file.name.lower().endswith(".kmz"):
        with zipfile.ZipFile(BytesIO(raw
