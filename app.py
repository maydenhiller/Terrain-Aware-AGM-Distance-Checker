# app.py — Terrain-Aware AGM Distances (Geodesic + Elevation, 25 m spacing)

import io, math, zipfile, xml.etree.ElementTree as ET
import numpy as np, pandas as pd, requests, streamlit as st
from PIL import Image
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import linemerge
from pyproj import Geod, Transformer

# ---------------- UI & CONFIG ----------------
st.set_page_config("Terrain AGM Distance — Geodesic", layout="wide")
st.title("📏 Terrain-Aware AGM Distance Calculator — Geodesic + Elevation")

# --- UNIVERSAL MAPBOX TOKEN HANDLER ---
def get_mapbox_token():
    if "MAPBOX_TOKEN" in st.secrets:
        return st.secrets["MAPBOX_TOKEN"]
    if "mapbox" in st.secrets:
        mb = st.secrets["mapbox"]
        if isinstance(mb, dict) and "token" in mb:
            return mb["token"]
        if isinstance(mb, str):
            return mb
    return None

MAPBOX_TOKEN = get_mapbox_token()

if not MAPBOX_TOKEN:
    st.error("❌ Missing Mapbox token. Add either:\n"
             "• MAPBOX_TOKEN = \"pk...\"\n"
             "or\n"
             "[mapbox]\n  token = \"pk...\"\n"
             "to your `.streamlit/secrets.toml`.")
    st.stop()

# ------------- REST OF SCRIPT BELOW UNCHANGED -------------
