import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import math
import re
import zipfile
from io import BytesIO
import srtm  # pip install srtm.py

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
FT_PER_M   = 3.28084
MI_PER_FT  = 1 / 5280

# ─── LOAD & CACHE LOCAL SRTM ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_srtm_data():
    return srtm.get_data()

elev_data = load_srtm_data()

# ─── HELPER: PARSE STATION LABELS ──────────────────────────────────────────────
def parse_station(label: str) -> tuple[int, str]:
    """
    Given '240A' → (240, 'A'), '000' → (0, '')
    """
    m = re.match(r"^(\d+)([A-Za-z]*)$", label)
    if not m:
        return 0, ""
    return int(m.group(1)), m.group(2)

# ─── ELEVATION LOOKUP w/ FALLBACKS ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=5000, ttl=24*3600)
def get_elevation(lat: float, lon: float) -> float:
    # 1) SRTM (local cache)
    elev = elev
