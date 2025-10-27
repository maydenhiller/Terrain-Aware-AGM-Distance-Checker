# app.py
# Terrain-Aware AGM Distance Calculator (FAST, Parser Unchanged)

import io
import math
import time
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from shapely.geometry import LineString, Point
from pyproj import Geod, Transformer

# ---------------- CONFIG ----------------
st.set_page_config("Terrain AGM Distance — FAST", layout="wide")
st.title("📏 Terrain-Aware AGM Distance Calculator — FAST")

MAPBOX_TOKEN = st.secrets.get("MAPBOX_TOKEN", "")
MAPBOX_ZOOM = 14
RESAMPLE_M = 10
SMOOTH_WINDOW = 40
DZ_THRESH = 0.5
FT_PER_M = 3.28084
GEOD = Geod(ellps="WGS84")
MAX_SNAP_M = 80

# ---------------- TERRAIN CACHE ----------------
class TerrainCache:
    def __init__(self, token: str, zoom: int):
        self.token = token
        self.zoom = int(zoom)
        self.tiles = {}

    @staticmethod
    def decode_rgb_arr(rgb_arr: np.ndarray) -> np.ndarray:
        r = rgb_arr[..., 0].astype(np.int64)
        g = rgb_arr[..., 1].astype(np.int64)
        b = rgb_arr[..., 2].astype(np.int64)
        return -10000.0 + (r * 256 * 256 + g * 256 + b) * 0.1

    def fetch_tile(self, z: int, x: int, y: int):
        key = (z, x, y)
        if key in self.tiles:
            return self.tiles[key]
        url = f"https://api.mapbox.com/v1/mapbox/terrain-rgb/{z}/{x}/{y}.pngraw"
        r = requests.get(url, params={"access_token": self.token}, timeout=12)
        if r.status_code == 401:
            st.error("❌ Mapbox 401 Unauthorized — check MAPBOX_TOKEN.")
            st.stop()
        if r.status_code == 403:
            st.error("❌ Mapbox 403 Forbidden — check token tileset access.")
            st.stop()
        if r.status_code == 429:
            time.sleep(2)
            return None
        if r.status_code != 200:
            print(f"[Mapbox {r.status_code}] {z}/{x}/{y}")
            return None
        arr = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=np.uint8)
        self.tiles[key] = arr
        return arr

    def elevations_bulk(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        z = self.zoom
        n = float(2 ** z)
        xt = (lons + 180.0) / 360.0 * n
        lat_r = np.radians(lats)
        yt = (1.0 - np.log(np.tan(lat_r) + 1.0 / np.cos(lat_r)) / math.pi) / 2.0 * n
        x_tile = np.floor(xt).astype(np.int64)
        y_tile = np.floor(yt).astype(np.int64)
        xp = (xt - x_tile) * 255.0
        yp = (yt - y_tile) * 255.0
