import streamlit as st
import zipfile
import tempfile
import os
import xml.etree.ElementTree as ET
from math import radians, cos, sin, sqrt
import pandas as pd
import requests

# --- Constants ---
ELEVATION_API_KEY = "AIzaSyB9HxznAvlGb02e-K1rhld_CPeAm_wvPWU"
ELEVATION_API_URL = "https://maps.googleapis.com/maps/api/elevation/json"

# --- Functions ---

def parse_kml_from_kmz(uploaded_file):
    with tempfile.TemporaryDirectory() as tmpdirname:
        kmz_path = os.path.join(tmpdirname, "uploaded.kmz")
        with open(kmz_path, "wb
