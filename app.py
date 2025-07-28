# app.py

import streamlit as st
import requests
import math
import logging

# ── Logging Configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Hard-Coded API Key ─────────────────────────────────────────────────────────
OPTO_KEY = "49a90bbd39265a2efa15a52c00575150"

# ── Session State Initialization ───────────────────────────────────────────────
if "globaldem_ok" not in st.session_state:
    st.session_state.globaldem_ok = True

# ── Sidebar Controls ───────────────────────────────────────────────────────────
st.sidebar.header("Service Controls")
if st.sidebar.button("Reset GlobalDEM"):
    st.session_state.globaldem_ok = True
    st.sidebar.success("GlobalDEM re-enabled for this session")

# ── Elevation-Fetch Functions ───────────────────────────────────────────────────

def fetch_globaldem(lat: float, lon: float, demtype: str = "SRTMGL3") -> float:
    """
    Query OpenTopography’s GlobalDEM point endpoint.
    Raises on HTTP errors or unexpected payloads.
    """
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype":      demtype,
        "location":     f"{lat:.6f},{lon:.6f}",
        "outputFormat": "JSON",
        "API_Key":      OPTO_KEY,
    }

    resp = requests.get(url, params=params, timeout=10)
    logger.info("GlobalDEM URL      → %s", resp.request.url)
    logger.info("GlobalDEM Status   → %s", resp.status_code)
    logger.info("GlobalDEM Response → %s", resp.text)

    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data")
    if not data or ("elevation" not in data[0] and "z" not in data[0]):
        raise RuntimeError("GlobalDEM returned unexpected payload")

    return float(data[0].get("elevation", data[0].get("z")))


def fetch_open_elev(lat: float, lon: float) -> float:
    """
    Query Open-Elevation for a single (lat, lon).
    """
    url = "https://api.open-elevation.com/api/v1/lookup"
    params = {"locations": f"{lat:.6f},{lon:.6f}"}

    resp = requests.get(url, params=params, timeout=10)
    logger.info("Open-Elev URL      → %s", resp.request.url)
    logger.info("Open-Elev Status   → %s", resp.status_code)
    logger.info("Open-Elev Response → %s", resp.text)

    resp.raise_for_status()
    results = resp.json().get("results")
    if not results or "elevation" not in results[0]:
        raise RuntimeError("Open-Elevation returned unexpected payload")

    return float(results[0]["elevation"])


def get_elevation(lat: float, lon: float, demtype: str, method: str):
    """
    Returns (elevation_m, source_label).
    Falls back to Open-Elevation if GlobalDEM fails or is disabled.
    """
    if method == "GlobalDEM (point)" and st.session_state.globaldem_ok:
        try:
            return fetch_globaldem(lat, lon, demtype), "GlobalDEM"
        except Exception as e:
            logger.warning("GlobalDEM failed: %s", e)
            st.session_state.globaldem_ok = False
            st.error("GlobalDEM service unavailable. Now defaulting to Open-Elevation.")

    return fetch_open_elev(lat, lon), "Open-Elevation"

# ── Distance Calculations ───────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two points (in meters).
    """
    R = 6_371_000  # Earth radius in meters
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_terrain_distance(lat1, lon1, lat2, lon2, demtype, method):
    """
    Returns (planar_m, elev1, src1, elev2, src2, d3d_m).
    """
    planar = haversine(lat1, lon1, lat2, lon2)
    e1, s1 = get_elevation(lat1, lon1, demtype, method)
    e2, s2 = get_elevation(lat2, lon2, demtype, method)
    d3d = math.sqrt(planar**2 + (e2 - e1)**2)
    return planar, e1, s1, e2, s2, d3d

# ── Streamlit App Layout ───────────────────────────────────────────────────────

st.set_page_config(page_title="Terrain-Aware AGM Distance Checker", layout="wide")
st.title("Terrain-Aware AGM Distance Checker")
st.write("Compute planar + 3D distances using terrain elevation.")

# Inputs
col1, col2 = st.columns(2)
with col1:
    lat1 = st.number_input("Start Latitude", value=34.703428, format="%.6f")
    lon1 = st.number_input("Start Longitude", value=-95.101749, format="%.6f")
with col2:
    lat2 = st.number_input("End Latitude", value=34.705000, format="%.6f")
    lon2 = st.number_input("End Longitude", value=-95.100000, format="%.6f")

demtype = st.selectbox("DEM Type", ["SRTMGL3", "AW3D30"])

if not st.session_state.globaldem_ok:
    st.warning("GlobalDEM disabled; using Open-Elevation")
    method = "Open-Elevation"
else:
    method = st.radio("Elevation Source", ["GlobalDEM (point)", "Open-Elevation"])

if st.button("🧮 Calculate Distance"):
    try:
        planar, e1, s1, e2, s2, d3d = compute_terrain_distance(
            lat1, lon1, lat2, lon2, demtype, method
        )
        st.subheader("Results")
        st.write(f"Planar (2D): {planar:.2f} m")
        st.write(f"Start Elevation ({s1}): {e1:.2f} m")
        st.write(f"End Elevation   ({s2}): {e2:.2f} m")
        st.write(f"3D Distance:    {d3d:.2f} m")
    except Exception as exc:
        logger.exception("Calculation error")
        st.error(f"Error calculating distance: {exc}")
```
