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

# ── Elevation-Fetch Functions ──────────────────────────────────────────────────

def fetch_globaldem(lat: float, lon: float, demtype: str = "SRTMGL3") -> float:
    """
    Query OpenTopography point-based endpoint for a single (lat, lon).
    Returns elevation in meters or raises on failure.
    """
    url = "https://portal.opentopography.org/API/getdem"
    params = {
        "demtype":      demtype,
        "latitude":     f"{lat:.6f}",
        "longitude":    f"{lon:.6f}",
        "outputFormat": "JSON",
        "API_Key":      OPTO_KEY,
    }

    resp = requests.get(url, params=params, timeout=10)

    # Log full request & response for diagnostics
    logger.info("GlobalDEM Point URL      → %s", resp.request.url)
    logger.info("GlobalDEM Point Status   → %s", resp.status_code)
    logger.info("GlobalDEM Point Response → %s", resp.text)

    resp.raise_for_status()

    payload = resp.json()
    data = payload.get("data")
    if not data or "z" not in data[0]:
        msg = f"Unexpected payload: {resp.text}"
        logger.error(msg)
        raise RuntimeError("GlobalDEM point lookup returned no elevation")

    return float(data[0]["z"])


def fetch_open_elev(lat: float, lon: float) -> float:
    """
    Query Open-Elevation for a single (lat, lon).
    Returns elevation in meters or raises on failure.
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
        msg = f"Unexpected payload: {resp.text}"
        logger.error(msg)
        raise RuntimeError("Open-Elevation lookup returned no elevation")

    return float(results[0]["elevation"])


def get_elevation(
    lat: float, lon: float, demtype: str, method: str
) -> tuple[float, str]:
    """
    Returns (elevation_m, source_label), auto-falls back if GlobalDEM fails.
    """
    if method == "GlobalDEM (point)":
        try:
            elev = fetch_globaldem(lat, lon, demtype)
            return elev, "GlobalDEM"
        except Exception as e:
            logger.warning("GlobalDEM failed, falling back: %s", e)
            elev = fetch_open_elev(lat, lon)
            return elev, "Open-Elevation (fallback)"
    else:
        elev = fetch_open_elev(lat, lon)
        return elev, "Open-Elevation"

# ── Geodesic & Terrain Distance Calculations ──────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle (planar) distance between two points in meters.
    """
    R = 6_371_000  # Earth radius in meters
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_terrain_distance(
    lat1: float, lon1: float, lat2: float, lon2: float, demtype: str, method: str
):
    """
    Returns (planar_m, elev1_m, src1, elev2_m, src2, d3d_m).
    """
    planar = haversine(lat1, lon1, lat2, lon2)
    elev1, src1 = get_elevation(lat1, lon1, demtype, method)
    elev2, src2 = get_elevation(lat2, lon2, demtype, method)
    d3d = math.sqrt(planar ** 2 + (elev2 - elev1) ** 2)
    return planar, elev1, src1, elev2, src2, d3d

# ── Streamlit App Layout ───────────────────────────────────────────────────────

st.set_page_config(page_title="Terrain-Aware AGM Distance Checker", layout="wide")
st.title("Terrain-Aware AGM Distance Checker")
st.write("Compute planar + 3D distances using terrain elevation.")

# Sidebar: Raw API Diagnostics
st.sidebar.markdown("## 🔍 Elevation API Diagnostics")
with st.sidebar.expander("Run Raw Diagnostic", expanded=False):
    dlat = st.number_input("Latitude", value=34.703428, format="%.6f")
    dlon = st.number_input("Longitude", value=-95.101749, format="%.6f")
    dDEM = st.selectbox("DEM Type", ["SRTMGL3", "AW3D30"])
    dMethod = st.radio("Lookup Method", ["GlobalDEM (point)", "Open-Elevation"])

    if st.button("▶️ Test Now"):
        try:
            elev, src = get_elevation(dlat, dlon, dDEM, dMethod)
            st.success(f"[{src}] Elevation: {elev:.2f} m")
        except Exception as err:
            st.error(f"API call failed: {err}")

# Main pane: Distance inputs
st.header("Distance Calculation")
col1, col2 = st.columns(2)
with col1:
    lat1 = st.number_input("Start Latitude", value=34.703428, format="%.6f")
    lon1 = st.number_input("Start Longitude", value=-95.101749, format="%.6f")
with col2:
    lat2 = st.number_input("End Latitude", value=34.705000, format="%.6f")
    lon2 = st.number_input("End Longitude", value=-95.100000, format="%.6f")

demtype = st.selectbox("DEM Type", ["SRTMGL3", "AW3D30"])
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

        if "fallback" in (s1 + s2).lower():
            st.warning("One or more elevations used the Open-Elevation fallback.")

    except Exception:
        logger.exception("Calculation error")
        st.error("An unexpected error occurred. Check logs for details.")
