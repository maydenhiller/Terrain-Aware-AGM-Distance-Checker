import streamlit as st
import requests
import logging
import math

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
    Query OpenTopography GlobalDEM for a tiny bbox around (lat, lon).
    Returns elevation in meters.
    """
    url = "https://portal.opentopography.org/API/globaldem"
    delta = 1e-5
    params = {
        "demtype":      demtype,
        "south":        lat - delta,
        "north":        lat + delta,
        "west":         lon - delta,
        "east":         lon + delta,
        "outputFormat": "JSON",
        "API_Key":      OPTO_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        elevation = payload["data"][0]["elevation"]
        logger.info("GlobalDEM success: lat=%.6f lon=%.6f → elev=%.2f", lat, lon, elevation)
        return elevation

    except Exception as e:
        logger.exception("GlobalDEM API call failed")
        raise RuntimeError("GlobalDEM lookup failed") from e


def fetch_open_elev(lat: float, lon: float) -> float:
    """
    Query open-elevation.com for a single (lat, lon).
    Returns elevation in meters.
    """
    url = "https://api.open-elevation.com/api/v1/lookup"
    params = {"locations": f"{lat:.6f},{lon:.6f}"}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        elevation = payload["results"][0]["elevation"]
        logger.info("Open-Elevation success: lat=%.6f lon=%.6f → elev=%.2f", lat, lon, elevation)
        return elevation

    except Exception as e:
        logger.exception("Open-Elevation API call failed")
        raise RuntimeError("Open-Elevation lookup failed") from e


def get_elevation(
    lat: float, lon: float, demtype: str, method: str
) -> float:
    """
    Wrapper to choose between GlobalDEM bbox or fallback Open-Elevation.
    """
    if method == "GlobalDEM (bbox)":
        return fetch_globaldem(lat, lon, demtype)
    else:
        return fetch_open_elev(lat, lon)


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
    Returns a tuple of:
      (planar_distance_m, elev1_m, elev2_m, 3D_distance_m)
    """
    planar = haversine(lat1, lon1, lat2, lon2)
    elev1 = get_elevation(lat1, lon1, demtype, method)
    elev2 = get_elevation(lat2, lon2, demtype, method)
    height_diff = elev2 - elev1
    d3d = math.sqrt(planar ** 2 + height_diff ** 2)
    return planar, elev1, elev2, d3d


# ── Streamlit App Layout ───────────────────────────────────────────────────────
st.set_page_config(page_title="Terrain-Aware AGM Distance Checker", layout="wide")
st.title("Terrain-Aware AGM Distance Checker")
st.write(
    "Compute straight-line distances between two coordinates, "
    "adjusted for Earth curvature and terrain elevation."
)

# Sidebar: API Key Diagnostics & Method Selection
st.sidebar.markdown("## 🔍 Elevation API Diagnostics")

with st.sidebar.expander("Test API & View Raw Response", expanded=False):
    test_lat = st.number_input("Latitude", value=34.703428, format="%.6f")
    test_lon = st.number_input("Longitude", value=-95.101749, format="%.6f")
    test_dem = st.selectbox("DEM Type", ["SRTMGL3", "AW3D30"])
    test_method = st.radio("Lookup Method", ["GlobalDEM (bbox)", "Fallback Open-Elevation"])

    if st.button("▶️ Run Diagnostic"):
        try:
            elev = get_elevation(test_lat, test_lon, test_dem, test_method)
            st.success(f"Elevation: {elev:.2f} m")
        except Exception as err:
            st.error(f"API call failed: {err}")

# Main Inputs & Calculation
st.header("Distance Calculation")
col1, col2 = st.columns(2)

with col1:
    lat1 = st.number_input("Start Latitude", value=34.703428, format="%.6f")
    lon1 = st.number_input("Start Longitude", value=-95.101749, format="%.6f")

with col2:
    lat2 = st.number_input("End Latitude", value=34.705000, format="%.6f")
    lon2 = st.number_input("End Longitude", value=-95.100000, format="%.6f")

demtype = st.selectbox("DEM Type", ["SRTMGL3", "AW3D30"])
method = st.radio("Elevation Source", ["GlobalDEM (bbox)", "Fallback Open-Elevation"])

if st.button("🧮 Calculate Distance"):
    try:
        planar_dist, elev_start, elev_end, dist_3d = compute_terrain_distance(
            lat1, lon1, lat2, lon2, demtype, method
        )

        st.subheader("Results")
        st.write(f"Planar (2D) Distance: {planar_dist:.2f} m")
        st.write(f"Start Elevation: {elev_start:.2f} m")
        st.write(f"End Elevation: {elev_end:.2f} m")
        st.write(f"Adjusted 3D Distance: {dist_3d:.2f} m")

    except Exception:
        logger.exception("Failed to compute terrain-aware distance")
        st.error("An error occurred during calculation. Check logs for details.")
