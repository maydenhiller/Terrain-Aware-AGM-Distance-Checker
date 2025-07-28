import streamlit as st
import numpy as np
import math
import srtm

# ── Setup ───────────────────────────────────────────────────────────────────────

st.set_page_config("Fast AGM Distances", layout="wide")
st.title("🗻 Fast Terrain-Aware Distances via Local SRTM")

# Load local SRTM data (first run will download needed tiles)
@st.cache_data(show_spinner=False)
def load_srtm():
    return srtm.get_data()

elev_data = load_srtm()

# Haversine vectorized
def haversine_batch(lats, lons):
    R = 6_371_000  # Earth radius in meters
    φ = np.radians(lats)
    λ = np.radians(lons)
    dφ = φ[1:] - φ[:-1]
    dλ = λ[1:] - λ[:-1]
    a = np.sin(dφ/2)**2 + np.cos(φ[:-1])*np.cos(φ[1:])*np.sin(dλ/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# ── UI & Logic ─────────────────────────────────────────────────────────────────

uploaded = st.file_uploader("Upload centerline (KML/KMZ)", type=["kml","kmz"])
if not uploaded:
    st.info("Please upload a file to begin.")
    st.stop()

# Parse coords (your existing parse_centerline function)
pts = parse_centerline(uploaded)
if len(pts) < 2:
    st.error("Need at least 2 points.")
    st.stop()

lats = np.array([p[0] for p in pts])
lons = np.array([p[1] for p in pts])

if st.button("▶️ Compute Fast Distances"):
    with st.spinner("Crunching numbers…"):
        # 2D distances vectorized
        d2d = haversine_batch(lats, lons)

        # Elevations via local SRTM
        elevs = np.array([elev_data.get_elevation(lat, lon) or 0 
                          for lat, lon in pts])

        # 3D distances
        delta_e = elevs[1:] - elevs[:-1]
        d3d = np.sqrt(d2d**2 + delta_e**2)

        # Totals
        total_2d = d2d.sum()
        total_3d = d3d.sum()

    # Display results
    st.markdown(f"**Total 2D Distance:** {total_2d:,.2f} m")  
    st.markdown(f"**Total 3D Distance:** {total_3d:,.2f} m")

    # Optional: show a progress bar and table of first 100 segments
    if st.checkbox("Show sample segments"):
        rows = []
        for i in range(min(100, len(d2d))):
            rows.append({
                "Segment": i+1,
                "2D (m)": f"{d2d[i]:.2f}",
                "ΔElev (m)": f"{delta_e[i]:.2f}",
                "3D (m)": f"{d3d[i]:.2f}",
            })
        st.table(rows)
