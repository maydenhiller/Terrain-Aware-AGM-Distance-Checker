import streamlit as st
import pandas as pd

# --- Terrain-aware 3D distance logic ---
# This version includes the elevation-aware calculation inline
# so you don't have to swap imports or modules.

from math import sqrt
import requests

def get_elevation(lat, lon):
    """
    Replace this with your actual multi-source elevation lookup.
    Below is a placeholder using Open-Elevation API for demonstration.
    """
    url = "https://api.open-elevation.com/api/v1/lookup"
    params = {"locations": f"{lat},{lon}"}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        return data['results'][0]['elevation']
    except Exception:
        return 0.0

def compute_3d_distance(coords):
    """
    Computes total 3D distance along the given coordinate list,
    integrating elevation at each point.
    """
    total = 0.0
    if len(coords) < 2:
        return total

    elev_cache = {}
    for i in range(len(coords) - 1):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[i + 1]

        # Elevations with caching
        if (lat1, lon1) not in elev_cache:
            elev_cache[(lat1, lon1)] = get_elevation(lat1, lon1)
        if (lat2, lon2) not in elev_cache:
            elev_cache[(lat2, lon2)] = get_elevation(lat2, lon2)

        z1 = elev_cache[(lat1, lon1)]
        z2 = elev_cache[(lat2, lon2)]

        # Haversine for horizontal distance in meters
        from math import radians, sin, cos, atan2
        R = 6371000  # Earth radius in meters
        phi1, phi2 = radians(lat1), radians(lat2)
        dphi = radians(lat2 - lat1)
        dlambda = radians(lon2 - lon1)
        a = sin(dphi/2)**2 + cos(phi1) * cos(phi2) * sin(dlambda/2)**2
        horiz_dist = 2 * R * atan2(sqrt(a), sqrt(1-a))

        # Vertical difference
        vert_diff = z2 - z1

        # 3D segment length
        seg_len = sqrt(horiz_dist**2 + vert_diff**2)
        total += seg_len

    return total

# --- Streamlit UI ---
st.title("Terrain‑Aware Centerline Distance Calculator")

uploaded_centerline = st.file_uploader(
    "Upload Centerline TXT",
    type=["txt", "csv"],
    help="Extracted table with latitude, longitude, color, kml_folder"
)

if uploaded_centerline:
    # Load tab‑ or comma‑separated file
    df = pd.read_csv(uploaded_centerline, sep=None, engine="python")

    # Filter for red CENTERLINE entries
    mask = (
        df['kml_folder'].astype(str).str.strip().eq('CENTERLINE')
    ) | (
        df['color'].astype(str).str.strip().eq('#ff0000')
    )
    centerline_df = df[mask].copy()

    centerline_coords = list(
        zip(centerline_df['latitude'], centerline_df['longitude'])
    )

    st.write(f"Loaded {len(centerline_coords)} centerline points.")

    if st.button("Compute Terrain‑Aware Distance"):
        total_distance_m = compute_3d_distance(centerline_coords)
        st.success(f"Total terrain‑aware centerline length: {total_distance_m:,.2f} m")
