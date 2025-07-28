import streamlit as st
import requests
OPTO_KEY = st.secrets["49a90bbd39265a2efa15a52c00575150"]
# ── API Key & Endpoint Diagnostics ──────────────────────────────────────────

st.sidebar.markdown("## 🔍 API Key Diagnostics")
with st.sidebar.expander("Run Diagnostics", expanded=True):
    lat = st.number_input("Latitude", value=34.703428, format="%.6f")
    lon = st.number_input("Longitude", value=-95.101749, format="%.6f")
    demtype = st.selectbox("DEM Type", ["SRTMGL3", "AW3D30"])
    method = st.radio("Lookup Method", ["GlobalDEM (bbox)", "Fallback Open-Elevation"])

    if st.button("▶️ Test Now"):
        if method == "GlobalDEM (bbox)":
            # Build a tiny bbox ±1e-5°
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
            url = "https://portal.opentopography.org/API/globaldem"
        else:
            url = "https://api.open-elevation.com/api/v1/lookup"
            params = {"locations": f"{lat},{lon}"}

        resp = requests.get(url, params=params, timeout=10)

        # Show what was actually called
        st.write("**Request URL**")
        st.code(resp.request.url)

        st.write("**Status Code**")
        st.write(resp.status_code)

        st.write("**Response Body**")
        st.write(resp.text)

        st.markdown("---")
        st.write("**Browser-Ready URLs**")

        # GlobalDEM browser URL
        globaldem_url = (
            f"https://portal.opentopography.org/API/globaldem"
            f"?demtype={demtype}"
            f"&south={lat:.6f}&north={lat:.6f}"
            f"&west={lon:.6f}&east={lon:.6f}"
            f"&outputFormat=JSON&API_Key={OPTO_KEY}"
        )
        st.code(globaldem_url, language="bash")

        # Open-Elevation browser URL
        open_elev_url = (
            f"https://api.open-elevation.com/api/v1/lookup"
            f"?locations={lat:.6f},{lon:.6f}"
        )
        st.code(open_elev_url, language="bash")

# ───────────────────────────────────────────────────────────────────────────────
# Sanity check – you should see “OK” in the sidebar
st.sidebar.write("Sanity check: OK")
