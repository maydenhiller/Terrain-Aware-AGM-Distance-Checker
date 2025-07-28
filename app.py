import streamlit as st
import requests

OPTO_KEY = "49a90bbd39265a2efa15a52c00575150"

st.sidebar.markdown("## 🌐 API Key & Endpoint Checker")
with st.sidebar.expander("Run Diagnostics", True):
    lat = st.number_input("Latitude",   value=34.703428, format="%.6f")
    lon = st.number_input("Longitude",  value=-95.101749, format="%.6f")
    demtype = st.selectbox("DEM Type", ["SRTMGL3", "AW3D30"])
    method = st.radio("Lookup Method", ["GlobalDEM (bbox)", "Fallback Open-Elevation"])

    if st.button("▶️ Test Now"):
        if method == "GlobalDEM (bbox)":
            # build a tiny box ±1e-5 degrees (~1m)
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
            # Open-Elevation fallback
            params = {"locations": f"{lat},{lon}"}
            url    = "https://api.open-elevation.com/api/v1/lookup"

        resp = requests.get(url, params=params, timeout=10)

        st.write("**Request URL**")
        st.code(resp.request.url, language="bash")

        st.write("**Status Code**")
        st.write(resp.status_code)

        st.write("**Response Body**")
        st.write(resp.text)

        st.markdown("---")
        st.markdown("**Browser Test URLs**")
        st.markdown(f"""
1. GlobalDEM (tiny box):
