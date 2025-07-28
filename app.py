import streamlit as st
import requests

# Hard-coded key
OPTO_KEY = "49a90bbd39265a2efa15a52c00575150"

# … your other imports & functions …

st.sidebar.markdown("## 🔍 Key & Endpoint Test")
with st.sidebar.expander("Run Diagnostics", expanded=True):
    lat = st.number_input("Latitude",   value=34.703428, format="%.6f")
    lon = st.number_input("Longitude",  value=-95.101749, format="%.6f")
    endpoint = st.selectbox(
        "Endpoint",
        options=["/API/globaldem", "/API/point"],
        format_func=lambda e: e.replace("/API/", "")
    )
    demtype = st.selectbox("DEM Type", ["AW3D30", "SRTMGL3"]);
    if st.button("▶️ Test Now"):
        base = "https://portal.opentopography.org"
        params = {
            "demtype":      demtype,
            "outputFormat": "JSON",
            "API_Key":      OPTO_KEY,
        }

        # Choose params key names by endpoint
        if endpoint.endswith("globaldem"):
            params.update({"lat": lat, "lon": lon})
        else:  # /API/point
            params.update({"latitude": lat, "longitude": lon})

        resp = requests.get(f"{base}{endpoint}", params=params, timeout=10)

        st.write("**Full Request URL**")
        st.code(resp.request.url, language="bash")

        st.write("**Status Code**")
        st.write(resp.status_code)

        st.write("**Response Body**")
        st.write(resp.text)

# … rest of your app follows …
