import streamlit as st
import requests

# Your hard-coded key
OPTO_KEY = "49a90bbd39265a2efa15a52c00575150"

# … rest of your imports and functions …

# ── Insert this block right before st.set_page_config ──
st.sidebar.markdown("## 🔧 API Key Diagnostics")
with st.sidebar.expander("Run Key Test", expanded=True):
    test_lat = st.number_input("Latitude", value=34.703428, format="%.6f")
    test_lon = st.number_input("Longitude", value=-95.101749, format="%.6f")
    if st.button("▶️ Test OpenTopography Key"):
        url = "https://portal.opentopography.org/API/globaldem"
        params = {
            "demtype":      "AW3D30",
            "lat":          test_lat,
            "lon":          test_lon,
            "outputFormat": "JSON",
            "API_Key":      OPTO_KEY,
        }
        resp = requests.get(url, params=params, timeout=10)

        st.write("**Request URL**")
        st.code(resp.request.url, language="bash")

        st.write("**Status Code**")
        st.write(resp.status_code)

        st.write("**Response Body**")
        st.write(resp.text)
# ────────────────────────────────────────────────────────────

# Then your existing Streamlit app follows …
st.set_page_config(page_title="AGM Distance Debugger", layout="centered")
