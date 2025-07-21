import streamlit as st
import zipfile
import io

def extract_kml_from_kmz(kmz_bytes):
    with zipfile.ZipFile(io.BytesIO(kmz_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith(".kml"):
                return zf.read(name)
    return None

# Streamlit upload widget
uploaded_file = st.file_uploader("Upload a KMZ or KML file with a red centerline and numbered AGMs", type=["kmz", "kml"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".kmz"):
        kml_data = extract_kml_from_kmz(uploaded_file.read())
        if kml_data is None:
            st.error("No KML file found inside KMZ.")
            st.stop()
    elif uploaded_file.name.endswith(".kml"):
        kml_data = uploaded_file.read()
    else:
        st.error("Unsupported file format.")
        st.stop()

    centerline, agms = parse_kml(kml_data)

