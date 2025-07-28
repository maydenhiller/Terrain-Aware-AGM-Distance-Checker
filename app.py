import streamlit as st
import requests
import zipfile
import io
import xml.etree.ElementTree as ET
import pandas as pd
import json

def extract_coords_from_kml_bytes(data_bytes):
    """
    Parse raw KML bytes and extract (lon, lat) tuples from all <coordinates> elements.
    """
    try:
        root = ET.fromstring(data_bytes)
    except ET.ParseError as e:
        st.error(f"XML parsing error: {e}")
        return []
    coords = []
    # find all coordinate blocks, regardless of namespace
    for elem in root.findall('.//{*}coordinates'):
        text = elem.text.strip() if elem.text else ""
        for group in text.split():
            parts = group.split(',')
            if len(parts) >= 2:
                try:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    coords.append((lon, lat))
                except ValueError:
                    continue
    return coords

def parse_kml_coords(uploaded_file):
    """
    Read uploaded_file (.kml or .kmz), extract raw KML bytes,
    and return list of (lon, lat) tuples.
    """
    raw = uploaded_file.read()
    name = uploaded_file.name.lower()

    # If KMZ, unzip the first .kml inside
    if name.endswith('.kmz'):
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for info in z.infolist():
                    if info.filename.lower().endswith('.kml'):
                        raw = z.read(info)
                        break
        except zipfile.BadZipFile:
            st.error("Uploaded KMZ is not a valid ZIP archive.")
            return []

    return extract_coords_from_kml_bytes(raw)

def query_opentopo(lat, lon):
    """
    Query OpenTopography global DEM API and return elevation or error message.
    """
    url = (
        "https://portal.opentopography.org/API/globaldem"
        f"?demtype=AW3D30&lat={lat}&lon={lon}&outputFormat=json"
    )
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json().get("elevation", "No elevation returned")
        return f"HTTP {resp.status_code}"
    except Exception as e:
        return f"Request error: {e}"

# Streamlit UI
st.title("🧭 OpenTopography Elevation Diagnostic Tool")

uploaded = st.file_uploader(
    "Upload a KML or KMZ file", type=["kml", "kmz"]
)

if uploaded:
    st.success(f"Received: {uploaded.name}")
    coords = parse_kml_coords(uploaded)

    if not coords:
        st.warning("No valid coordinates found in the file.")
    else:
        st.write(f"📌 Found **{len(coords)}** coordinate points.")

        # Build diagnostics list
        diagnostics = []
        for idx, (lon, lat) in enumerate(coords, start=1):
            elev = query_opentopo(lat, lon)
            diagnostics.append({
                "index": idx,
                "latitude": lat,
                "longitude": lon,
                "elevation": elev
            })

        # Show first 10 results
        st.subheader("🗻 Elevation Samples (first 10 points)")
        for row in diagnostics[:10]:
            st.write(
                f"{row['index']}. "
                f"({row['latitude']:.6f}, {row['longitude']:.6f}) → "
                f"Elevation: `{row['elevation']}`"
            )

        # Prepare DataFrame for export
        df = pd.DataFrame(diagnostics)

        # CSV download
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download diagnostics as CSV",
            data=csv_data,
            file_name="opentopo_diagnostics.csv",
            mime="text/csv"
        )

        # JSON download
        json_data = json.dumps(diagnostics, indent=2)
        st.download_button(
            label="📥 Download diagnostics as JSON",
            data=json_data,
            file_name="opentopo_diagnostics.json",
            mime="application/json"
        )
