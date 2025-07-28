import streamlit as st
import requests
import zipfile
import io
import xml.etree.ElementTree as ET
import pandas as pd
import json
import re

def extract_coords_from_kml_text(xml_text):
    """
    Parse a KML string (without <?xml…?> declaration) and extract (lon, lat) tuples.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        st.error(f"XML parsing error: {e}")
        return []
    coords = []
    for elem in root.findall('.//{*}coordinates'):
        if elem.text:
            for group in elem.text.strip().split():
                parts = group.split(',')
                if len(parts) >= 2:
                    try:
                        lon, lat = map(float, parts[:2])
                        coords.append((lon, lat))
                    except ValueError:
                        continue
    return coords

def parse_kml_coords(uploaded_file):
    """
    Read raw bytes from uploaded .kml or .kmz, strip the XML declaration,
    and return a list of (lon, lat) coordinate tuples.
    """
    raw = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

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

    # Decode to text, remove XML declaration, parse as KML
    text = raw.decode('utf-8', errors='ignore')
    text = re.sub(r'^<\?xml[^>]+\?>', '', text, count=1)
    return extract_coords_from_kml_text(text)

def query_opentopo(lat, lon):
    """
    Query the AW3D30 global DEM endpoint and return elevation or an error string.
    """
    url = (
        "https://portal.opentopography.org/API/globaldem"
        f"?demtype=AW3D30&lat={lat}&lon={lon}&outputFormat=json"
    )
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json().get("elevation", "No elevation returned")
        return f"HTTP {r.status_code}"
    except Exception as e:
        return f"Request error: {e}"

# Streamlit UI
st.title("🧭 OpenTopography Elevation Diagnostic Tool")
uploaded = st.file_uploader("Upload KML or KMZ file", type=["kml", "kmz"])

if uploaded:
    st.success(f"Received: {uploaded.name}")
    coords = parse_kml_coords(uploaded)

    if not coords:
        st.warning("No valid coordinates found.")
    else:
        st.write(f"📌 Found **{len(coords)}** coordinate points.")

        diagnostics = []
        for idx, (lon, lat) in enumerate(coords, start=1):
            elev = query_opentopo(lat, lon)
            diagnostics.append({
                "index": idx,
                "latitude": lat,
                "longitude": lon,
                "elevation": elev
            })

        st.subheader("🗻 Elevation Samples (first 10 points)")
        for row in diagnostics[:10]:
            st.write(
                f"{row['index']}. "
                f"({row['latitude']:.6f}, {row['longitude']:.6f}) → "
                f"{row['elevation']}"
            )

        df = pd.DataFrame(diagnostics)
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download diagnostics as CSV",
            data=csv_bytes,
            file_name="opentopo_diagnostics.csv",
            mime="text/csv"
        )
        json_bytes = json.dumps(diagnostics, indent=2)
        st.download_button(
            "📥 Download diagnostics as JSON",
            data=json_bytes,
            file_name="opentopo_diagnostics.json",
            mime="application/json"
        )
