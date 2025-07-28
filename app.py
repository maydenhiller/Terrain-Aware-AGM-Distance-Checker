import streamlit as st
import requests
import zipfile
import io
import xml.etree.ElementTree as ET
import pandas as pd
import json
import re

def extract_coords_from_kml_text(xml_text):
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        st.error(f"XML parsing error: {e}")
        return []
    coords = []
    for elem in root.findall('.//{*}coordinates'):
        if elem.text:
            for point in elem.text.strip().split():
                parts = point.split(',')
                if len(parts) >= 2:
                    try:
                        lon, lat = map(float, parts[:2])
                        coords.append((lon, lat))
                    except ValueError:
                        continue
    return coords

def parse_kml_coords(uploaded_file):
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

    text = raw.decode('utf-8', errors='ignore')
    text = re.sub(r'^<\?xml[^>]+\?>', '', text, count=1)
    return extract_coords_from_kml_text(text)

def query_opentopo(lat, lon):
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

st.set_page_config(page_title="🗺️ AGM Distance Debugger", layout="centered")
st.title("🚧 Terrain-Aware AGM Distance Debugger (OpenTopography)")

uploaded = st.file_uploader(
    "Drag and drop a KML or KMZ file (Limit 200MB)", 
    type=["kml", "kmz"]
)

if uploaded:
    st.success(f"Received: {uploaded.name}")
    coords = parse_kml_coords(uploaded)
    total = len(coords)

    if total == 0:
        st.warning("No valid coordinates found.")
    else:
        st.write(f"📌 Found **{total}** coordinate points.")
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
        for d in diagnostics[:10]:
            st.write(f"{d['index']}. ({d['latitude']:.6f}, {d['longitude']:.6f}) → {d['elevation']}")

        df = pd.DataFrame(diagnostics)
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv_bytes, "opentopo_diag.csv", "text/csv")

        json_str = json.dumps(diagnostics, indent=2)
        st.download_button("📥 Download JSON", json_str, "opentopo_diag.json", "application/json")
