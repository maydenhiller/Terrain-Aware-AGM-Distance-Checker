import streamlit as st
import requests
import zipfile
import io
import xml.etree.ElementTree as ET
import pandas as pd
import json
import re

# Hard-coded OpenTopography API key
OPTO_KEY = "49a90bbd39265a2efa15a52c00575150"

def extract_coords_from_kml_text(xml_text):
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        st.error(f"XML parsing error: {e}")
        return []
    pts = []
    for elem in root.findall('.//{*}coordinates'):
        text = elem.text or ""
        for group in text.strip().split():
            parts = group.split(',')
            if len(parts) >= 2:
                try:
                    lon, lat = map(float, parts[:2])
                    pts.append((lon, lat))
                except ValueError:
                    continue
    return pts

def parse_kml_coords(uploaded_file):
    raw = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    if name.endswith(".kmz"):
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for info in z.infolist():
                    if info.filename.lower().endswith(".kml"):
                        raw = z.read(info)
                        break
        except zipfile.BadZipFile:
            st.error("Uploaded KMZ is not a valid archive.")
            return []

    text = raw.decode("utf-8", errors="ignore")
    text = re.sub(r"^<\?xml[^>]+\?>", "", text, count=1)
    return extract_coords_from_kml_text(text)

@st.cache_data(show_spinner=False)
def query_opentopo(lat, lon):
    params = {
        "demtype": "AW3D30",
        "lat": lat,
        "lon": lon,
        "outputFormat": "JSON",
        "API_Key": OPTO_KEY
    }
    resp = requests.get(
        "https://portal.opentopography.org/API/globaldem",
        params=params,
        timeout=5
    )
    if resp.status_code == 200:
        return resp.json().get("elevation", "No elevation")
    return f"HTTP {resp.status_code}"

st.set_page_config(page_title="AGM Distance Debugger", layout="centered")
st.title("🚧 Terrain-Aware AGM Distance Debugger")

uploaded = st.file_uploader("Drag & drop KML/KMZ (≤200 MB)", type=["kml", "kmz"])
if not uploaded:
    st.info("Upload a KML or KMZ to begin.")
    st.stop()

coords = parse_kml_coords(uploaded)
total = len(coords)
st.success(f"Found {total:,} coordinate points in {uploaded.name}")

if total == 0:
    st.warning("No valid coordinates—check your file.")
    st.stop()

sample_size = st.slider(
    "Sample how many points for elevation?",
    min_value=1,
    max_value=min(1000, total),
    value=min(10, total)
)

if st.button("▶️ Run Elevation Diagnostics"):
    diagnostics = []
    progress = st.progress(0)
    for i, (lon, lat) in enumerate(coords[:sample_size], start=1):
        elev = query_opentopo(lat, lon)
        diagnostics.append({
            "index": i,
            "latitude": lat,
            "longitude": lon,
            "elevation": elev
        })
        progress.progress(i / sample_size)

    st.subheader(f"Results (first {sample_size} points)")
    for row in diagnostics:
        st.write(f"{row['index']}. ({row['latitude']:.6f}, {row['longitude']:.6f}) → {row['elevation']}")

    df = pd.DataFrame(diagnostics)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv_bytes, "diagnostics.csv", "text/csv")
    st.download_button(
        "📥 Download JSON",
        json.dumps(diagnostics, indent=2),
        "diagnostics.json",
        "application/json"
    )
