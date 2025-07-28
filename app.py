import streamlit as st
import requests, zipfile, io, re, json, pandas as pd, xml.etree.ElementTree as ET
from time import sleep

# ——————————————————————————————————————————————————————————————
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
            lon, lat = group.split(",")[:2]
            try:
                pts.append((float(lon), float(lat)))
            except:
                continue
    return pts

def parse_kml_coords(uploaded_file) -> list[tuple]:
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
            st.error("Invalid KMZ archive.")
            return []
    # decode & strip <?xml…?> so ElementTree won’t choke on encoding declarations
    text = raw.decode("utf-8", errors="ignore")
    text = re.sub(r"^<\?xml[^>]+\?>", "", text, count=1)
    return extract_coords_from_kml_text(text)

@st.cache_data(show_spinner=False)
def query_opentopo(lat, lon):
    url = (
        "https://portal.opentopography.org/API/globaldem"
        f"?demtype=AW3D30&lat={lat}&lon={lon}&outputFormat=json"
    )
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json().get("elevation", "No elevation")
        return f"HTTP {r.status_code}"
    except Exception as e:
        return f"Error: {e}"

# ——————————————————————————————————————————————————————————————
st.set_page_config(page_title="AGM Distance Debugger", layout="centered")
st.title("🚧 Terrain-Aware AGM Distance Debugger")

uploaded = st.file_uploader("Drag & drop KML/KMZ (≤ 200 MB)", type=["kml", "kmz"])
if not uploaded:
    st.info("Upload a KML or KMZ to get started.")
    st.stop()

# parse immediately so we know count
coords = parse_kml_coords(uploaded)
total = len(coords)
st.success(f"Found {total:,} coordinate points in {uploaded.name}")

if total == 0:
    st.stop()

# let user pick how many to sample
sample_size = st.slider(
    "How many points to sample for diagnostics?",
    min_value=1, max_value=min(1000, total), value=min(10, total), step=1
)
run = st.button("▶️ Run Elevation Diagnostics")

if run:
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
        sleep(0.05)  # tiny break so UI can update

    st.subheader(f"Results (first {sample_size} points)")
    for row in diagnostics:
        st.write(f"{row['index']}. ({row['latitude']:.6f}, {row['longitude']:.6f}) → {row['elevation']}")

    df = pd.DataFrame(diagnostics)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv_bytes, "diagnostics.csv", "text/csv")
    st.download_button("📥 Download JSON", json.dumps(diagnostics, indent=2),
                       "diagnostics.json", "application/json")
