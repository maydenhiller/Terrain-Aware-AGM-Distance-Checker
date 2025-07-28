import streamlit as st
import requests
import zipfile, io, time
import xml.etree.ElementTree as ET
import pandas as pd
import json, re

# hard-coded OpenTopography key:
OPTO_KEY = "49a90bbd39265a2efa15a52c00575150"

def extract_coords_from_kml_text(xml_text):
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        st.error(f"XML parsing error: {e}")
        return []
    pts = []
    for elem in root.findall('.//{*}coordinates'):
        for pair in (elem.text or "").strip().split():
            lon, lat = pair.split(',')[:2]
            try:
                pts.append((float(lon), float(lat)))
            except:
                continue
    return pts

def parse_kml_coords(uploaded):
    raw = uploaded.getvalue()
    name = uploaded.name.lower()

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

    text = raw.decode("utf-8", "ignore")
    text = re.sub(r"^<\?xml[^>]+\?>", "", text, count=1)
    return extract_coords_from_kml_text(text)

@st.cache_data(show_spinner=False)
def query_opentopo(lat, lon):
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype":      "AW3D30",
        "lat":          lat,
        "lon":          lon,
        "outputFormat": "JSON",
        "api_key":      OPTO_KEY,     # <— lowercase
    }

    # retry up to 3 times on 5xx
    for attempt in range(3):
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            try:
                return resp.json().get("elevation", "no elevation")
            except ValueError:
                return f"Invalid JSON response"
        if resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        return f"HTTP {resp.status_code}: {resp.text}"

    # fallback to Open-Elevation
    try:
        fb = requests.get(
            "https://api.open-elevation.com/api/v1/lookup",
            params={"locations": f"{lat},{lon}"},
            timeout=5
        )
        if fb.status_code == 200:
            return fb.json()["results"][0]["elevation"] \
                   + " (open-elevation)"
        return f"Fallback HTTP {fb.status_code}"
    except Exception as e:
        return f"Fallback error: {e}"

st.set_page_config(page_title="AGM Distance Debugger")
st.title("🚧 Terrain-Aware AGM Distance Debugger")

uploaded = st.file_uploader("Drag & drop KML/KMZ", type=["kml","kmz"])
if not uploaded:
    st.info("Upload a KML or KMZ file to begin.")
    st.stop()

coords = parse_kml_coords(uploaded)
if not coords:
    st.warning("No coordinates found.")
    st.stop()

st.success(f"Found {len(coords)} points in {uploaded.name}")

sample = st.slider("How many points to sample?", 1, min(1000, len(coords)), 10)
if st.button("Run Elevation Diagnostics"):
    results, prog = [], st.progress(0)
    for i, (lon, lat) in enumerate(coords[:sample], 1):
        elev = query_opentopo(lat, lon)
        results.append({
            "index":    i,
            "latitude": lat,
            "longitude":lon,
            "elevation":elev
        })
        prog.progress(i/ sample)

    st.subheader("Results")
    for r in results:
        st.write(f"{r['index']}. ({r['latitude']:.6f}, {r['longitude']:.6f}) → {r['elevation']}")

    df = pd.DataFrame(results)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), "diag.csv")
    st.download_button("Download JSON", json.dumps(results, indent=2), "diag.json")
