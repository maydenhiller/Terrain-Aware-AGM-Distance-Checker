import streamlit as st
import zipfile, tempfile, os, requests, math, pandas as pd
import xml.etree.ElementTree as ET

GOOGLE_ELEVATION_API_KEY = "AIzaSyB9HxznAvlGb02e-K1rhld_CPeAm_wvPWU"
ELEVATION_URL = "https://maps.googleapis.com/maps/api/elevation/json"
ns = {"kml": "http://www.opengis.net/kml/2.2"}

st.set_page_config(layout="wide")
st.title("📐 Terrain-Aware AGM Distance Checker")

uploaded = st.file_uploader("Upload KMZ or KML file", type=["kmz","kml"])

def extract_kml_text(uploaded):
    content = uploaded.getvalue()
    if uploaded.name.endswith(".kmz"):
        with zipfile.ZipFile(tempfile.BytesIO(content)) as kmz:
            for f in kmz.namelist():
                if f.endswith(".kml"):
                    return kmz.read(f)
        raise ValueError("No KML inside KMZ.")
    else:
        return content

def parse_kml_bytes(kml_bytes):
    return ET.fromstring(kml_bytes)

def find_folder(root, name):
    for folder in root.findall(".//kml:Folder", ns):
        nm = folder.find("kml:name", ns)
        if nm is not None and nm.text.strip().upper() == name.upper():
            return folder
    return None

def extract_agms(folder):
    agms = []
    for pm in folder.findall("kml:Placemark", ns):
        nm = pm.find("kml:name", ns)
        pt = pm.find(".//kml:Point/kml:coordinates", ns)
        if nm is not None and pt is not None:
            if nm.text.strip().isdigit():
                lon, lat, *_ = map(float, pt.text.strip().split(","))
                agms.append((nm.text.strip(), lat, lon))
    agms.sort(key=lambda x:int(x[0]))
    return agms

def extract_centerline(folder):
    coords = []
    for pm in folder.findall("kml:Placemark", ns):
        ls = pm.find(".//kml:LineString/kml:coordinates", ns)
        if ls is not None:
            for line in ls.text.strip().split():
                lon, lat, *_ = map(float, line.split(","))
                coords.append((lat, lon))
    return coords

def get_elev(lat, lon):
    res = requests.get(f"{ELEVATION_URL}?locations={lat},{lon}&key={GOOGLE_ELEVATION_API_KEY}").json()
    return res["results"][0]["elevation"] if "results" in res and res["results"] else 0

def hav(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

def terrain_dist(p1, p2):
    lat1, lon1 = p1; lat2, lon2 = p2
    e1, e2 = get_elev(lat1, lon1), get_elev(lat2, lon2)
    f = hav(lat1, lon1, lat2, lon2)
    return math.sqrt(f*f + (e2 - e1)**2)

def find_index(pt, line):
    dmin, idx = float("inf"), 0
    for i, (lat, lon) in enumerate(line):
        d = hav(pt[0], pt[1], lat, lon)
        if d < dmin:
            dmin, idx = d, i
    return idx

if uploaded:
    try:
        data = extract_kml_text(uploaded)
        root = parse_kml_bytes(data)
        f_cl = find_folder(root, "CENTERLINE")
        f_a = find_folder(root, "AGMs")
        if not f_cl or not f_a:
            st.error("❌ Missing CENTERLINE or AGMs folder.")
            st.stop()
        cl = extract_centerline(f_cl)
        agms = extract_agms(f_a)
        if len(cl)<2 or len(agms)<2:
            st.error("❌ Need at least two AGMs and a centerline.")
            st.stop()

        df_rows, cumm = [], 0
        for i in range(len(agms)-1):
            a0,a1,a2 = agms[i]; b0,b1,b2 = agms[i+1]
            i0, i1 = find_index((a1,a2), cl), find_index((b1,b2), cl)
            if i0>i1: i0,i1 = i1,i0
            seg = cl[i0:i1+1]
            dist_m = sum(terrain_dist((x1,y1),(x2,y2)) for (x1,y1),(x2,y2) in zip(seg, seg[1:]))
            cumm += dist_m
            ft = dist_m*3.28084; mi = ft/5280
            c_ft = cumm*3.28084
            df_rows.append({"From":a0,"To":b0,"Segment (ft)":round(ft,2),
                            "Segment (mi)":round(mi,4),"Cumulative (ft)":round(c_ft,2)})
        df = pd.DataFrame(df_rows)
        st.success("✅ Done!")
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode(), "distances.csv", "text/csv")
    except Exception as ex:
        st.error(f"❌ Error: {ex}")
