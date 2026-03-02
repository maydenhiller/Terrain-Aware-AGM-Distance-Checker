import streamlit as st
import zipfile
import tempfile
import xml.etree.ElementTree as ET
import math
import requests

st.title("Terrain-Aware AGM Distance Checker")

# RESTORED — EXACT ORIGINAL TOKEN METHOD
MAPBOX_TOKEN = st.secrets["mapbox"]["token"]

# ------------------ HAVERSINE ------------------

def haversine(p1, p2):
    R = 6371000
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ------------------ MAPBOX ELEVATION ------------------

@st.cache_data(show_spinner=False)
def get_elevation(lat, lon):
    url = f"https://api.mapbox.com/v4/mapbox.terrain-rgb/tilequery/{lon},{lat}.json"
    params = {"layers": "contour", "limit": 50, "access_token": MAPBOX_TOKEN}
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return 0
    data = r.json()
    if "features" not in data or not data["features"]:
        return 0
    return data["features"][0]["properties"]["ele"]


def terrain_distance(p1, p2, samples=10):
    total = 0
    prev = p1
    prev_z = get_elevation(*p1)

    for i in range(1, samples + 1):
        t = i / samples
        lat = p1[0] + (p2[0] - p1[0]) * t
        lon = p1[1] + (p2[1] - p1[1]) * t
        z = get_elevation(lat, lon)

        horizontal = haversine(prev, (lat, lon))
        dz = z - prev_z
        total += math.sqrt(horizontal**2 + dz**2)

        prev = (lat, lon)
        prev_z = z

    return total


# ------------------ KMZ PARSER ------------------

def parse_kmz(path):
    with zipfile.ZipFile(path) as z:
        kml_name = [n for n in z.namelist() if n.endswith(".kml")][0]
        root = ET.fromstring(z.read(kml_name))

    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    centerline = []
    agms = []

    for pm in root.findall(".//kml:Placemark", ns):
        name_el = pm.find("kml:name", ns)
        if name_el is None:
            continue

        name = name_el.text.strip()

        line = pm.find(".//kml:LineString/kml:coordinates", ns)
        if line is not None:
            coords = line.text.strip().split()
            for c in coords:
                lon, lat, *_ = map(float, c.split(","))
                centerline.append((lat, lon))

        point = pm.find(".//kml:Point/kml:coordinates", ns)
        if point is not None:
            if name.startswith("SP"):
                continue

            lon, lat, *_ = map(float, point.text.strip().split(","))
            agms.append({"name": name, "coord": (lat, lon)})

    return centerline, agms


# ------------------ PROJECT TO CENTERLINE ------------------

def cumulative(line):
    d = [0]
    for i in range(1, len(line)):
        d.append(d[-1] + haversine(line[i-1], line[i]))
    return d


def project(point, line, cum):
    best = float("inf")
    chain = 0

    for i in range(len(line)-1):
        d1 = haversine(point, line[i])
        d2 = haversine(point, line[i+1])

        if d1 < best:
            best = d1
            chain = cum[i]

        if d2 < best:
            best = d2
            chain = cum[i+1]

    return chain


# ------------------ MAIN ------------------

uploaded = st.file_uploader("Upload KMZ", type=["kmz"])

if uploaded:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".kmz") as tmp:
        tmp.write(uploaded.read())
        kmz_path = tmp.name

    centerline, agms = parse_kmz(kmz_path)

    if not centerline or not agms:
        st.error("Missing Centerline or AGMs folder.")
        st.stop()

    cum = cumulative(centerline)

    for a in agms:
        a["chain"] = project(a["coord"], centerline, cum)

    agms.sort(key=lambda x: x["chain"])

    start_keywords = ["000", "launcher", "launcher valve", "launch valve"]

    start_index = None
    for i, a in enumerate(agms):
        name_lower = a["name"].lower()
        if any(k in name_lower for k in start_keywords):
            start_index = i
            break

    if start_index is None:
        start_index = 0

    agms = agms[start_index:] + agms[:start_index]

    rows = []
    cumulative_dist = 0

    for i in range(len(agms) - 1):
        a1 = agms[i]
        a2 = agms[i+1]

        dist = terrain_distance(a1["coord"], a2["coord"])
        cumulative_dist += dist

        rows.append([
            a1["name"],
            a2["name"],
            round(dist * 3.28084, 2),
            round(cumulative_dist * 3.28084, 2)
        ])

    st.success("Done")
    st.dataframe(rows)

    csv = "From,To,Segment_ft,Cumulative_ft\n"
    for r in rows:
        csv += ",".join(map(str, r)) + "\n"

    st.download_button("Download CSV", csv, "agm_distances.csv")
