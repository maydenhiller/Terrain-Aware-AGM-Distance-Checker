import streamlit as st
import zipfile
import tempfile
import xml.etree.ElementTree as ET
import math
import csv

st.title("Terrain-Aware AGM Distance Checker")

# ---------- Helpers ----------

def haversine(p1, p2):
    R = 6371000
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


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

        # ---- LineString (centerline)
        line = pm.find(".//kml:LineString/kml:coordinates", ns)
        if line is not None:
            coords = line.text.strip().split()
            for c in coords:
                lon, lat, *_ = map(float, c.split(","))
                centerline.append((lat, lon))

        # ---- Point (AGMs)
        point = pm.find(".//kml:Point/kml:coordinates", ns)
        if point is not None:
            if name.startswith("SP"):   # IGNORE SP points (capital only)
                continue

            lon, lat, *_ = map(float, point.text.strip().split(","))
            agms.append({"name": name, "coord": (lat, lon)})

    return centerline, agms


def cumulative_distances(line):
    dists = [0]
    for i in range(1, len(line)):
        dists.append(dists[-1] + haversine(line[i-1], line[i]))
    return dists


def project_to_line(point, line, cum_dists):
    best_dist = float("inf")
    best_chain = 0

    for i in range(len(line) - 1):
        p1 = line[i]
        p2 = line[i+1]

        d1 = haversine(point, p1)
        d2 = haversine(point, p2)

        if d1 < best_dist:
            best_dist = d1
            best_chain = cum_dists[i]

        if d2 < best_dist:
            best_dist = d2
            best_chain = cum_dists[i+1]

    return best_chain


# ---------- Processing ----------

uploaded = st.file_uploader("Upload KMZ", type=["kmz"])

if uploaded:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".kmz") as tmp:
        tmp.write(uploaded.read())
        kmz_path = tmp.name

    centerline, agms = parse_kmz(kmz_path)

    if not centerline or not agms:
        st.error("Missing centerline or AGM points")
        st.stop()

    cum_line = cumulative_distances(centerline)

    # ---- Project AGMs onto centerline
    for a in agms:
        a["chain"] = project_to_line(a["coord"], centerline, cum_line)

    # ---- Find starting AGM
    start_keywords = ["000", "launcher", "launcher valve", "launch valve"]

    start_agm = None
    for a in agms:
        name_lower = a["name"].lower()
        if any(k in name_lower for k in start_keywords):
            start_agm = a
            break

    if start_agm is None:
        start_agm = min(agms, key=lambda x: x["chain"])

    # ---- Sort AGMs along centerline
    agms_sorted = sorted(agms, key=lambda x: x["chain"])

    # ---- Rotate so start AGM is first
    start_index = agms_sorted.index(start_agm)
    agms_ordered = agms_sorted[start_index:] + agms_sorted[:start_index]

    # ---------- Measure ----------
    rows = []
    cumulative = 0

    for i in range(len(agms_ordered) - 1):
        a1 = agms_ordered[i]
        a2 = agms_ordered[i+1]

        seg = abs(a2["chain"] - a1["chain"])
        cumulative += seg

        rows.append([
            a1["name"],
            a2["name"],
            round(seg * 3.28084, 2),
            round(cumulative * 3.28084, 2)
        ])

    # ---------- Output ----------
    st.success("Processing complete")

    st.write("### Distances (feet)")
    st.dataframe(rows)

    csv_data = "From,To,Segment_ft,Cumulative_ft\n"
    for r in rows:
        csv_data += ",".join(map(str, r)) + "\n"

    st.download_button(
        "Download CSV",
        csv_data,
        "agm_distances.csv",
        "text/csv"
    )
