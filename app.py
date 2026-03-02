import zipfile
import xml.etree.ElementTree as ET
import math
import requests

# ================= SETTINGS =================
KMZ_FILE = "input.kmz"
OUTPUT_CSV = "Final_Terrain_Distances.csv"
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
# ============================================


# ---- Distance helpers ----
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))


def get_elevation(lat, lon):
    url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat},{lon}&key={GOOGLE_API_KEY}"
    r = requests.get(url).json()
    return r["results"][0]["elevation"] if r["results"] else 0


def terrain_distance(p1, p2):
    ground = haversine(p1[1], p1[0], p2[1], p2[0])
    z1 = get_elevation(p1[1], p1[0])
    z2 = get_elevation(p2[1], p2[0])
    return math.sqrt(ground**2 + (z2 - z1)**2)


# ---- Extract KMZ ----
def load_kmz(kmz_path):
    with zipfile.ZipFile(kmz_path) as z:
        kml_name = [f for f in z.namelist() if f.endswith(".kml")][0]
        return ET.fromstring(z.read(kml_name))


# ---- Parse centerline + AGMs ----
def parse_kml(root):
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    # Get centerline (red line assumed only LineString)
    coords_text = root.find(".//kml:LineString/kml:coordinates", ns).text.strip()
    centerline = [tuple(map(float, c.split(",")[:2])) for c in coords_text.split()]

    agms = []

    for pm in root.findall(".//kml:Placemark", ns):
        name_el = pm.find("kml:name", ns)
        pt = pm.find(".//kml:Point/kml:coordinates", ns)

        if name_el is None or pt is None:
            continue

        name = name_el.text.strip()

        # ❌ Ignore SP (capital only)
        if name.startswith("SP"):
            continue

        lon, lat = map(float, pt.text.strip().split(",")[:2])
        agms.append((name, (lon, lat)))

    return centerline, agms


# ---- Project point onto centerline ----
def project_onto_line(point, line):
    best_dist = float("inf")
    best_pos = 0
    cumulative = 0

    for i in range(len(line)-1):
        a = line[i]
        b = line[i+1]

        seg_len = haversine(a[1], a[0], b[1], b[0])

        # Approximate projection using distance to segment ends
        da = haversine(point[1], point[0], a[1], a[0])
        db = haversine(point[1], point[0], b[1], b[0])
        d = min(da, db)

        if d < best_dist:
            best_dist = d
            best_pos = cumulative

        cumulative += seg_len

    return best_pos


# ---- Determine start AGM ----
def find_start_index(agms):
    start_names = [
        "000",
        "launcher",
        "launcher valve",
        "launch valve"
    ]

    for i, (name, _) in enumerate(agms):
        if name.lower() in start_names:
            return i

    return 0  # fallback


# ---- Main processing ----
def process(kmz_path):
    root = load_kmz(kmz_path)
    centerline, agms = parse_kml(root)

    # Project AGMs onto centerline
    projected = []
    for name, coord in agms:
        pos = project_onto_line(coord, centerline)
        projected.append((name, coord, pos))

    # Sort along line
    projected.sort(key=lambda x: x[2])

    # Force starting point
    start_idx = find_start_index([(p[0], p[1]) for p in projected])
    projected = projected[start_idx:] + projected[:start_idx]

    # ---- Measure terrain distances ----
    results = []
    cumulative = 0

    for i in range(len(projected)-1):
        name1, c1, _ = projected[i]
        name2, c2, _ = projected[i+1]

        dist_m = terrain_distance(c1, c2)
        dist_ft = dist_m * 3.28084

        cumulative += dist_ft

        results.append((name1, name2, dist_ft, cumulative))

    # ---- Save CSV ----
    with open(OUTPUT_CSV, "w") as f:
        f.write("From,To,Segment_ft,Cumulative_ft\n")
        for r in results:
            f.write(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f}\n")


# ---- RUN ----
if __name__ == "__main__":
    process(KMZ_FILE)
