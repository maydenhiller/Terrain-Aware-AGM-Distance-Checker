import streamlit as st, pandas as pd, numpy as np, math, io, zipfile, xml.etree.ElementTree as ET, requests, time
from shapely.geometry import Point, LineString
from pyproj import Transformer, Geod
from PIL import Image

# ---------------- CONFIG ----------------
MAPBOX_TOKEN = st.secrets.get("MAPBOX_TOKEN", "")
MAPBOX_ZOOM = 14
RESAMPLE_M = 20
SMOOTH_WINDOW = 40
FT_PER_M = 3.28084
DZ_THRESH = 0.5
GEOD = Geod(ellps="WGS84")

# ---------------- TERRAIN CACHE (patched for Mapbox v1) ----------------
class TerrainCache:
    def __init__(self, token, zoom):
        self.token = token
        self.zoom = zoom
        self.cache = {}

    def _fetch(self, z, x, y):
        key = (z, x, y)
        if key in self.cache:
            return self.cache[key]

        url = f"https://api.mapbox.com/v1/mapbox/terrain-rgb/{z}/{x}/{y}.pngraw"
        try:
            r = requests.get(url, params={"access_token": self.token}, timeout=10)
        except Exception as e:
            print(f"[Mapbox fetch error] {e}")
            return None

        if r.status_code == 401:
            st.error(
                "❌ Mapbox returned 401 Unauthorized — your MAPBOX_TOKEN is invalid, missing, or "
                "does not include 'Downloads: Tilesets' scope. "
                "Go to https://account.mapbox.com → Access Tokens → Create token → Enable Tilesets."
            )
            st.stop()
        elif r.status_code == 403:
            st.error("❌ Mapbox 403 Forbidden — token domain restriction or tileset access missing.")
            st.stop()
        elif r.status_code == 429:
            st.warning("⚠ Mapbox rate limit hit — retrying...")
            time.sleep(2)
            return None
        elif r.status_code != 200:
            print(f"[Mapbox HTTP {r.status_code}] for tile {z}/{x}/{y}")
            return None

        arr = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), np.uint8)
        self.cache[key] = arr
        return arr

    def elev(self, lon, lat):
        n = 2 ** self.zoom
        xt = (lon + 180.0) / 360.0 * n
        lat_rad = math.radians(lat)
        yt = (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n
        x, y = int(xt), int(yt)
        arr = self._fetch(self.zoom, x, y)
        if arr is None:
            return 0
        rx, ry = int((xt - x) * 255), int((yt - y) * 255)
        r, g, b = arr[min(ry, 255), min(rx, 255)]
        return -10000 + (r * 256 * 256 + g * 256 + b) * 0.1


# ---------------- HELPERS ----------------
def smooth(a, win, dx):
    if len(a) < 3: 
        return a
    n = max(3, int(round(win / dx)))
    if n % 2 == 0:
        n += 1
    k = np.ones(n) / n
    return np.convolve(a, k, "same")


def parse(file):
    """Parse AGMs and CENTERLINE from a KML or KMZ file."""
    if file.name.lower().endswith(".kmz"):
        with zipfile.ZipFile(file) as z:
            kml_name = [n for n in z.namelist() if n.endswith(".kml")][0]
            xml = z.read(kml_name)
    else:
        xml = file.read()

    root = ET.fromstring(xml)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    agms, lines = [], []

    for pm in root.findall(".//kml:Folder[kml:name='AGMs']/kml:Placemark", ns):
        name = pm.findtext("kml:name", "", ns).strip()
        if name.startswith("SP"):  # ignore SP-prefixed AGMs
            continue
        c = pm.find(".//kml:coordinates", ns)
        if c is not None:
            vals = c.text.strip().split(",")
            if len(vals) >= 2:
                agms.append((name, Point(float(vals[0]), float(vals[1]))))

    for ln in root.findall(".//kml:Folder[kml:name='CENTERLINE']//kml:LineString", ns):
        c = ln.find("kml:coordinates", ns)
        if c is None:
            continue
        pts = []
        for item in c.text.strip().split():
            parts = item.split(",")
            if len(parts) >= 2:
                pts.append((float(parts[0]), float(parts[1])))
        if pts:
            lines.append(LineString(pts))

    return agms, lines


# ---------------- STREAMLIT UI ----------------
st.set_page_config("Terrain AGM Distance", layout="wide")
st.title("📏 Terrain-Aware AGM Distance Calculator")

u = st.file_uploader("Upload KML/KMZ", type=["kml", "kmz"])
if not u:
    st.stop()

agms, lines = parse(u)
st.text(f"{len(agms)} AGMs | {len(lines)} centerline part(s)")
if not agms or not lines:
    st.warning("Need both AGMs and CENTERLINE folders.")
    st.stop()

xf_fwd = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
xf_inv = Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)

parts_m = []
for p in lines:
    if p.length > 0:
        x, y = xf_fwd.transform(*p.xy)
        parts_m.append(LineString(list(zip(x, y))))

cache = TerrainCache(MAPBOX_TOKEN, MAPBOX_ZOOM)
rows, cum_mi = [], 0.0
bar = st.progress(0)
msg = st.empty()
total = len(agms) - 1

for i in range(total):
    n1, a1 = agms[i]
    n2, a2 = agms[i + 1]
    msg.text(f"⏱ Calculating {n1} → {n2} ({i + 1}/{total}) …")
    bar.progress((i + 1) / total)

    p1 = Point(xf_fwd.transform(a1.x, a1.y))
    p2 = Point(xf_fwd.transform(a2.x, a2.y))
    part = parts_m[0]

    s1, s2 = part.project(p1), part.project(p2)
    s_lo, s_hi = sorted((s1, s2))
    if s_hi - s_lo <= 0:
        continue

    si = np.arange(s_lo, s_hi, RESAMPLE_M)
    if si.size == 0 or si[-1] < s_hi:
        si = np.append(si, s_hi)

    x, y = np.asarray(part.xy[0]), np.asarray(part.xy[1])
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    xi, yi = np.interp(si, s, x), np.interp(si, s, y)
    xi, yi = smooth(xi, SMOOTH_WINDOW, RESAMPLE_M), smooth(yi, SMOOTH_WINDOW, RESAMPLE_M)

    lons, lats = xf_inv.transform(xi.tolist(), yi.tolist())
    pts = list(zip(lons, lats))
    elev = [cache.elev(lo, la) for lo, la in pts]
    elev = smooth(np.array(elev), SMOOTH_WINDOW, RESAMPLE_M)

    dist = 0.0
    for j in range(len(pts) - 1):
        lon1, lat1 = pts[j]
        lon2, lat2 = pts[j + 1]
        _, _, dxy = GEOD.inv(lon1, lat1, lon2, lat2)
        dz = elev[j + 1] - elev[j]
        if abs(dz) < DZ_THRESH:
            dz = 0
        dist += math.hypot(dxy, dz)

    ft, mi = dist * FT_PER_M, dist * FT_PER_M / 5280
    cum_mi += mi
    rows.append({
        "From AGM": n1,
        "To AGM": n2,
        "Feet": round(ft, 1),
        "Miles": round(mi, 4),
        "Cumulative": round(cum_mi, 4)
    })

msg.text("✅ Complete.")
bar.progress(1.0)
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)
st.download_button("Download CSV", df.to_csv(index=False).encode(), "terrain_distances.csv", "text/csv")
