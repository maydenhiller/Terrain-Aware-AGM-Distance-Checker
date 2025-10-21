import io, math, zipfile, requests, xml.etree.ElementTree as ET
import streamlit as st, pandas as pd, numpy as np
from PIL import Image
from shapely.geometry import LineString, Point
from pyproj import Transformer

# --- CONFIG ---
MAPBOX_TOKEN = st.secrets["mapbox"]["token"]
TERRAIN_TILE_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"
ZOOM = 15          # Max precision for Terrain-RGB
SPACING_M = 1.0    # 1m sampling along centerline

# --- CRS transformers (lon/lat <-> Web Mercator meters) ---
to_merc = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
to_geo  = Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)

# --- KML/KMZ parsing: only centerline ---
def parse_centerline(uploaded_file):
    if uploaded_file.name.endswith(".kmz"):
        with zipfile.ZipFile(uploaded_file) as zf:
            kml_file = next((f for f in zf.namelist() if f.endswith(".kml")), None)
            if not kml_file:
                return None
            with zf.open(kml_file) as f:
                kml_data = f.read()
    else:
        kml_data = uploaded_file.read()

    root = ET.fromstring(kml_data)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    centerline = None
    for folder in root.findall(".//kml:Folder", ns):
        name_el = folder.find("kml:name", ns)
        if not name_el or not name_el.text:
            continue
        if name_el.text.strip().lower() == "centerline":
            for pm in folder.findall("kml:Placemark", ns):
                coords_el = pm.find(".//kml:coordinates", ns)
                if coords_el is None:
                    continue
                pts = []
                for pair in coords_el.text.strip().split():
                    lon, lat, *_ = map(float, pair.split(","))
                    pts.append((lon, lat))
                if len(pts) >= 2:
                    centerline = LineString(pts)
    return centerline

# --- reprojection helpers ---
def line_ll_to_m(line_ll):
    return LineString([to_merc.transform(x, y) for (x, y) in line_ll.coords])

def merc_to_lonlat(pt_m):
    lon, lat = to_geo.transform(pt_m.x, pt_m.y)
    return lon, lat

# --- meter-spacing interpolation along the line ---
def interpolate_line_m(line_m, spacing_m=1.0):
    total = line_m.length
    steps = max(int(total / spacing_m), 1)
    pts = [line_m.interpolate(i * spacing_m) for i in range(steps)]
    pts.append(line_m.interpolate(total))
    return pts

# --- Terrain-RGB decoding ---
def decode_terrain_rgb(r, g, b):
    # E(m) = -10000 + (R*256^2 + G*256 + B) * 0.1
    return -10000.0 + (r * 256.0 * 256.0 + g * 256.0 + b) * 0.1

def lonlat_to_tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y  # fractional tile coords

class TerrainTileCache:
    def __init__(self, token, zoom=15):
        self.token = token
        self.zoom = zoom
        self.cache = {}  # (z, x_int, y_int) -> PIL Image (RGB)

    def get_tile_image(self, x_int, y_int):
        key = (self.zoom, x_int, y_int)
        if key in self.cache:
            return self.cache[key]
        url = TERRAIN_TILE_URL.format(z=self.zoom, x=x_int, y=y_int)
        r = requests.get(url, params={"access_token": self.token}, timeout=20)
        if r.status_code != 200:
            return None
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        self.cache[key] = img
        return img

    def elevation_bilinear(self, lon, lat):
        # fractional tile coordinates
        xf, yf = lonlat_to_tile_xy(lon, lat, self.zoom)
        x0, y0 = int(math.floor(xf)), int(math.floor(yf))
        dx, dy = xf - x0, yf - y0
        # pixel in each neighbor tile (256x256)
        def pix_in_tile(xf, yf):
            return int((xf - math.floor(xf)) * 256), int((yf - math.floor(yf)) * 256)
        x_pix, y_pix = pix_in_tile(xf, yf)
        # neighbor tiles
        tiles = [
            (x0,     y0    , x_pix,           y_pix          , (1 - dx) * (1 - dy)),  # top-left
            (x0 + 1, y0    , x_pix - 256,     y_pix          , dx * (1 - dy)),        # top-right
            (x0,     y0 + 1, x_pix,           y_pix - 256    , (1 - dx) * dy),        # bottom-left
            (x0 + 1, y0 + 1, x_pix - 256,     y_pix - 256    , dx * dy),              # bottom-right
        ]
        elev_sum, w_sum = 0.0, 0.0
        for xt, yt, px, py, w in tiles:
            # clamp pixels to tile bounds
            px = max(0, min(255, px))
            py = max(0, min(255, py))
            img = self.get_tile_image(xt, yt)
            if img is None: 
                continue
            r, g, b = img.getpixel((px, py))
            elev = decode_terrain_rgb(r, g, b)
            elev_sum += elev * w
            w_sum += w
        if w_sum == 0:
            return 0.0
        return elev_sum / w_sum

# --- distance ---
def distance_3d_m(p1_m, p2_m, e1, e2):
    dx = p2_m.x - p1_m.x
    dy = p2_m.y - p1_m.y
    dz = e2 - e1
    return math.sqrt(dx * dx + dy * dy + dz * dz)

# --- Streamlit UI ---
st.title("3D Centerline Length (Mapbox Terrain-RGB)")

uploaded = st.file_uploader("Upload KML or KMZ (Folder name = CENTERLINE)", type=["kml", "kmz"])

if uploaded:
    centerline_ll = parse_centerline(uploaded)
    if not centerline_ll or len(centerline_ll.coords) < 2:
        st.error("CENTERLINE missing or invalid.")
    else:
        st.success("CENTERLINE found.")
        # Project to meters
        centerline_m = line_ll_to_m(centerline_ll)
        # Interpolate 1m path points
        path_pts_m = interpolate_line_m(centerline_m, SPACING_M)
        # Sample Terrain-RGB elevations with bilinear interpolation
        cache = TerrainTileCache(MAPBOX_TOKEN, zoom=ZOOM)
        elevations = []
        for pm in path_pts_m:
            lon, lat = merc_to_lonlat(pm)
            elevations.append(cache.elevation_bilinear(lon, lat))
        # Sum 3D length
        total_3d_m = 0.0
        for i in range(len(path_pts_m) - 1):
            total_3d_m += distance_3d_m(path_pts_m[i], path_pts_m[i + 1],
                                        elevations[i], elevations[i + 1])
        total_ft = total_3d_m * 3.28084
        total_mi = total_ft / 5280.0

        # Output
        st.subheader("Results")
        st.markdown(f"**Total 3D centerline length (feet):** {total_ft:,.2f}")
        st.markdown(f"**Total 3D centerline length (miles):** {total_mi:,.6f}")

        # Optional CSV download (per-sample for audit)
        df = pd.DataFrame({
            "lon": [merc_to_lonlat(p)[0] for p in path_pts_m],
            "lat": [merc_to_lonlat(p)[1] for p in path_pts_m],
            "elevation_m": elevations
        })
        st.download_button("Download sampled points CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="centerline_3d_samples.csv", mime="text/csv")
