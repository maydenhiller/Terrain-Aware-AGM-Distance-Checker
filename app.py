import streamlit as st
import zipfile, tempfile, io, math, re
import xml.etree.ElementTree as ET
import requests
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import linemerge
from pyproj import CRS, Transformer
import srtm
import pandas as pd

# ---------------- Config ----------------
METERS_TO_FEET = 3.28084
FEET_PER_MILE = 5280
MAX_POINTS_PER_SEGMENT = 200  # max number of points to sample along segment

EPQS_URL = "https://nationalmap.gov/epqs/pqs.php"
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"

# ---------------- UI ----------------
st.set_page_config(page_title="Terrain-aware AGM distances", layout="wide")
st.title("Terrain-aware AGM distances (KML/KMZ)")

with st.expander("Options", expanded=False):
    show_debug = st.checkbox("Show debug info", value=True)

uploaded = st.file_uploader("Upload KML or KMZ containing AGMs and CENTERLINE", type=["kml","kmz"])
if not uploaded:
    st.stop()

# ---------------- Helpers ----------------
def read_uploaded_bytes(file) -> bytes:
    file.seek(0)
    return file.read()

def utm_crs_for(lats, lons):
    lat_mean = sum(lats)/len(lats)
    lon_mean = sum(lons)/len(lons)
    zone = int((lon_mean + 180)//6)+1
    epsg = 32600 + zone if lat_mean >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)

def parse_station(label: str):
    m = re.match(r"^\s*(\d+)\s*([A-Za-z]*)\s*$", label or "")
    return (int(m.group(1)), m.group(2)) if m else (0, "")

# ---------------- KML Parsing ----------------
def parse_kml_for_agms_and_centerline(kml_bytes: bytes):
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    root = ET.fromstring(kml_bytes)

    # Numeric AGMs only
    agms=[]
    for fld in root.findall(".//kml:Folder", ns):
        nm=fld.find("kml:name", ns)
        if nm is None or (nm.text or "").strip().upper()!="AGMS":
            continue
        for pm in fld.findall("kml:Placemark", ns):
            name_el=pm.find("kml:name", ns)
            coord_el=pm.find(".//kml:Point/kml:coordinates", ns)
            if name_el is None or coord_el is None or not (coord_el.text or "").strip():
                continue
            label=(name_el.text or "").strip()
            lon,lat,*_=coord_el.text.strip().split(",")
            if label.isnumeric():
                agms.append((label,float(lon),float(lat)))

    # CENTERLINE only
    centerline_segments=[]
    for fld in root.findall(".//kml:Folder", ns):
        nm=fld.find("kml:name", ns)
        if nm is None or not (nm.text or "").strip().upper().startswith("CENTERLINE"):
            continue
        for pm in fld.findall(".//kml:LineString", ns):
            coords_el = pm.find("kml:coordinates", ns)
            if coords_el is None or not (coords_el.text or "").strip():
                continue
            seg=[]
            for token in coords_el.text.strip().split():
                lon,lat,*_=token.split(",")
                seg.append((float(lon),float(lat)))
            if len(seg)>=2:
                centerline_segments.append(seg)
    return agms, centerline_segments

# ---------------- Elevation ----------------
@st.cache_resource
def get_srtm():
    return srtm.get_data()

srtm_data = get_srtm()

@st.cache_data(ttl=86400)
def get_elevation(lat, lon):
    # EPQS
    try:
        r = requests.get(EPQS_URL, params={"x":lon,"y":lat,"units":"Meters","output":"json"}, timeout=6)
        r.raise_for_status()
        e=r.json()["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]["Elevation"]
        if e is not None:
            return float(e)
    except: pass
    # SRTM fallback
    try:
        e = srtm_data.get_elevation(lat,lon)
        if e is not None:
            return float(e)
    except: pass
    # Open Elevation
    try:
        r = requests.get(OPEN_ELEVATION_URL, params={"locations":f"{lat:.6f},{lon:.6f}"}, timeout=6)
        r.raise_for_status()
        return float(r.json()["results"][0]["elevation"])
    except: return 0.0

# ---------------- Geometry ----------------
def build_centerline_utm(segments_ll, to_utm):
    utm_lines=[]
    for seg in segments_ll:
        xs, ys = to_utm.transform(*zip(*seg))
        utm_lines.append(LineString(list(zip(xs, ys))))
    merged=linemerge(MultiLineString(utm_lines)) if len(utm_lines)>1 else utm_lines[0]
    if isinstance(merged, LineString):
        return merged
    parts=list(merged.geoms)
    parts.sort(key=lambda g:g.length, reverse=True)
    return parts[0]

def adaptive_densify(line_utm, s0, s1):
    dist = abs(s1-s0)
    n = min(MAX_POINTS_PER_SEGMENT, max(2,int(dist)))  # max 200 points
    return [line_utm.interpolate(s0 + (s1-s0)*i/(n-1)) for i in range(n)]

def terrain_distance_m(points, to_wgs84):
    xs, ys = zip(*[(p.x,p.y) for p in points])
    lons, lats = to_wgs84.transform(xs, ys)
    elevs = [get_elevation(lat, lon) for lat,lon in zip(lats,lons)]
    total=0.0
    for i in range(len(points)-1):
        x1,y1=xs[i],ys[i]; x2,y2=xs[i+1]; h=math.hypot(x2-x1,y2-y1)
        v=elevs[i+1]-elevs[i]
        total+=math.hypot(h,v)
    return total

# ---------------- Main ----------------
data = read_uploaded_bytes(uploaded)
if uploaded.name.lower().endswith(".kmz"):
    with zipfile.ZipFile(io.BytesIO(data),"r") as zf:
        kml_filename = [f for f in zf.namelist() if f.lower().endswith(".kml")][0]
        kml_bytes = zf.read(kml_filename)
else:
    kml_bytes = data

agms, centerline_ll = parse_kml_for_agms_and_centerline(kml_bytes)
if not agms: st.error("No numeric AGMs found."); st.stop()
if not centerline_ll: st.error("No CENTERLINE found."); st.stop()

all_lons = [lon for seg in centerline_ll for lon,_ in seg]
all_lats = [lat for seg in centerline_ll for _,lat in seg]
crs_utm = utm_crs_for(all_lats, all_lons)
to_utm = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
to_wgs84 = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)

line_utm = build_centerline_utm(centerline_ll, to_utm)

agms_sorted = sorted(agms, key=lambda x: parse_station(x[0]))
agm_chain=[]
for label, lon, lat in agms_sorted:
    x,y = to_utm.transform(lon, lat)
    s = line_utm.project(Point(x,y))
    agm_chain.append((label, lon, lat, s))

# Rebase to AGM 000
offset=None
for lab,lo,la,s in agm_chain:
    if parse_station(lab)[0]==0:
        offset=s
        break
if offset is None: st.error("No AGM 000 found"); st.stop()
agm_chain = [(lab,lo,la,s-offset) for lab,lo,la,s in agm_chain]

if show_debug:
    st.subheader("🔧 AGM projection (chainage in feet)")
    st.dataframe([{"AGM":lab,"lon":lo,"lat":la,"chainage_ft":round(s*METERS_TO_FEET,2)}
                  for lab,lo,la,s in agm_chain])

# Compute distances
rows=[]
total_ft=0.0
for i in range(len(agm_chain)-1):
    lab1,lo1,la1,s0 = agm_chain[i]
    lab2,lo2,la2,s1 = agm_chain[i+1]
    pts = adaptive_densify(line_utm, s0+offset, s1+offset)
    d_m = terrain_distance_m(pts, to_wgs84)
    d_ft = d_m*METERS_TO_FEET
    d_mi = d_ft/FEET_PER_MILE
    total_ft+=d_ft
    rows.append({
        "Segment": f"{lab1} → {lab2}",
        "Distance (ft)": round(d_ft,2),
        "Distance (mi)": round(d_mi,4),
        "Cumulative distance (ft)": round(total_ft,2),
        "Cumulative distance (mi)": round(total_ft/FEET_PER_MILE,4)
    })

df=pd.DataFrame(rows)

st.subheader("AGM Segment Distances (Terrain-aware)")
st.dataframe(df,use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name="agm_segment_distances.csv", mime="text/csv")
