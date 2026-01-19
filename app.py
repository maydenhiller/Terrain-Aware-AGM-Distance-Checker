import streamlit as st
import zipfile, io, math
import xml.etree.ElementTree as ET
import numpy as np
import requests
from PIL import Image

# ======================
# CONFIG
# ======================
MAPBOX_TOKEN = "pk.eyJ1IjoibWF5ZGVuaGlsbGVyIiwiYSI6ImNtZ2ljMnN5ejA3amwyam9tNWZnYnZibWwifQ.GXoTyHdvCYtr7GvKIW9LPA"
DENSIFY_M = 10
EARTH_R = 6371000
TILE_ZOOM = 14

# ======================
# GEO
# ======================
def haversine(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2 * EARTH_R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def densify(line):
    out = [line[0]]
    for i in range(len(line)-1):
        a, b = line[i], line[i+1]
        d = haversine(*a, *b)
        n = max(1, int(d // DENSIFY_M))
        for j in range(1, n+1):
            f = j / n
            out.append((a[0]+f*(b[0]-a[0]), a[1]+f*(b[1]-a[1])))
    return out

# ======================
# MAPBOX TERRAIN (CACHED)
# ======================
tile_cache = {}

def tile_xy(lat, lon, z):
    lat = math.radians(lat)
    n = 2**z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(lat) + 1/math.cos(lat)) / math.pi) / 2 * n)
    return x, y

def get_tile(z, x, y):
    key = (z,x,y)
    if key in tile_cache:
        return tile_cache[key]
    url = f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw?access_token={MAPBOX_TOKEN}"
    r = requests.get(url)
    img = Image.open(io.BytesIO(r.content))
    tile_cache[key] = img
    return img

def elevation(lat, lon):
    x,y = tile_xy(lat,lon,TILE_ZOOM)
    img = get_tile(TILE_ZOOM,x,y)
    px = img.load()
    w,h = img.size
    fx = int((lon + 180) / 360 * w) % w
    fy = int((1 - math.log(math.tan(math.radians(lat)) + 1/math.cos(math.radians(lat))) / math.pi) / 2 * h) % h
    R,G,B = px[fx,fy][:3]
    return -10000 + (R*256*256 + G*256 + B)*0.1

# ======================
# KML / KMZ PARSE
# ======================
def load_kml(upload):
    if upload.name.lower().endswith(".kmz"):
        z = zipfile.ZipFile(upload)
        kml = z.read([n for n in z.namelist() if n.endswith(".kml")][0])
    else:
        kml = upload.read()
    return ET.fromstring(kml)

def parse(root):
    ns={"k":"http://www.opengis.net/kml/2.2"}
    agms=[]
    center=[]
    for f in root.findall(".//k:Folder",ns):
        name=f.find("k:name",ns)
        if name is None: continue
        if name.text.upper()=="AGMS":
            for p in f.findall(".//k:Placemark",ns):
                n=p.find("k:name",ns).text
                lon,lat,_=map(float,p.find(".//k:coordinates",ns).text.split(","))
                agms.append((n,lat,lon))
        if name.text.upper()=="CENTERLINE":
            for ls in f.findall(".//k:LineString",ns):
                for c in ls.find("k:coordinates",ns).text.split():
                    lon,lat,_=map(float,c.split(","))
                    center.append((lat,lon))
    return agms, center

# ======================
# MAIN LOGIC
# ======================
st.title("Terrain-Aware AGM Distance Checker")

f=st.file_uploader("KML / KMZ",["kml","kmz"])

if f:
    root=load_kml(f)
    agms, center=parse(root)

    st.write(f"{len(agms)} AGMs | {len(center)} centerline pts")

    if not agms or not center:
        st.error("Need both AGMs and CENTERLINE")
        st.stop()

    center=densify(center)
    elev=[elevation(lat,lon) for lat,lon in center]

    # snap AGMs
    idx=[]
    for _,lat,lon in agms:
        d=[haversine(lat,lon,p[0],p[1]) for p in center]
        idx.append(int(np.argmin(d)))

    rows=[]
    for i in range(len(idx)-1):
        d=0
        for j in range(idx[i],idx[i+1]):
            h=haversine(*center[j],*center[j+1])
            v=elev[j+1]-elev[j]
            d+=math.sqrt(h*h+v*v)
        rows.append((agms[i][0],agms[i+1][0],d/1609.34))

    st.table({
        "From AGM":[r[0] for r in rows],
        "To AGM":[r[1] for r in rows],
        "Terrain Miles":[round(r[2],3) for r in rows]
    })

    st.success(f"Total: {sum(r[2] for r in rows):.3f} miles")
