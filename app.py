import streamlit as st, pandas as pd, numpy as np, math, io, zipfile, xml.etree.ElementTree as ET, requests, time
from shapely.geometry import Point, LineString
from pyproj import Transformer, Geod
from PIL import Image

MAPBOX_TOKEN = st.secrets.get("MAPBOX_TOKEN", "")
MAPBOX_ZOOM = 14
RESAMPLE_M = 20
SPACING_M = 20
FT_PER_M = 3.28084
DZ_THRESH = 0.5
MAX_SNAP_M = 80
GEOD = Geod(ellps="WGS84")
TERRAIN = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"

# ---- elevation tile cache with retry + timeout ----
class TerrainCache:
    def __init__(self, token, zoom):
        self.t, self.z, self.c = token, zoom, {}

    def _fetch(self, z, x, y):
        key = (z, x, y)
        if key in self.c:
            return self.c[key]
        for attempt in range(3):
            try:
                r = requests.get(
                    TERRAIN.format(z=z, x=x, y=y),
                    params={"access_token": self.t},
                    timeout=10
                )
                if r.status_code == 200:
                    arr = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), np.uint8)
                    self.c[key] = arr
                    return arr
                elif r.status_code in (401, 403, 429):
                    print(f"Mapbox {r.status_code} on tile {x},{y}")
                    time.sleep(1)
            except Exception as e:
                print(f"Retry {attempt}: {e}")
                time.sleep(1)
        return None

    def elev(self, lon, lat):
        n = 2 ** self.z
        xt = (lon + 180.0) / 360.0 * n
        yt = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
        x, y = int(xt), int(yt)
        arr = self._fetch(self.z, x, y)
        if arr is None:
            return 0
        rx, ry = int((xt - x) * 255), int((yt - y) * 255)
        r, g, b = arr[min(ry, 255), min(rx, 255)]
        return -10000 + (r * 256 * 256 + g * 256 + b) * 0.1

# ---- simple helpers ----
def smooth(a, win, dx):
    if len(a) < 3: return a
    n = max(3, int(round(win / dx)))
    n = n + 1 if n % 2 == 0 else n
    k = np.ones(n) / n
    return np.convolve(a, k, "same")

def parse(file):
    if file.name.lower().endswith(".kmz"):
        with zipfile.ZipFile(file) as z:
            kml = [n for n in z.namelist() if n.endswith(".kml")][0]
            xml = z.read(kml)
    else:
        xml = file.read()
    root = ET.fromstring(xml)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    agms, lines = [], []
    for pm in root.findall(".//kml:Folder[kml:name='AGMs']/kml:Placemark", ns):
        name = pm.findtext("kml:name", "", ns).strip()
        if name.startswith("SP"): continue
        c = pm.find(".//kml:coordinates", ns)
        if c is not None:
            vals = c.text.strip().split(",")
            if len(vals) >= 2:
                agms.append((name, Point(float(vals[0]), float(vals[1]))))
    for ln in root.findall(".//kml:Folder[kml:name='CENTERLINE']//kml:LineString", ns):
        c = ln.find("kml:coordinates", ns)
        if c is None: continue
        pts = [(float(a), float(b)) for a,b,*_ in (x.split(",") for x in c.text.strip().split()) if len(a)>0]
        if pts: lines.append(LineString(pts))
    return agms, lines

# ---- Streamlit UI ----
st.set_page_config("Terrain AGM Distance", layout="wide")
st.title("📏 Terrain-Aware AGM Distance Calculator")
u = st.file_uploader("Upload KML/KMZ", type=["kml","kmz"])
if not u: st.stop()

agms, lines = parse(u)
st.text(f"{len(agms)} AGMs | {len(lines)} centerline part(s)")
if not agms or not lines:
    st.warning("Need both AGMs and CENTERLINE folders.")
    st.stop()

xf_fwd = Transformer.from_crs("epsg:4326","epsg:3857",always_xy=True)
xf_inv = Transformer.from_crs("epsg:3857","epsg:4326",always_xy=True)
parts_m=[]
for p in lines:
    if p.length>0:
        x,y=xf_fwd.transform(*p.xy)
        parts_m.append(LineString(list(zip(x,y))))

cache=TerrainCache(MAPBOX_TOKEN,MAPBOX_ZOOM)
rows,cum_mi,skipped=[],0.0,0
bar=st.progress(0); msg=st.empty()
total=len(agms)-1

for i in range(total):
    start=time.time()
    n1,a1=agms[i]; n2,a2=agms[i+1]
    msg.text(f"⏱ Calculating {n1} → {n2} ({i+1}/{total}) …")
    bar.progress((i+1)/total)
    p1=Point(xf_fwd.transform(a1.x,a1.y))
    p2=Point(xf_fwd.transform(a2.x,a2.y))
    part=parts_m[0]  # single centerline assumption
    s1,s2=part.project(p1),part.project(p2)
    s_lo,s_hi=sorted((s1,s2))
    if s_hi-s_lo<=0: skipped+=1; continue
    si=np.arange(s_lo,s_hi,RESAMPLE_M)
    if si.size==0 or si[-1]<s_hi: si=np.append(si,s_hi)
    x,y=np.asarray(part.xy[0]),np.asarray(part.xy[1])
    s=np.concatenate([[0],np.cumsum(np.hypot(np.diff(x),np.diff(y)))])
    xi,yi=np.interp(si,s,x),np.interp(si,s,y)
    xi,yi=smooth(xi,40,RESAMPLE_M),smooth(yi,40,RESAMPLE_M)
    lons,lats=xf_inv.transform(xi.tolist(),yi.tolist())
    pts=list(zip(lons,lats))
    elev=[cache.elev(lo,la) for lo,la in pts]
    elev=smooth(np.array(elev),40,SPACING_M)
    dist=0.0
    for j in range(len(pts)-1):
        lon1,lat1=pts[j]; lon2,lat2=pts[j+1]
        _,_,dxy=GEOD.inv(lon1,lat1,lon2,lat2)
        dz=elev[j+1]-elev[j]
        if abs(dz)<DZ_THRESH: dz=0
        dist+=math.hypot(dxy,dz)
    ft,mi=dist*FT_PER_M,dist*FT_PER_M/5280
    cum_mi+=mi
    rows.append({"From AGM":n1,"To AGM":n2,"Feet":round(ft,1),
                 "Miles":round(mi,4),"Cumulative":round(cum_mi,4)})
    if time.time()-start>10:
        print(f"⚠ Segment {n1}->{n2} took >10 s – skipped.")
        break

msg.text("✅ Done.")
bar.progress(1.0)
df=pd.DataFrame(rows)
st.dataframe(df,use_container_width=True)
st.download_button("Download CSV",
    df.to_csv(index=False).encode(),
    "terrain_distances.csv","text/csv")
