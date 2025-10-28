# app.py — Detect CENTERLINE in Folder, Document, or Placemark names

import io, math, zipfile, re, xml.etree.ElementTree as ET
import numpy as np, pandas as pd, requests, streamlit as st
from PIL import Image
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import linemerge
from pyproj import Geod, Transformer

st.set_page_config("Terrain AGM Distance — Geodesic", layout="wide")
st.title("📏 Terrain-Aware AGM Distance Calculator — Centerline-Aware Parser")

MAPBOX_TOKEN = "pk.eyJ1IjoibWF5ZGVuaGlsbGVyIiwiYSI6ImNtZ2ljMnN5ejA3amwyam9tNWZnYnZibWwifQ.GXoTyHdvCYtr7GvKIW9LPA"
RESAMPLE_M, SMOOTH_WINDOW_M, ELEV_DZ_THRESHOLD, MAPBOX_ZOOM = 25, 50, 0.25, 14
FT_PER_M, MAX_SNAP_M = 3.28084, 120
GEOD = Geod(ellps="WGS84")
TERRAIN_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"

# --- Terrain cache identical to before ---
class TerrainRGBCache:
    def __init__(self, token, zoom): self.token, self.z, self.cache = token, zoom, {}
    @staticmethod
    def lonlat_to_tile(lon, lat, z):
        n=2**z; xt=(lon+180)/360*n; lat_r=np.radians(lat)
        yt=(1-np.log(np.tan(lat_r)+1/np.cos(lat_r))/math.pi)/2*n
        return xt,yt
    def fetch_tile(self,x,y):
        k=(self.z,x,y)
        if k in self.cache: return self.cache[k]
        url=TERRAIN_URL.format(z=self.z,x=x,y=y)
        r=requests.get(url,params={"access_token":self.token},timeout=12)
        if r.status_code!=200: self.cache[k]=None; return None
        self.cache[k]=np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"),dtype=np.uint8)
        return self.cache[k]
    def elevations_bulk(self,lons,lats):
        z=self.z; xt,yt=self.lonlat_to_tile(lons,lats,z)
        tx,ty=np.floor(xt).astype(int),np.floor(yt).astype(int)
        xp,yp=(xt-tx)*255,(yt-ty)*255
        out=np.zeros_like(lons,dtype=float)
        keys,inv=np.unique(np.stack([tx,ty],1),axis=0,return_inverse=True)
        for i,(x_tile,y_tile) in enumerate(keys):
            arr=self.fetch_tile(int(x_tile),int(y_tile))
            if arr is None: continue
            m=inv==i
            x0,y0=np.clip(xp[m].astype(int),0,254),np.clip(yp[m].astype(int),0,254)
            x1,y1=x0+1,y0+1; dx,dy=xp[m]-x0,yp[m]-y0
            p=lambda A:-10000+(A[:,0]*65536+A[:,1]*256+A[:,2])*0.1
            e00,e10,e01,e11=p(arr[y0,x0]),p(arr[y0,x1]),p(arr[y1,x0]),p(arr[y1,x1])
            out[m]=e00*(1-dx)*(1-dy)+e10*dx*(1-dy)+e01*(1-dx)*dy+e11*dx*dy
        return out

# --- XML cleaner (same as before, removes bad prefixes) ---
def clean_xml(raw):
    if isinstance(raw,bytes): raw=raw.decode("utf-8","ignore")
    raw=re.sub(r"[^\x09\x0A\x0D\x20-\x7E]","",raw)
    raw=re.sub(r"\bxsi:|gx:|kml:|xmlns:gx=\"[^\"]*\"|xmlns:xsi=\"[^\"]*\"","",raw)
    raw=re.sub(r"xsi:schemaLocation=\"[^\"]*\"","",raw)
    raw=re.sub(r"<href>[^<>\s]+ [^<]*</href>","",raw)
    start,end=raw.find("<"),raw.rfind(">")
    return raw[start:end+1]
def strip_ns(e):
    e.tag=e.tag.split("}",1)[-1]
    e.attrib={k.split("}",1)[-1]:v for k,v in e.attrib.items()}
    for c in list(e): strip_ns(c)
def safe_parse_kml(raw):
    xml=clean_xml(raw)
    root=ET.fromstring(xml)
    strip_ns(root)
    return root

# --- Parse KML/KMZ: now finds CENTERLINE anywhere ---
def parse_kml_kmz(u):
    raw = zipfile.ZipFile(u).read(next(n for n in zipfile.ZipFile(u).namelist() if n.lower().endswith(".kml"))) \
          if u.name.lower().endswith(".kmz") else u.read()
    try: root=safe_parse_kml(raw)
    except Exception as e: st.error(f"Parse fail: {e}"); return [],[]
    agms,lines=[],[]

    # collect ALL candidates with <name>CENTERLINE</name> under any tag
    centerline_containers=[n for n in root.iter() if (n.find("name") is not None and
                                                      "CENTERLINE" in n.findtext("name","").upper())]

    # fallback: any LineString if no explicit CENTERLINE container
    if not centerline_containers:
        centerline_containers=[root]

    for el in root.iter("Placemark"):
        nm=el.findtext("name","").strip()
        if not nm: continue
        if re.match(r"^\d{2,}$",nm):  # numeric AGMs like 010,020,...
            coord=el.findtext(".//Point/coordinates","")
            try: lon,lat=map(float,coord.split(",")[:2]); agms.append((nm,Point(lon,lat)))
            except: pass

    for container in centerline_containers:
        for coords in container.findall(".//LineString/coordinates"):
            pts=[]
            for pair in coords.text.strip().split():
                try: lon,lat=map(float,pair.split(",")[:2]); pts.append((lon,lat))
                except: pass
            if len(pts)>=2: lines.append(LineString(pts))

    if len(lines)>1:
        merged=linemerge(MultiLineString(lines))
        lines=[merged] if isinstance(merged,LineString) else list(merged.geoms)
    agms.sort(key=lambda p:int(re.sub(r"\D","",p[0])) if re.search(r"\d",p[0]) else -1)
    return agms,lines

# --- Math helpers ---
def build_local_meter_crs(line_ll):
    xs,ys=zip(*line_ll.coords); lon0,lat0=np.mean(xs),np.mean(ys)
    proj=f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m"
    to_m=Transformer.from_crs("EPSG:4326",proj,always_xy=True)
    to_ll=Transformer.from_crs(proj,"EPSG:4326",always_xy=True)
    return to_m,to_ll
def choose_best_centerline(lines,p1,p2):
    best,best_d=None,1e12
    for ln in lines:
        to_m,_=build_local_meter_crs(ln)
        ln_m=LineString(zip(*to_m.transform(*zip(*ln.coords))))
        p1_m,p2_m=Point(*to_m.transform(p1.x,p1.y)),Point(*to_m.transform(p2.x,p2.y))
        d=p1_m.distance(ln_m)+p2_m.distance(ln_m)
        if d<best_d: best_d,best=d,ln
    return best
def moving_average(v,w,s):
    if len(v)<3: return v
    n=max(3,int(round(w/max(s,1e-6)))); n+=n%2==0
    return np.convolve(v,np.ones(n)/n,"same")

# --- Streamlit main ---
u=st.file_uploader("Upload KML/KMZ",type=["kml","kmz"])
if not u: st.stop()
agms,lines=parse_kml_kmz(u)
st.text(f"{len(agms)} AGMs | {len(lines)} centerline part(s)")
if not agms or not lines: st.warning("Need both AGMs and CENTERLINE."); st.stop()

cache=TerrainRGBCache(MAPBOX_TOKEN,MAPBOX_ZOOM)
rows,cum_mi=[],0.0; bar,status=st.progress(0.0),st.empty()
for i in range(len(agms)-1):
    n1,a1=agms[i]; n2,a2=agms[i+1]
    status.text(f"⏱ {n1} → {n2} ({i+1}/{len(agms)-1})")
    ln=choose_best_centerline(lines,a1,a2)
    if not ln: continue
    to_m,to_ll=build_local_meter_crs(ln)
    ln_m=LineString(zip(*to_m.transform(*zip(*ln.coords))))
    p1_m,p2_m=Point(*to_m.transform(a1.x,a1.y)),Point(*to_m.transform(a2.x,a2.y))
    if p1_m.distance(ln_m)>MAX_SNAP_M or p2_m.distance(ln_m)>MAX_SNAP_M: continue
    s1,s2=ln_m.project(p1_m),ln_m.project(p2_m)
    s_start,s_end=sorted((s1,s2))
    count=max(2,int(round((s_end-s_start)/RESAMPLE_M))+1)
    si=np.linspace(s_start,s_end,count)
    pts_m=[ln_m.interpolate(s) for s in si]
    pts_ll=np.array([to_ll.transform(p.x,p.y) for p in pts_m])
    elev=moving_average(cache.elevations_bulk(pts_ll[:,0],pts_ll[:,1]),SMOOTH_WINDOW_M,RESAMPLE_M)
    dist=sum(math.hypot(GEOD.inv(*pts_ll[j],*pts_ll[j+1])[2], elev[j+1]-elev[j]
                        if abs(elev[j+1]-elev[j])>=ELEV_DZ_THRESHOLD else 0)
             for j in range(len(pts_ll)-1))
    feet, miles = dist*FT_PER_M, dist*FT_PER_M/5280; cum_mi+=miles
    rows.append({"From AGM":n1,"To AGM":n2,"Distance (feet)":round(feet,2),
                 "Distance (miles)":round(miles,6),"Cumulative (miles)":round(cum_mi,6)})
    bar.progress((i+1)/(len(agms)-1))
status.success("✅ Complete.")
df=pd.DataFrame(rows)
st.dataframe(df,use_container_width=True)
st.download_button("Download CSV",df.to_csv(index=False).encode("utf-8"),
                   "terrain_distances.csv","text/csv")
