# Terrain-Aware AGM Distance Calculator — Google-Earth-Matched 3D v4
import io, math, zipfile, requests, xml.etree.ElementTree as ET
import streamlit as st, pandas as pd, numpy as np
from PIL import Image
from shapely.geometry import Point, LineString
from pyproj import CRS, Transformer, Geod

st.set_page_config(page_title="Terrain-Aware AGM Distance Calculator", layout="wide")
st.title("Terrain-Aware AGM Distance Calculator (3D Matched v4)")

MAPBOX_TOKEN = st.secrets["mapbox"]["token"]
TERRAIN_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"
GEOD = Geod(ellps="WGS84")
FT_PER_M, MI_PER_FT = 3.28084, 1/5280

# tuned constants
SPACING_M = 15.0
SIMPLIFY_TOL_M = 30.0
SMOOTH_WIN = 9
MAX_SNAP_M = 60.0
MAPBOX_ZOOM = 17

# ─────────────────────────── parse KML/KMZ
def agm_sort_key(p):
    n=p[0]; d=''.join(filter(str.isdigit,n)); s=''.join(filter(str.isalpha,n))
    return (int(d) if d else -1, s)

def parse_kml_kmz(f):
    if f.name.endswith(".kmz"):
        with zipfile.ZipFile(f) as z:
            kml=[n for n in z.namelist() if n.endswith(".kml")]
            if not kml: return [],[]
            data=z.read(kml[0])
    else: data=f.read()
    root=ET.fromstring(data)
    agms,cls=[],[]
    for fold in root.findall(".//{*}Folder"):
        nm=(fold.find("{*}name").text or "").strip().upper() if fold.find("{*}name") else ""
        if nm=="AGMS":
            for pm in fold.findall(".//{*}Placemark"):
                n=pm.find("{*}name"); c=pm.find(".//{*}coordinates")
                if n and c:
                    lon,lat,*_=map(float,c.text.strip().split(","))
                    agms.append((n.text.strip(),Point(lon,lat)))
        elif nm=="CENTERLINE":
            for pm in fold.findall(".//{*}Placemark"):
                c=pm.find(".//{*}coordinates")
                if not c or not c.text: continue
                pts=[]
                for pair in c.text.strip().split():
                    try: lon,lat,*_=map(float,pair.split(",")); pts.append((lon,lat))
                    except: pass
                if len(pts)>1: cls.append(LineString(pts))
    agms.sort(key=agm_sort_key)
    return agms,cls

# ─────────────────────────── projections
def get_local_utm_crs(lines):
    xs=[x for l in lines for x,_ in l.coords]; ys=[y for l in lines for _,y in l.coords]
    cx,cy=np.mean(xs),np.mean(ys); zone=int((cx+180)/6)+1
    return CRS.from_epsg(32600+zone if cy>=0 else 32700+zone)
def xf_ll_to(crs): return Transformer.from_crs("EPSG:4326",crs,always_xy=True)
def xf_to_ll(crs): return Transformer.from_crs(crs,"EPSG:4326",always_xy=True)
def tf_line(l,xf): x,y=zip(*l.coords); X,Y=xf.transform(x,y); return LineString(zip(X,Y))
def tf_pt(p,xf): x,y=xf.transform(p.x,p.y); return Point(x,y)

# ─────────────────────────── terrain
def lonlat_to_tile(lon,lat,z):
    n=2**z; x=(lon+180)/360*n
    y=(1-math.log(math.tan(math.radians(lat))+1/math.cos(math.radians(lat)))/math.pi)/2*n
    return int(x),int(y),x,y
def decode_rgb(r,g,b):
    R,G,B=int(r),int(g),int(b)
    return -10000+((R*256*256)+(G*256)+B)*0.1

class TerrainCache:
    def __init__(self,tok,zoom): self.t,self.z,self.c=tok,int(zoom),{}
    def get(self,z,x,y):
        k=(z,x,y)
        if k in self.c: return self.c[k]
        r=requests.get(TERRAIN_URL.format(z=z,x=x,y=y),
                       params={"access_token":self.t},timeout=20)
        if r.status_code!=200: return None
        arr=np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"),np.uint8)
        self.c[k]=arr; return arr
    def elev(self,lon,lat):
        lon=max(-179.9,min(179.9,float(lon))); lat=max(-85,min(85,float(lat)))
        z=self.z; xt,yt,xf,yf=lonlat_to_tile(lon,lat,z)
        xp,yp=(xf-xt)*256,(yf-yt)*256
        x0,y0=int(xp),int(yp); dx,dy=xp-x0,yp-y0
        x0,y0=max(0,min(255,x0)),max(0,min(255,y0))
        x1,y1=min(x0+1,255),min(y0+1,255)
        arr=self.get(z,xt,yt)
        if arr is None: return 0
        p00=decode_rgb(*arr[y0,x0]); p10=decode_rgb(*arr[y0,x1])
        p01=decode_rgb(*arr[y1,x0]); p11=decode_rgb(*arr[y1,x1])
        return float(p00*(1-dx)*(1-dy)+p10*dx*(1-dy)+p01*(1-dx)*dy+p11*dx*dy)

def smooth(vals,k):
    if k<=1: return vals
    arr=np.array(vals,float)
    ker=np.ones(k)/k
    return np.convolve(arr,ker,"same")

# ─────────────────────────── helpers
def build_arrays(ls):
    c=list(ls.coords); x=np.array([p[0] for p in c]); y=np.array([p[1] for p in c])
    d=np.hypot(np.diff(x),np.diff(y)); cum=np.concatenate([[0],np.cumsum(d)])
    return x,y,cum
def interp(x,y,cum,s):
    if s<=0:return x[0],y[0]
    if s>=cum[-1]:return x[-1],y[-1]
    i=int(np.searchsorted(cum,s)-1)
    seg=cum[i+1]-cum[i]; t=(s-cum[i])/seg if seg>0 else 0
    return x[i]+t*(x[i+1]-x[i]), y[i]+t*(y[i+1]-y[i])
def sample(ls,s1,s2,sp):
    x,y,cum=build_arrays(ls); s_lo,s_hi=sorted((s1,s2))
    L=abs(s_hi-s_lo); steps=np.arange(0,L,sp)
    if steps.size==0 or steps[-1]<L: steps=np.append(steps,L)
    pts=[interp(x,y,cum,s_lo+d) for d in steps]
    X,Y=zip(*pts); return np.array(X),np.array(Y)
def choose_part(parts,p1,p2,xf,max_off):
    p1m,p2m=tf_pt(p1,xf),tf_pt(p2,xf); best=None
    for i,p in enumerate(parts):
        s1,s2=p.project(p1m),p.project(p2m)
        sp1,sp2=p.interpolate(s1),p.interpolate(s2)
        o1=((p1m.x-sp1.x)**2+(p1m.y-sp1.y)**2)**0.5
        o2=((p2m.x-sp2.x)**2+(p2m.y-sp2.y)**2)**0.5
        if o1<=max_off and o2<=max_off:
            tot=o1+o2
            if best is None or tot<best[0]: best=(tot,i,s1,s2)
    return (None,None,None) if best is None else (best[1],best[2],best[3])

# ─────────────────────────── main
u=st.file_uploader("Upload KML or KMZ",type=["kml","kmz"])
if u:
    agms,parts=parse_kml_kmz(u)
    st.text(f"{len(agms)} AGMs | {len(parts)} centerline part(s)")
    if not agms or not parts:
        st.warning("Need both AGMs + centerline."); st.stop()

    crs=get_local_utm_crs(parts)
    xf_fwd,xf_inv=xf_ll_to(crs),xf_to_ll(crs)
    parts_m=[tf_line(p,xf_fwd).simplify(SIMPLIFY_TOL_M,False) for p in parts]
    cache=TerrainCache(MAPBOX_TOKEN,MAPBOX_ZOOM)

    rows=[]; cum_mi=0; skip=0
    for i in range(len(agms)-1):
        n1,a1=agms[i]; n2,a2=agms[i+1]
        idx,s1,s2=choose_part(parts_m,a1,a2,xf_fwd,MAX_SNAP_M)
        if idx is None: skip+=1; continue
        Xs,Ys=sample(parts_m[idx],s1,s2,SPACING_M)
        if len(Xs)<2: skip+=1; continue
        lons,lats=xf_inv.transform(Xs.tolist(),Ys.tolist())
        pts=list(zip(lons,lats))
        elev=smooth([cache.elev(lo,la) for lo,la in pts],SMOOTH_WIN)

        dist_m=0
        for j in range(len(pts)-1):
            lon1,lat1=pts[j]; lon2,lat2=pts[j+1]
            _,_,dxy=GEOD.inv(lon1,lat1,lon2,lat2)
            dz=elev[j+1]-elev[j]
            dist_m+=math.hypot(dxy,dz)
        ft=dist_m*FT_PER_M; mi=ft*MI_PER_FT; cum_mi+=mi
        rows.append({"From AGM":n1,"To AGM":n2,
                     "Feet":round(ft,2),"Miles":round(mi,6),
                     "Cumulative":round(cum_mi,6)})
    df=pd.DataFrame(rows)
    st.dataframe(df,use_container_width=True)
    st.text(f"Skipped segments: {skip}")
    st.download_button("Download CSV",
                       df.to_csv(index=False).encode(),
                       "terrain_distances.csv","text/csv")
