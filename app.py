# app.py
# Terrain-Aware AGM Distance Calculator — GE 3D MATCH + ignore "SP" AGMs
import io, math, zipfile, requests, xml.etree.ElementTree as ET
import streamlit as st, pandas as pd, numpy as np
from PIL import Image
from shapely.geometry import Point, LineString
from pyproj import CRS, Transformer, Geod

st.set_page_config(page_title="Terrain-Aware AGM Distance Calculator", layout="wide")
st.title("Terrain-Aware AGM Distance Calculator (Google Earth 3D Match, SP Filtered)")

MAPBOX_TOKEN = st.secrets["mapbox"]["token"]
TERRAIN_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"
GEOD = Geod(ellps="WGS84")
FT_PER_M, MI_PER_FT = 3.28084, 1/5280.0

# ---- Calibrated constants ----
SPACING_M = 40.0
SIMPLIFY_TOL_M = 50.0
SMOOTH_WIN = 31
DZ_THRESHOLD_M = 0.75
MAX_SNAP_M = 80.0
MAPBOX_ZOOM = 17

# ─────────────────────────── Helpers
def strip_namespaces(elem):
    elem.tag = elem.tag.split('}', 1)[-1]
    for k in list(elem.attrib.keys()):
        nk = k.split('}', 1)[-1]
        if nk != k:
            elem.attrib[nk] = elem.attrib.pop(k)
    for c in list(elem):
        strip_namespaces(c)

# ─────────────────────────── Parse KML/KMZ (AGMs + CENTERLINE)
def agm_sort_key(p):
    n = p[0]; d = ''.join(filter(str.isdigit, n)); s = ''.join(filter(str.isalpha, n))
    return (int(d) if d else -1, s)

def parse_kml_kmz(upload):
    if upload.name.lower().endswith(".kmz"):
        with zipfile.ZipFile(upload) as z:
            name = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
            if not name: return [], []
            data = z.read(name)
    else:
        data = upload.read()

    root = ET.fromstring(data)
    strip_namespaces(root)
    agms, centerlines = [], []

    for folder in root.findall(".//Folder"):
        nm = folder.find("name")
        if nm is None or not nm.text:
            continue
        fname = nm.text.strip().upper()

        # ---- AGMs ----
        if fname == "AGMS":
            for pm in folder.findall(".//Placemark"):
                n = pm.find("name")
                if n is None or not n.text:
                    continue
                agm_name = n.text.strip()
                # skip any AGM whose name starts with "SP"
                if agm_name.upper().startswith("SP"):
                    continue
                c = pm.find(".//Point/coordinates")
                if c is None or not c.text:
                    continue
                txt = c.text.strip().split()[0]
                try:
                    lon, lat, *_ = map(float, txt.split(","))
                    agms.append((agm_name, Point(lon, lat)))
                except:
                    pass

        # ---- CENTERLINE ----
        elif fname == "CENTERLINE":
            for pm in folder.findall(".//Placemark"):
                c = pm.find(".//LineString/coordinates")
                if c is None or not c.text:
                    continue
                pts = []
                for pair in c.text.strip().split():
                    try:
                        lon, lat, *_ = map(float, pair.split(","))
                        pts.append((lon, lat))
                    except:
                        pass
                if len(pts) >= 2:
                    centerlines.append(LineString(pts))

    agms.sort(key=agm_sort_key)
    return agms, centerlines

# ─────────────────────────── Projections
def get_local_utm_crs(lines):
    xs=[x for l in lines for x,_ in l.coords]; ys=[y for l in lines for _,y in l.coords]
    cx,cy=np.mean(xs),np.mean(ys); zone=int((cx+180)/6)+1
    return CRS.from_epsg(32600+zone if cy>=0 else 32700+zone)
def xf_ll_to(crs): return Transformer.from_crs("EPSG:4326",crs,always_xy=True)
def xf_to_ll(crs): return Transformer.from_crs(crs,"EPSG:4326",always_xy=True)
def tf_line(l,xf): x,y=zip(*l.coords); X,Y=xf.transform(x,y); return LineString(zip(X,Y))
def tf_pt(p,xf): x,y=xf.transform(p.x,p.y); return Point(x,y)

# ─────────────────────────── Terrain-RGB
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
        lon=max(-179.9,min(179.9,lon)); lat=max(-85,min(85,lat))
        z=self.z; xt,yt,xf,yf=lonlat_to_tile(lon,lat,z)
        xp,yp=(xf-xt)*256,(yf-yt)*256
        x0,y0=int(xp),int(yp); dx,dy=xp-x0,yp-y0
        x0,y0=max(0,min(255,x0)),max(0,min(255,y0))
        x1,y1=min(x0+1,255),min(y0+1,255)
        arr=self.get(z,xt,yt)
        if arr is None: return 0.0
        p00=decode_rgb(*arr[y0,x0]); p10=decode_rgb(*arr[y0,x1])
        p01=decode_rgb(*arr[y1,x0]); p11=decode_rgb(*arr[y1,x1])
        return float(p00*(1-dx)*(1-dy)+p10*dx*(1-dy)+p01*(1-dx)*dy+p11*dx*dy)

def smooth_hann(vals, k):
    if k <= 1: return [float(v) for v in vals]
    k = int(k) if int(k) % 2 == 1 else int(k) + 1
    arr = np.asarray(vals, dtype=float)
    win = np.hanning(k); win /= win.sum()
    pad = k // 2
    arr_pad = np.pad(arr, (pad, pad), mode="reflect")
    sm = np.convolve(arr_pad, win, mode="same")[pad:-pad]
    return sm.tolist()

# ─────────────────────────── Geometry sampling / snapping
def build_arrays(ls):
    c=list(ls.coords); x=np.array([p[0] for p in c],float); y=np.array([p[1] for p in c],float)
    d=np.hypot(np.diff(x),np.diff(y)); cum=np.concatenate([[0.0],np.cumsum(d)])
    return x,y,cum
def interp(x,y,cum,s):
    if s<=0:return x[0],y[0]
    if s>=cum[-1]:return x[-1],y[-1]
    i=int(np.searchsorted(cum,s)-1); i=max(0,min(i,len(x)-2))
    seg=cum[i+1]-cum[i]; t=(s-cum[i])/seg if seg>0 else 0.0
    return x[i]+t*(x[i+1]-x[i]),y[i]+t*(y[i+1]-y[i])
def sample_between(ls,s1,s2,sp):
    x,y,cum=build_arrays(ls); s_lo,s_hi=sorted((float(s1),float(s2)))
    L=abs(s_hi-s_lo)
    if L<=0: return np.array([]),np.array([])
    steps=np.arange(0.0,L,sp)
    if steps.size==0 or steps[-1]<L: steps=np.append(steps,L)
    pts=[interp(x,y,cum,s_lo+d) for d in steps]
    X,Y=zip(*pts); return np.array(X),np.array(Y)
def choose_part(parts_m,p1_ll,p2_ll,xf,max_off):
    p1m,p2m=tf_pt(p1_ll,xf),tf_pt(p2_ll,xf); best=None
    for i,p in enumerate(parts_m):
        s1,s2=p.project(p1m),p.project(p2m)
        sp1,sp2=p.interpolate(s1),p.interpolate(s2)
        o1=((p1m.x-sp1.x)**2+(p1m.y-sp1.y)**2)**0.5
        o2=((p2m.x-sp2.x)**2+(p2m.y-sp2.y)**2)**0.5
        if o1<=max_off and o2<=max_off:
            tot=o1+o2
            if best is None or tot<best[0]: best=(tot,i,s1,s2)
    return (None,None,None) if best is None else (best[1],best[2],best[3])

# ─────────────────────────── UI + computation
u = st.file_uploader("Upload KML or KMZ", type=["kml", "kmz"])
if u:
    agms, parts = parse_kml_kmz(u)
    st.text(f"{len(agms)} AGMs | {len(parts)} centerline part(s)")
    if not agms or not parts:
        st.warning("Need both AGMs + centerline."); st.stop()

    crs = get_local_utm_crs(parts)
    xf_fwd, xf_inv = xf_ll_to(crs), xf_to_ll(crs)
    parts_m = []
    for p in parts:
        pm = tf_line(p, xf_fwd).simplify(SIMPLIFY_TOL_M, preserve_topology=False)
        if pm.length > 0 and len(pm.coords) >= 2:
            parts_m.append(pm)

    cache = TerrainCache(MAPBOX_TOKEN, MAPBOX_ZOOM)

    rows, cum_mi, skipped = [], 0.0, 0
    for i in range(len(agms) - 1):
        n1, a1 = agms[i]; n2, a2 = agms[i + 1]
        idx, s1, s2 = choose_part(parts_m, a1, a2, xf_fwd, MAX_SNAP_M)
        if idx is None:
            skipped += 1; continue

        Xs, Ys = sample_between(parts_m[idx], s1, s2, SPACING_M)
        if Xs.size < 2:
            skipped += 1; continue

        lons, lats = xf_inv.transform(Xs.tolist(), Ys.tolist())
        pts = list(zip(lons, lats))
        elev_raw = [cache.elev(lo, la) for lo, la in pts]
        elev = smooth_hann(elev_raw, SMOOTH_WIN)

        dist_m = 0.0
        for j in range(len(pts) - 1):
            lon1, lat1 = pts[j]; lon2, lat2 = pts[j + 1]
            _, _, dxy = GEOD.inv(lon1, lat1, lon2, lat2)
            dz = elev[j + 1] - elev[j]
            if abs(dz) < DZ_THRESHOLD_M: dz = 0.0
            dist_m += math.hypot(dxy, dz)

        ft = dist_m * FT_PER_M
        mi = ft * MI_PER_FT
        cum_mi += mi
        rows.append({
            "From AGM": n1,
            "To AGM": n2,
            "Distance (feet)": round(ft, 2),
            "Distance (miles)": round(mi, 6),
            "Cumulative (miles)": round(cum_mi, 6)
        })

    df = pd.DataFrame(rows)
    st.subheader("📊 Distance table")
    st.dataframe(df, use_container_width=True)
    st.text(f"Skipped segments: {skipped}")
    st.download_button("Download CSV",
                       df.to_csv(index=False).encode("utf-8"),
                       "terrain_distances.csv", "text/csv")
