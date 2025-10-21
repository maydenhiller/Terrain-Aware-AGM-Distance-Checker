import io, math, zipfile, requests, xml.etree.ElementTree as ET
import streamlit as st, pandas as pd, numpy as np
from PIL import Image
from shapely.geometry import Point, LineString
from shapely.ops import substring
from pyproj import Transformer

# --- CONFIG ---
MAPBOX_TOKEN = st.secrets["mapbox"]["token"]
TERRAIN_TILE_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"

to_merc = Transformer.from_crs("epsg:4326","epsg:3857",always_xy=True)
to_geo  = Transformer.from_crs("epsg:3857","epsg:4326",always_xy=True)

# --- PARSING ---
def parse_kml_kmz(uploaded_file):
    if uploaded_file.name.endswith(".kmz"):
        with zipfile.ZipFile(uploaded_file) as zf:
            kml_file = next((f for f in zf.namelist() if f.endswith(".kml")),None)
            with zf.open(kml_file) as f: kml_data = f.read()
    else:
        kml_data = uploaded_file.read()
    root = ET.fromstring(kml_data)
    ns={"kml":"http://www.opengis.net/kml/2.2"}
    agms, centerline = [], None
    for folder in root.findall(".//kml:Folder",ns):
        name_el = folder.find("kml:name",ns)
        if name_el is None: continue
        fname = name_el.text.strip().lower()
        if fname=="agms":
            for pm in folder.findall("kml:Placemark",ns):
                pname=pm.find("kml:name",ns); coords=pm.find(".//kml:coordinates",ns)
                if pname is None or coords is None: continue
                try:
                    lon,lat,*_=map(float,coords.text.strip().split(","))
                    agms.append((pname.text.strip(),Point(lon,lat)))
                except: continue
        elif fname=="centerline":
            for pm in folder.findall("kml:Placemark",ns):
                coords=pm.find(".//kml:coordinates",ns)
                if coords is None: continue
                pts=[tuple(map(float,p.split(",")[:2])) for p in coords.text.strip().split()]
                if len(pts)>=2: centerline=LineString(pts)
    return agms,centerline

# --- PROJECTION HELPERS ---
def to_merc_point(pt): x,y=to_merc.transform(pt.x,pt.y); return Point(x,y)
def to_merc_line(line): return LineString([to_merc.transform(x,y) for (x,y) in line.coords])
def to_geo_point(pt): lon,lat=to_geo.transform(pt.x,pt.y); return Point(lon,lat)

# --- SLICING & INTERPOLATION ---
def slice_centerline(line_m,p1_m,p2_m):
    d1,d2=line_m.project(p1_m),line_m.project(p2_m)
    if d1==d2: return None
    start,end=(d1,d2) if d1<d2 else (d2,d1)
    seg=substring(line_m,start,end,normalized=False)
    return seg if seg and seg.length>0 else None

def interpolate_line(line_m,spacing=1.0):
    total=line_m.length; steps=max(int(total/spacing),1)
    pts=[line_m.interpolate(i*spacing) for i in range(steps)]
    pts.append(line_m.interpolate(total))
    return pts

# --- Mapbox Terrain-RGB ---
def lonlat_to_tile(lon,lat,z):
    n=2**z; x=(lon+180)/360*n
    y=(1-math.log(math.tan(math.radians(lat))+1/math.cos(math.radians(lat)))/math.pi)/2*n
    return int(x),int(y),x,y
def pixel_in_tile(x_tile,y_tile,x_float,y_float):
    x_pix=int((x_float-x_tile)*256); y_pix=int((y_float-y_tile)*256)
    return max(0,min(255,x_pix)),max(0,min(255,y_pix))
def decode_rgb(r,g,b): return -10000+(r*256*256+g*256+b)*0.1

class TerrainCache:
    def __init__(self,token,zoom=15): self.t=token; self.z=zoom; self.cache={}
    def tile(self,z,x,y):
        key=(z,x,y)
        if key in self.cache: return self.cache[key]
        url=TERRAIN_TILE_URL.format(z=z,x=x,y=y)
        r=requests.get(url,params={"access_token":self.t},timeout=20)
        if r.status_code!=200: return None
        img=Image.open(io.BytesIO(r.content)).convert("RGB")
        self.cache[key]=img; return img
    def elevation(self,lon,lat):
        z=self.z; x_tile,y_tile,x_f,y_f=lonlat_to_tile(lon,lat,z)
        x_pix,y_pix=pixel_in_tile(x_tile,y_tile,x_f,y_f)
        img=self.tile(z,x_tile,y_tile)
        if img is None: return 0.0
        r,g,b=img.getpixel((x_pix,y_pix))
        return decode_rgb(r,g,b)

def get_elevations(points_m,cache):
    elevs=[]
    for pm in points_m:
        lonlat=to_geo_point(pm)
        elev=cache.elevation(lonlat.x,lonlat.y)
        elevs.append(float(elev))
    return elevs

# --- DISTANCES ---
def dist3d(p1,p2,e1,e2):
    dx,dy=p2.x-p1.x,p2.y-p1.y; dz=e2-e1
    return math.sqrt(dx*dx+dy*dy+dz*dz)

# --- STREAMLIT UI ---
st.title("Terrain-Aware AGM Distance Calculator (Name-order)")

uploaded=st.file_uploader("Upload KML or KMZ",type=["kml","kmz"])
if uploaded:
    agms_ll,cl_ll=parse_kml_kmz(uploaded)
    st.text(f"AGMs: {len(agms_ll)} | Centerline: {'found' if cl_ll else 'missing'}")
    if cl_ll and len(agms_ll)>=2:
        cl_m=to_merc_line(cl_ll)
        agms_m=[(n,to_merc_point(pt)) for n,pt in agms_ll]

        # Keep AGM order as given in file (name order)
        rows=[]; cum_mi=0; skipped=0
        cache=TerrainCache(MAPBOX_TOKEN,zoom=15)

        for i in range(len(agms_m)-1):
            n1,p1=agms_m[i]; n2,p2=agms_m[i+1]
            seg=slice_centerline(cl_m,p1,p2)
            if not seg: skipped+=1; continue
            pts=interpolate_line(seg,1.0)
            if len(pts)<2: skipped+=1; continue
            elevs=get_elevations(pts,cache)
            d2d=seg.length
            d3d=sum(dist3d(pts[j],pts[j+1],elevs[j],elevs[j+1]) for j in range(len(pts)-1))
            d2d_mi=d2d/1609.34; d3d_mi=d3d/1609.34
            cum_mi+=d3d_mi
            rows.append({
                "From AGM":n1,"To AGM":n2,
                "2D miles":round(d2d_mi,6),
                "3D miles":round(d3d_mi,6),
                "Ratio 3D/2D":round(d3d_mi/d2d_mi if d2d_mi>0 else 0,3),
                "Cumulative 3D miles":round(cum_mi,6)
            })
        st.dataframe(pd.DataFrame(rows))
        st.text(f"Skipped: {skipped}")
        csv=pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV",csv,"terrain_distances.csv","text/csv")
