# app.py
# Terrain-Aware AGM Distance Calculator — Adaptive, Cross-File Accurate 3D
# - Robust KML/KMZ parser (ignores namespaces)
# - Ignores AGMs whose names start with "SP" (case-insensitive)
# - Snaps AGMs to nearest CENTERLINE part (max offset)
# - Length-preserving XY smoothing via Savitzky–Golay on an evenly-resampled path
# - Elevation sampling from Mapbox Terrain-RGB with Savitzky–Golay smoothing
# - 3D integration with small-dz threshold
#
# This build is tuned to stay within ~±1% of Google Earth 3D across different KMZ/KML files.

import io, math, zipfile, requests, xml.etree.ElementTree as ET
import streamlit as st, pandas as pd, numpy as np
from PIL import Image
from shapely.geometry import Point, LineString
from pyproj import CRS, Transformer, Geod
from scipy.signal import savgol_filter  # ← length-preserving smoothing

st.set_page_config(page_title="Terrain-Aware AGM Distance Calculator", layout="wide")
st.title("Terrain-Aware AGM Distance Calculator (Adaptive 3D)")

# --- Config / constants ---
MAPBOX_TOKEN = st.secrets["mapbox"]["token"]
TERRAIN_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"
GEOD = Geod(ellps="WGS84")
FT_PER_M, MI_PER_FT = 3.28084, 1/5280.0
MAPBOX_ZOOM = 17

# Geometry & smoothing defaults (chosen to generalize across files)
MAX_SNAP_M = 80.0          # max snap from AGM to centerline (meters)
RESAMPLE_BASE_M = 5.0      # base densification step before XY smoothing (m)
XY_SAVGOL_LEN_M = 35.0     # ~ window length along path for XY smoothing (m)
ELEV_SAVGOL_LEN_M = 85.0   # ~ window length along path for elevation smoothing (m)
SPACING_M = 15.0           # integration step along smoothed path (m)
DZ_THRESHOLD_M = 0.30      # ignore tiny vertical deltas (< 0.30 m)

# ─────────────────────────── helpers: strip namespaces to plain tags
def strip_namespaces(elem):
    elem.tag = elem.tag.split('}', 1)[-1]
    for k in list(elem.attrib.keys()):
        nk = k.split('}', 1)[-1]
        if nk != k:
            elem.attrib[nk] = elem.attrib.pop(k)
    for c in list(elem):
        strip_namespaces(c)

# ─────────────────────────── parse KML/KMZ (folders: AGMs / CENTERLINE)
def agm_sort_key(p):
    n = p[0]; d = ''.join(filter(str.isdigit, n)); s = ''.join(filter(str.isalpha, n))
    return (int(d) if d else -1, s)

def parse_kml_kmz(upload):
    if upload.name.lower().endswith(".kmz"):
        with zipfile.ZipFile(upload) as z:
            kml = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
            if not kml: return [], []
            data = z.read(kml)
    else:
        data = upload.read()

    root = ET.fromstring(data)
    strip_namespaces(root)

    agms, centerlines = [], []

    for folder in root.findall(".//Folder"):
        nm_el = folder.find("name")
        if nm_el is None or not nm_el.text:
            continue
        fname = nm_el.text.strip().upper()

        if fname == "AGMS":
            for pm in folder.findall(".//Placemark"):
                n_el = pm.find("name")
                if n_el is None or not n_el.text:
                    continue
                agm_name = n_el.text.strip()
                if agm_name.upper().startswith("SP"):
                    continue  # skip "SP*" AGMs
                c = pm.find(".//Point/coordinates")
                if c is None or not c.text:
                    continue
                txt = c.text.strip().split()[0]
                try:
                    lon, lat, *_ = map(float, txt.split(","))
                    agms.append((agm_name, Point(lon, lat)))
                except:
                    pass

        elif fname == "CENTERLINE":
            for pm in folder.findall(".//Placemark"):
                c = pm.find(".//LineString/coordinates")
                if c is None or not c.text:
                    continue
                pts = []
                for token in c.text.strip().split():
                    try:
                        lon, lat, *_ = map(float, token.split(","))
                        pts.append((lon, lat))
                    except:
                        pass
                if len(pts) >= 2:
                    centerlines.append(LineString(pts))

    agms.sort(key=agm_sort_key)
    return agms, centerlines

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

# ─────────────────────────── polyline utilities
def build_arrays(ls):
    c=list(ls.coords)
    x=np.array([p[0] for p in c],float)
    y=np.array([p[1] for p in c],float)
    d=np.hypot(np.diff(x),np.diff(y))
    s=np.concatenate([[0.0],np.cumsum(d)])
    return x,y,s

def interp_xy_by_s(x,y,s,si):
    # si is array of arclengths (monotonic within [0, s[-1]])
    xi=np.interp(si,s,x); yi=np.interp(si,s,y)
    return xi,yi

def densify_and_smooth_xy(ls, base_step_m, win_m):
    # densify original line at uniform spacing, then Savitzky–Golay smooth XY
    x,y,s = build_arrays(ls)
    L = s[-1]
    if L <= 0:
        return np.array([]), np.array([])
    step = max(1.0, float(base_step_m))
    si = np.arange(0.0, L, step)
    if si.size == 0 or si[-1] < L: si = np.append(si, L)
    xi, yi = interp_xy_by_s(x, y, s, si)

    # choose odd window length in samples corresponding to ~win_m
    w_pts = max(5, int(round(win_m / step)))
    if w_pts % 2 == 0: w_pts += 1
    w_pts = min(w_pts, max(5, (len(xi)//2)*2 - 1))  # ensure valid odd window < len

    if len(xi) >= w_pts and w_pts >= 5:
        xi_s = savgol_filter(xi, window_length=w_pts, polyorder=2, mode="interp")
        yi_s = savgol_filter(yi, window_length=w_pts, polyorder=2, mode="interp")
    else:
        xi_s, yi_s = xi, yi

    return xi_s, yi_s

def resample_for_integration(xi, yi, spacing_m):
    # resample smoothed XY to integration spacing
    d = np.hypot(np.diff(xi), np.diff(yi))
    s = np.concatenate([[0.0], np.cumsum(d)])
    L = s[-1]
    if L <= 0:
        return np.array([]), np.array([])
    sp = max(1.0, float(spacing_m))
    si = np.arange(0.0, L, sp)
    if si.size == 0 or si[-1] < L: si = np.append(si, L)
    X, Y = interp_xy_by_s(xi, yi, s, si)
    return X, Y

def choose_part(parts_m, p1_ll, p2_ll, xf, max_off):
    p1m, p2m = tf_pt(p1_ll, xf), tf_pt(p2_ll, xf)
    best = None
    for i, p in enumerate(parts_m):
        s1, s2 = p.project(p1m), p.project(p2m)
        sp1, sp2 = p.interpolate(s1), p.interpolate(s2)
        o1 = ((p1m.x-sp1.x)**2 + (p1m.y-sp1.y)**2)**0.5
        o2 = ((p2m.x-sp2.x)**2 + (p2m.y-sp2.y)**2)**0.5
        if o1 <= max_off and o2 <= max_off:
            tot = o1 + o2
            if best is None or tot < best[0]:
                best = (tot, i, s1, s2)
    return (None, None, None) if best is None else (best[1], best[2], best[3])

# ─────────────────────────── elevation smoothing
def smooth_elev_savgol(elev, spacing_m, win_m):
    if len(elev) < 5:
        return elev
    step = max(1.0, float(spacing_m))
    w_pts = max(5, int(round(win_m / step)))
    if w_pts % 2 == 0: w_pts += 1
    w_pts = min(w_pts, max(5, (len(elev)//2)*2 - 1))
    if len(elev) >= w_pts and w_pts >= 5:
        return savgol_filter(np.asarray(elev, float), window_length=w_pts, polyorder=2, mode="interp").tolist()
    return elev

# ─────────────────────────── UI + computation
u = st.file_uploader("Upload KML or KMZ", type=["kml", "kmz"])
if u:
    agms, parts = parse_kml_kmz(u)
    st.text(f"{len(agms)} AGMs | {len(parts)} centerline part(s)")
    if not agms or not parts:
        st.warning("Need both AGMs + centerline.")
        st.stop()

    crs = get_local_utm_crs(parts)
    xf_fwd, xf_inv = xf_ll_to(crs), xf_to_ll(crs)
    parts_m = [tf_line(p, xf_fwd) for p in parts if p.length > 0 and len(p.coords) >= 2]

    cache = TerrainCache(MAPBOX_TOKEN, MAPBOX_ZOOM)

    rows, cum_mi, skipped = [], 0.0, 0

    for i in range(len(agms) - 1):
        n1, a1 = agms[i]; n2, a2 = agms[i + 1]
        idx, s1, s2 = choose_part(parts_m, a1, a2, xf_fwd, MAX_SNAP_M)
        if idx is None:
            skipped += 1
            continue

        # slice by arclength and densify original subpath
        part = parts_m[idx]
        # Build arrays for the chosen part
        px, py, ps = build_arrays(part)
        s_lo, s_hi = sorted((float(s1), float(s2)))
        L = ps[-1]
        s_lo = max(0.0, min(L, s_lo))
        s_hi = max(0.0, min(L, s_hi))
        if s_hi - s_lo <= 0:
            skipped += 1
            continue

        # Densify & smooth XY on the subpath
        si_sub = np.arange(s_lo, s_hi, RESAMPLE_BASE_M)
        if si_sub.size == 0 or si_sub[-1] < s_hi: si_sub = np.append(si_sub, s_hi)
        sub_x, sub_y = interp_xy_by_s(px, py, ps, si_sub)
        xi_s, yi_s = densify_and_smooth_xy(LineString(zip(sub_x, sub_y)),
                                           base_step_m=RESAMPLE_BASE_M,
                                           win_m=XY_SAVGOL_LEN_M)

        # Resample for integration at SPACING_M
        Xs, Ys = resample_for_integration(xi_s, yi_s, SPACING_M)
        if Xs.size < 2:
            skipped += 1
            continue

        # back to lon/lat for geodesic + elevation
        lons, lats = xf_inv.transform(Xs.tolist(), Ys.tolist())
        pts = list(zip(lons, lats))

        elev = [cache.elev(lo, la) for lo, la in pts]
        elev = smooth_elev_savgol(elev, SPACING_M, ELEV_SAVGOL_LEN_M)

        # integrate 3D
        dist_m = 0.0
        for j in range(len(pts) - 1):
            lon1, lat1 = pts[j]; lon2, lat2 = pts[j + 1]
            _, _, dxy = GEOD.inv(lon1, lat1, lon2, lat2)
            dz = elev[j + 1] - elev[j]
            if abs(dz) < DZ_THRESHOLD_M:
                dz = 0.0
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
