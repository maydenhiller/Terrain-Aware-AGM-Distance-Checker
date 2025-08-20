# --- AGM Terrain-Aware Chainage Calculator Streamlit App ---
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import zipfile
from fastkml import kml
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import linemerge
import requests
import re

# ---- SECTION 1: Streamlit UI Setup ----
st.set_page_config(page_title="AGM Terrain-Aware Chainage Calculator", layout="wide")
st.title("AGM Terrain-Aware Chainage Calculator")
st.markdown(
    """
    Upload a KML or KMZ file containing AGMs (Points) with names (e.g., 'AGM 000', 'AGM 010', etc.) and a centerline as a LineString.
    The app will calculate terrain-aware (3D) distances between AGMs, output a rebased chainage table, and allow you to download results as CSV.
    """
)

# ---- SECTION 2: File Upload ----
uploaded_file = st.file_uploader(
    "Upload your KML or KMZ file",
    type=["kml", "kmz"]
)

if not uploaded_file:
    st.info("Please upload a .kml or .kmz file to begin.")
    st.stop()

# ---- SECTION 3: Extract KML Data from (KML/KMZ) ----
def extract_kml_text(filelike, ext):
    if ext.lower() == ".kmz":
        try:
            with zipfile.ZipFile(filelike) as zf:
                # Find the first .kml file in the archive
                kml_fname = next((f for f in zf.namelist() if f.endswith(".kml")), None)
                if not kml_fname:
                    raise ValueError("No .kml file found inside KMZ.")
                return zf.read(kml_fname).decode("utf-8")
        except Exception as e:
            st.error(f"Failed to extract .kml from .kmz: {e}")
            st.stop()
    else:
        # Assume KML is text
        return filelike.read().decode("utf-8")

file_ext = "." + uploaded_file.name.rsplit(".", 1)[-1].lower()
kml_text = extract_kml_text(uploaded_file, file_ext)

# ---- SECTION 4: KML Parsing Utilities ----

def parse_kml_features(kml_text):
    """Parse KML, return lists of (AGM Placemark) and the centerline coordinates."""
    k = kml.KML()
    try:
        k.from_string(kml_text)
    except Exception as e:
        st.error(f"KML parse error: {e}")
        st.stop()

    agm_points = []
    centerlines = []

    # Recursive function to visit Placemarks at all levels
    def visit_feature(feat):
        for f in getattr(feat, 'features', lambda: [])():
            if isinstance(f, kml.Placemark):
                geom = f.geometry
                if hasattr(geom, 'geom_type'):
                    if geom.geom_type == 'Point':
                        name = getattr(f, 'name', None)
                        point = (name, (geom.y, geom.x))  # (lat, lon)
                        agm_points.append(point)
                    elif geom.geom_type in ('LineString', 'MultiLineString'):
                        centerlines.append(geom)
            else:
                visit_feature(f)
    for feat in k.features():
        visit_feature(feat)
    return agm_points, centerlines

agm_list, centerlines = parse_kml_features(kml_text)

# ---- Validate AGMs and Centerlines ----
if len(agm_list) < 2:
    st.error("Less than two AGMs were detected in the file. At least two AGMs are required.")
    st.stop()
if len(centerlines) == 0:
    st.error("No centerline (LineString or MultiLineString) found in KML.")
    st.stop()

# ---- SECTION 5: AGM Processing: Sort, Clean, Prepare ----

def parse_agm_number(name):
    """Return integer AGM number from AGM name like 'AGM 000', 'AGM010', etc."""
    if not name:
        return None
    m = re.search(r"AGM\s*0*(\d+)", name.upper())
    if m: return int(m.group(1))
    # Fallback: look for integer in name
    m = re.search(r"0*(\d+)", name)
    if m: return int(m.group(1))
    return None

agm_processed = []
for name, (lat, lon) in agm_list:
    num = parse_agm_number(name)
    if num is not None:
        agm_processed.append({"name": name, "num": num, "lat": lat, "lon": lon})
if len(agm_processed) < 2:
    st.error("Could not reliably extract AGM numbers from AGM Points. Check naming (AGM labels required).")
    st.stop()
# Sort AGMs by number (e.g., 0, 10, 20, ...)
agm_processed = sorted(agm_processed, key=lambda x: x["num"])

# Convert to DataFrame for later
agm_df = pd.DataFrame(agm_processed)

# ---- SECTION 6: Centerline Geometry Preparation ----

# Merge (possibly multiple) centerline geometries into a single LineString
centerline_geoms = []
for geom in centerlines:
    if hasattr(geom, "coords"):
        centerline_geoms.append(LineString(list(geom.coords)))
    elif isinstance(geom, MultiLineString):
        for l in geom.geoms:
            centerline_geoms.append(LineString(list(l.coords)))
if not centerline_geoms:
    st.error("No valid centerline geometry found.")
    st.stop()

centerline_merged = linemerge(centerline_geoms)
if isinstance(centerline_merged, LineString):
    centerline_line = centerline_merged
else:
    # Multiple disconnected lines, take the longest one
    lines = list(centerline_merged.geoms)
    centerline_line = max(lines, key=lambda l: l.length)

cl_coords = list(centerline_line.coords)  # (lon,lat [,z]); KML order

# For 3D, if z (altitude) is present, it will be used; otherwise, elevation must be looked up.

# ---- SECTION 7: Snap AGMs to Centerline ----

from shapely.geometry import Point as ShapelyPoint

def project_point_onto_line(lat, lon, line):
    """Project AGM (lat, lon) onto centerline; returns distance along (fraction) and projected point."""
    pt = ShapelyPoint(lon, lat)
    proj_dist = line.project(pt)
    snapped = line.interpolate(proj_dist)
    return proj_dist, snapped

agm_chain = []
for idx, row in agm_df.iterrows():
    dist, snapped = project_point_onto_line(row["lat"], row["lon"], centerline_line)
    agm_chain.append({
        "name": row["name"],
        "num": row["num"],
        "lat": row["lat"],
        "lon": row["lon"],
        "centerline_dist": dist,
        "cl_snap_lat": snapped.y,
        "cl_snap_lon": snapped.x
    })
agm_chain_sorted = sorted(agm_chain, key=lambda x: x["centerline_dist"])
agm_snap_df = pd.DataFrame(agm_chain_sorted).reset_index(drop=True)

# ---- SECTION 8: Elevation Data Function ----

# Caching decorator for API efficiency
@st.cache_data(show_spinner=False)
def get_elevations(latlons):
    """Batch query Open-Elevation for elevations (meters) given list of (lat, lon)."""
    if len(latlons) == 0:
        return []
    url = "https://api.open-elevation.com/api/v1/lookup"
    # Limit batch size due to service restrictions (use batch of 100)
    elevations = []
    batch_size = 100
    for i in range(0, len(latlons), batch_size):
        coords = latlons[i:i+batch_size]
        locations = [{"latitude": lat, "longitude": lon} for (lat, lon) in coords]
        try:
            resp = requests.post(
                url, json={"locations": locations},
                timeout=10
            )
            resp.raise_for_status()
            rets = resp.json().get("results", [])
            elevations.extend([r.get("elevation", None) for r in rets])
        except Exception as e:
            st.warning(f"Elevation API batch failed: {e}")
            elevations.extend([None]*len(coords))
    return elevations

# ---- SECTION 9: AGM-to-AGM Segmental Chainage Calculation ----

st.header("Calculating Terrain-Aware Chainages")
st.write("This may take a minute for a long centerline.")

def sample_points_along_line(line, s_start, s_end, interval_feet=30):
    """Return list of points (lat, lon) along `line` between start/end, evenly spaced by interval_feet."""
    distance = abs(s_end - s_start)
    interval_m = interval_feet * 0.3048  # convert feet to meters
    n_samples = max(2, int(np.ceil(distance / interval_m)))  # at least 2 points
    fractions = np.linspace(s_start, s_end, n_samples)
    coords = [line.interpolate(f).coords[0] for f in fractions]
    # coords: list of (lon, lat, [z])
    return [(c[1], c[0]) for c in coords]  # (lat, lon)

segment_data = []
total_distance_ft = 0.0
progressbar = st.progress(0, text="Processing segments...")

for i in range(agm_snap_df.shape[0] - 1):
    r1 = agm_snap_df.iloc[i]
    r2 = agm_snap_df.iloc[i+1]
    # Projected centerline positions (distance units in centerline's CRS, likely degrees or meters, but treat as arc length)
    s1, s2 = r1["centerline_dist"], r2["centerline_dist"]
    pts = sample_points_along_line(centerline_line, s1, s2, interval_feet=30)
    elevs = get_elevations(pts)
    # For any response gaps default to 0
    elevs = [e if e is not None else 0 for e in elevs]
    # Compute 3D segment distances
    dist_sum = 0.0
    for j in range(1, len(pts)):
        lat1, lon1, z1 = pts[j-1][0], pts[j-1][1], elevs[j-1]
        lat2, lon2, z2 = pts[j][0], pts[j][1], elevs[j]
        # Geodesic (surface) distance
        r = 6371000  # meters, earth radius
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        c = 2*np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        planar = r * c   # meters
        dz = (z2 - z1)
        dist = np.sqrt(planar**2 + dz**2)  # meters
        dist_sum += dist
    dist_ft = dist_sum * 3.28084  # meters to feet
    total_distance_ft += dist_ft
    segment_str = f"{str(r1['num']).zfill(3)} to {str(r2['num']).zfill(3)}"
    segment_data.append({
        'Segment': segment_str,
        'Distance (ft)': int(round(dist_ft)),
        'Distance (mi)': round(dist_ft/5280.0, 4),
        'Total Distance So Far (ft)': int(round(total_distance_ft))
    })
    progressbar.progress((i+1)/(agm_snap_df.shape[0] - 1))

progressbar.empty()

if len(segment_data) == 0:
    st.warning("No AGM-to-AGM segments found (are there at least two AGMs?)")
    st.stop()

results_df = pd.DataFrame(segment_data)
st.success("Chainage calculation complete!")

# ---- SECTION 10: Present Results Table ----

st.header("AGM-to-AGM Terrain-Aware Chainage Table")
st.dataframe(
    results_df,
    hide_index=True,
    use_container_width=True
)

# ---- SECTION 11: Download as CSV ----
@st.cache_data(show_spinner=False)
def make_csv(df):
    return df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Chainage Table as CSV",
    data=make_csv(results_df),
    file_name="terrain_aware_chainage.csv",
    mime="text/csv",
    help="Download the table as a CSV file."
)
