# Terrain-Aware AGM Distance Calculator via Streamlit
# Author: [Your Name]
# Date: [Current Date]

import streamlit as st
import zipfile
from io import BytesIO
import pandas as pd
import numpy as np
from pykml import parser
from shapely.geometry import Point, LineString
from math import radians, cos, sin, sqrt, atan2

# -- Helper Functions --

def kml_to_root(kml_bytes):
    """Parse KML bytes and return the root element."""
    try:
        root = parser.fromstring(kml_bytes)
        return root
    except Exception as e:
        st.error(f"KML parsing failed: {e}")
        return None

def extract_agms_and_centerline(kml_root):
    """Extract AGM Placemarks and centerline LineString from KML root."""
    NAMESPACE = {'kml': 'http://www.opengis.net/kml/2.2'}

    placemarks = kml_root.xpath('.//kml:Placemark', namespaces=NAMESPACE)
    agm_list = []
    centerline_coords = None

    for pm in placemarks:
        # AGM: must contain <Point>
        point = pm.find('kml:Point', namespaces=NAMESPACE)
        name = pm.find('kml:name', namespaces=NAMESPACE)
        linestring = pm.find('kml:LineString', namespaces=NAMESPACE)

        if point is not None and name is not None:
            coords_elem = point.find('kml:coordinates', namespaces=NAMESPACE)
            if coords_elem is not None:
                coord = coords_elem.text.strip()
                vals = coord.split(',')
                # ALTITUDE OPTIONAL
                if len(vals) >= 2:
                    agm_list.append({
                        'name': name.text.strip(),
                        'lon': float(vals[0]),
                        'lat': float(vals[1]),
                        'alt': float(vals[2]) if len(vals) >= 3 else 0.0
                    })
        elif linestring is not None and centerline_coords is None:
            coords_elem = linestring.find('kml:coordinates', namespaces=NAMESPACE)
            if coords_elem is not None:
                raw = coords_elem.text.strip().replace('\n', ' ').replace('\r', ' ')
                coord_lines = [s for s in raw.split() if s.strip()]
                centerline_coords = []
                for c in coord_lines:
                    parts = c.strip().split(',')
                    if len(parts) >= 2:
                        centerline_coords.append((
                            float(parts[0]),
                            float(parts[1]),
                            float(parts[2]) if len(parts) >= 3 else 0.0
                        ))
    return agm_list, centerline_coords

def sort_agms(agm_list):
    # Try to sort AGM names numerically (if possible)
    def extract_number(name):
        try:
            return int(name)
        except Exception:
            # Fallback: remove non-numeric prefixes: e.g. "AGM 000" -> 000
            try:
                return int(''.join(filter(str.isdigit, name)))
            except Exception:
                return name
    return sorted(agm_list, key=lambda x: extract_number(x['name']))

def find_closest_point_on_line(agm, centerline):
    point = Point(agm['lon'], agm['lat'])
    line2d = LineString([(c[0], c[1]) for c in centerline])
    dist_along = line2d.project(point)
    nearest = line2d.interpolate(dist_along)
    # Find index of nearest point for interpolation
    min_dist = None
    nearest_idx = 0
    for i, coord in enumerate(centerline):
        test_pt = Point(coord[0], coord[1])
        d = test_pt.distance(nearest)
        if min_dist is None or d < min_dist:
            min_dist = d
            nearest_idx = i
    return nearest_idx

def get_subsegment_indices(start_idx, end_idx):
    """Return indices to slice the centerline; always in increasing order."""
    if start_idx <= end_idx:
        return start_idx, end_idx
    else:
        return end_idx, start_idx

def haversine_3d(p1, p2):
    # Inputs: (lon, lat, alt) in degrees and meters
    # Output: 3D distance in meters
    R = 6371000.0  # Earth radius in meters
    lon1, lat1, alt1 = p1
    lon2, lat2, alt2 = p2
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = phi2 - phi1
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    d_2d = R * c
    dz = alt2 - alt1
    return sqrt(d_2d**2 + dz**2)

def meters_to_feet(m):
    return m * 3.280839895

def meters_to_miles(m):
    return m * 0.000621371

# -- Main Streamlit App --

st.title("AGM Segment Distance Calculator (Terrain-Aware)")

uploaded_file = st.file_uploader("Upload a KML or KMZ file containing AGMs and a centerline:",
                                 type=['kml', 'kmz'])

if uploaded_file:
    file_name = uploaded_file.name
    if file_name.lower().endswith('.kmz'):
        # Extract doc.kml from KMZ (zipfile)
        try:
            with zipfile.ZipFile(uploaded_file, 'r') as kmz:
                kml_path = None
                # find the first .kml in the zip if not doc.kml
                for candidate in kmz.namelist():
                    if candidate.lower().endswith('.kml'):
                        kml_path = candidate
                        break
                if kml_path is None:
                    st.error("No KML file found inside KMZ archive.")
                    st.stop()
                kml_bytes = kmz.read(kml_path)
        except Exception as e:
            st.error(f"Failed to extract KML from KMZ: {e}")
            st.stop()
    else:
        kml_bytes = uploaded_file.read()

    # Parse KML as bytes
    kml_root = kml_to_root(kml_bytes)
    if kml_root is None:
        st.stop()

    agms, centerline_coords = extract_agms_and_centerline(kml_root)
    if not agms:
        st.error("No AGM points (<Placemark> with <Point>) found in KML/KMZ.")
        st.stop()
    if centerline_coords is None:
        st.error("No centerline (<Placemark> with <LineString>) found in KML/KMZ.")
        st.stop()
    if len(centerline_coords) < 2:
        st.error("Centerline must contain at least two coordinates.")
        st.stop()

    # Sort AGMs by name
    agms_sorted = sort_agms(agms)

    # Get start/end indices along centerline for each AGM
    centerline2d = LineString([(c[0], c[1]) for c in centerline_coords])
    agm_point_indices = []
    for agm in agms_sorted:
        idx = find_closest_point_on_line(agm, centerline_coords)
        agm_point_indices.append(idx)

    # Build output table
    segment_names = []
    seg_distance_feet = []
    seg_distance_miles = []
    cum_distance_feet = []

    total_distance_m = 0.0
    for i in range(len(agms_sorted) - 1):
        agm_a = agms_sorted[i]
        agm_b = agms_sorted[i + 1]
        idx_a, idx_b = agm_point_indices[i], agm_point_indices[i + 1]
        start_idx, end_idx = get_subsegment_indices(idx_a, idx_b)

        # Slice centerline; include both endpoints
        path_coords = centerline_coords[start_idx:end_idx+1]
        if start_idx > end_idx:
            # If reversed, reverse the list so distance is positive
            path_coords = path_coords[::-1]

        # If the number of points is < 2 (degenerate), use direct AGM-AGM endpoint
        if len(path_coords) < 2:
            # Fallback: use AGM points only
            path_coords = [ (agm_a['lon'], agm_a['lat'], agm_a['alt']),
                            (agm_b['lon'], agm_b['lat'], agm_b['alt']) ]

        # 3D distance along the centerline between AGMs
        seg_dist_m = sum(haversine_3d(path_coords[j], path_coords[j+1])
                         for j in range(len(path_coords)-1))

        total_distance_m += seg_dist_m

        seg_dist_ft = meters_to_feet(seg_dist_m)
        cum_dist_ft = meters_to_feet(total_distance_m)
        seg_dist_mi = meters_to_miles(seg_dist_m)
        seg_name = f"{agm_a['name']} to {agm_b['name']}"

        segment_names.append(seg_name)
        seg_distance_feet.append(round(seg_dist_ft,2))
        seg_distance_miles.append(round(seg_dist_mi, 5))
        cum_distance_feet.append(round(cum_dist_ft,2))

    # "Rebase" chainage: Set the cumulative distance for the first segment to 0 at AGM 000
    initial_agm_name = agms_sorted[0]['name']
    initial_cum_ft = 0.0  # by definition

    output_df = pd.DataFrame({
        'Segment': segment_names,
        'Distance (ft)': seg_distance_feet,
        'Distance (mi)': seg_distance_miles,
        'Total Distance So Far (ft)': cum_distance_feet
    })
    # Zero chainage for initial AGM, increments thereafter.
    output_df.loc[0, 'Total Distance So Far (ft)'] = 0.0

    # Display in Streamlit
    st.header("AGM Segment Distance Table")
    st.dataframe(output_df, use_container_width=True)

    # CSV Download
    csv_bytes = output_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download table as CSV",
        data=csv_bytes,
        file_name='agm_distances.csv',
        mime='text/csv'
    )
