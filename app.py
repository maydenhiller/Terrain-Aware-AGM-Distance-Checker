import streamlit as st
import xml.etree.ElementTree as ET
from geopy.distance import geodesic
import zipfile
import io
import pandas as pd
import requests
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
import re # Import for natural sorting

# --- Page Configuration ---
st.set_page_config(
    page_title="KML/KMZ Linestring Distance Calculator",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- KML Namespace ---
KML_NAMESPACE = "{http://www.opengis.net/kml/2.2}"

# --- Conversion Factors ---
KM_TO_FEET = 3280.84 # 1 kilometer = 3280.84 feet
KM_TO_MILES = 0.621371 # 1 kilometer = 0.621371 miles

# --- Helper Functions ---

def find_element_by_name(parent_element, name_to_find):
    """Recursively searches for an XML element (Folder or Placemark) by its KML name tag."""
    for element in parent_element.iter(f"{KML_NAMESPACE}Folder"):
        name_tag = element.find(f"{KML_NAMESPACE}name")
        if name_tag is not None and name_tag.text == name_to_find:
            return element
    for element in parent_element.iter(f"{KML_NAMESPACE}Placemark"):
        name_tag = element.find(f"{KML_NAMESPACE}name")
        if name_tag is not None and name_tag.text == name_to_find:
            return element
    return None

def parse_kml_content(kml_data):
    """
    Parses KML content using ElementTree to extract linestring coordinates and AGM points.
    Supports both KML and KMZ (by extracting KML from KMZ).
    """
    centerline_linestrings = []
    agm_points = []
    try:
        root = ET.fromstring(kml_data)

        # Find the main folder "1GOOGLE EARTH SEED FILE V2.0"
        main_folder = find_element_by_name(root, "1GOOGLE EARTH SEED FILE V2.0")
        target_scope = main_folder if main_folder is not None else root

        # --- REVERTED: Find CENTERLINE linestring directly under target_scope ---
        centerline_container = find_element_by_name(target_scope, "CENTERLINE")
        if centerline_container:
            if centerline_container.tag == f"{KML_NAMESPACE}Placemark":
                linestring_geom = centerline_container.find(f"{KML_NAMESPACE}LineString")
                if linestring_geom is not None:
                    coords_text = linestring_geom.find(f"{KML_NAMESPACE}coordinates").text
                    coords = []
                    for pair in coords_text.strip().split(' '):
                        if pair:
                            parts = pair.split(',')
                            lon, lat = float(parts[0]), float(parts[1])
                            alt = float(parts[2]) if len(parts) > 2 else 0.0
                            coords.append((lon, lat, alt)) # (lon, lat, alt)
                    centerline_linestrings.append({"name": centerline_container.find(f"{KML_NAMESPACE}name").text, "coordinates": coords})
                else:
                    st.warning("Found 'CENTERLINE' Placemark but no LineString geometry within it.")
            elif centerline_container.tag == f"{KML_NAMESPACE}Folder":
                # If CENTERLINE is a folder, look for linestrings inside it
                for placemark in centerline_container.iter(f"{KML_NAMESPACE}Placemark"):
                    linestring_geom = placemark.find(f"{KML_NAMESPACE}LineString")
                    if linestring_geom is not None:
                        coords_text = linestring_geom.find(f"{KML_NAMESPACE}coordinates").text
                        coords = []
                        for pair in coords_text.strip().split(' '):
                            if pair:
                                parts = pair.split(',')
                                lon, lat = float(parts[0]), float(parts[1])
                                alt = float(parts[2]) if len(parts) > 2 else 0.0
                                coords.append((lon, lat, alt))
                        centerline_linestrings.append({"name": placemark.find(f"{KML_NAMESPACE}name").text, "coordinates": coords})
        else:
            st.warning("Could not find a 'CENTERLINE' linestring or folder containing one directly under the main document.")


        # Find AGMs points (Placemarks, usually inside an 'AGMs' folder)
        agms_container = find_element_by_name(target_scope, "AGMs")
        if agms_container and agms_container.tag == f"{KML_NAMESPACE}Folder":
            for placemark in agms_container.iter(f"{KML_NAMESPACE}Placemark"):
                point_geom = placemark.find(f"{KML_NAMESPACE}Point")
                if point_geom is not None:
                    coords_text = point_geom.find(f"{KML_NAMESPACE}coordinates").text
                    parts = coords_text.strip().split(',')
                    lon, lat = float(parts[0]), float(parts[1])
                    alt = float(parts[2]) if len(parts) > 2 else 0.0
                    agm_points.append({
                        "name": placemark.find(f"{KML_NAMESPACE}name").text,
                        "coordinates": (lon, lat, alt) # (lon, lat, alt)
                    })
        else:
            st.warning("Could not find an 'AGMs' folder containing points.")

    except ET.ParseError as e:
        st.error(f"XML parsing error in KML content: {e}. Please ensure your KML is valid XML.")
        return [], []
    except Exception as e:
        st.error(f"Error parsing KML content: {e}")
        return [], []
    return centerline_linestrings, agm_points

@st.cache_data(ttl=3600) # Cache results for 1 hour to avoid repeated API calls for same points
def get_elevations(coordinates, api_key):
    """
    Fetches elevation data for a list of coordinates using Google Maps Elevation API.
    Coordinates are expected as (longitude, latitude) tuples.
    Returns a list of elevations in meters, or None if API call fails.
    """
    if not api_key:
        st.error("Google Maps API Key is required for terrain-aware distance calculation. Please ensure it's valid and enabled for Elevation API.")
        return None

    # Google Elevation API accepts locations as "latitude,longitude|latitude,longitude|..."
    locations_str = "|".join([f"{coord[1]},{coord[0]}" for coord in coordinates]) # coord is (lon, lat)

    # The API has a URL length limit (approx 2048 characters).
    # For simplicity, this example sends all at once. For very long linestrings,
    # you'd need to split `locations_str` into smaller chunks and make multiple requests.
    api_url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={locations_str}&key={api_key}"

    try:
        response = requests.get(api_url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if data["status"] == "OK":
            elevations = [result["elevation"] for result in data["results"]]
            return elevations
        else:
            st.error(f"Google Maps Elevation API Error: {data['status']}. {data.get('error_message', '')}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error or API request failed: {e}. Check your internet connection or API key restrictions.")
        return None
    except Exception as e:
        st.error(f"Error processing elevation data: {e}")
        return None

def natural_sort_key(s):
    """
    Key for natural sorting (e.g., '240A' before '240B', '9' before '10').
    Splits the string into numeric and non-numeric parts for comparison.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def calculate_terrain_aware_distances(path_coords, agm_coords_with_elevations):
    """
    Calculates terrain-aware (3D) distances between sorted AGMs along the path,
    using projection to the centerline.
    Assumes path_coords are ordered and represent the CENTERLINE.
    agm_coords_with_elevations should be a list of {'name': ..., 'coordinates': (lon, lat, alt)}
    """
    results = []
    
    if not path_coords or len(path_coords) < 2:
        st.warning("CENTERLINE path must have at least two points to calculate distances.")
        return results

    # Get elevations for the entire CENTERLINE path
    path_coords_for_elevation = [(c[0], c[1]) for c in path_coords] # (lon, lat) for API
    path_elevations = get_elevations(path_coords_for_elevation, st.session_state.google_api_key)

    if not path_elevations:
        st.error("Could not fetch elevations for the CENTERLINE path. Cannot calculate terrain-aware distances.")
        return results

    # Create a list of (lat, lon, alt) for the path vertices
    path_3d_points = [] # (lat, lon, alt)
    for i, c in enumerate(path_coords):
        path_3d_points.append((c[1], c[0], path_elevations[i])) # (lat, lon, alt)

    # Create a 2D Shapely LineString from (lon, lat) for projection
    path_2d_coords_for_shapely = [(c[0], c[1]) for c in path_coords] # (lon, lat) for Shapely
    centerline_2d_shapely = LineString(path_2d_coords_for_shapely)

    # Calculate cumulative 3D distances along the centerline vertices
    cumulative_3d_distances_along_path_km = [0.0]
    for i in range(1, len(path_3d_points)):
        p0 = path_3d_points[i-1] # (lat, lon, alt)
        p1 = path_3d_points[i]   # (lat, lon, alt)

        dist_2d_m = geodesic((p0[0], p0[1]), (p1[0], p1[1])).m
        delta_alt_m = p1[2] - p0[2]
        segment_3d_m = np.sqrt(dist_2d_m**2 + delta_alt_m**2)
        cumulative_3d_distances_along_path_km.append(cumulative_3d_distances_along_path_km[-1] + (segment_3d_m / 1000.0))

    # Calculate distance along path for each AGM by projecting onto centerline
    agms_with_path_distances = []
    for agm in agm_coords_with_elevations: # Iterate in the original KML order
        agm_name = agm['name']
        agm_lon, agm_lat, agm_alt = agm['coordinates'] # (lon, lat, alt) from input

        agm_point_2d = Point(agm_lon, agm_lat)

        # Find the closest point on the 2D centerline to the AGM
        closest_point_on_centerline_2d = nearest_points(centerline_2d_shapely, agm_point_2d)[0]
        projected_lon, projected_lat = closest_point_on_centerline_2d.x, closest_point_on_centerline_2d.y

        # Fetch elevation for this projected point on the centerline
        projected_elevation = get_elevations([(projected_lon, projected_lat)], st.session_state.google_api_key)
        if projected_elevation:
            projected_alt = projected_elevation[0]
        else:
            # Fallback: interpolate altitude along the 3D centerline based on 2D projection
            projected_measure_2d = centerline_2d_shapely.project(closest_point_on_centerline_2d)
            
            segment_index = -1
            for idx in range(len(path_2d_coords_for_shapely) - 1):
                p0_2d_len = centerline_2d_shapely.project(Point(path_2d_coords_for_shapely[idx]))
                p1_2d_len = centerline_2d_shapely.project(Point(path_2d_coords_for_shapely[idx+1]))

                if (min(p0_2d_len, p1_2d_len) <= projected_measure_2d <= max(p0_2d_len, p1_2d_len)):
                    segment_index = idx
                    break
            
            if segment_index != -1:
                p_start_3d_interp = path_3d_points[segment_index]
                p_end_3d_interp = path_3d_points[segment_index+1]
                
                segment_length_2d_actual = LineString([(p_start_3d_interp[1], p_start_3d_interp[0]), (p_end_3d_interp[1], p_end_3d_interp[0])]).length
                
                if segment_length_2d_actual > 0:
                    fraction_along_segment = geodesic((p_start_3d_interp[0], p_start_3d_interp[1]), (projected_lat, projected_lon)).m / segment_length_2d_actual
                else:
                    fraction_along_segment = 0.0

                projected_alt = p_start_3d_interp[2] + fraction_along_segment * (p_end_3d_interp[2] - p_start_3d_interp[2])
                st.warning(f"Could not fetch elevation for AGM {agm_name}'s projected point. Interpolating altitude.")
            else:
                projected_alt = agm_alt # Last resort fallback
                st.warning(f"Projected point for AGM {agm_name} not clearly on a segment. Using AGM's original elevation.")


        # Calculate shortest 3D distance from AGM to its projected point on centerline
        dist_2d_m_to_proj = geodesic((agm_lat, agm_lon), (projected_lat, projected_lon)).m
        delta_alt_m_to_proj = agm_alt - projected_alt
        shortest_distance_to_path_km = np.sqrt(dist_2d_m_to_proj**2 + delta_alt_m_to_proj**2) / 1000.0

        # Calculate the cumulative 3D distance along the path to the projected point of the AGM
        projected_measure_2d_along_full_line = centerline_2d_shapely.project(closest_point_on_centerline_2d)

        # Map this 2D measure to the 3D cumulative distances using interpolation
        cumulative_2d_lengths_for_interp_shapely_units = [0.0]
        for i in range(1, len(path_2d_coords_for_shapely)):
            p0_2d = Point(path_2d_coords_for_shapely[i-1])
            p1_2d = Point(path_2d_coords_for_shapely[i])
            cumulative_2d_lengths_for_interp_shapely_units.append(cumulative_2d_lengths_for_interp_shapely_units[-1] + p0_2d.distance(p1_2d))

        distance_along_path_km_for_agm = np.interp(
            projected_measure_2d_along_full_line,
            cumulative_2d_lengths_for_interp_shapely_units,
            cumulative_3d_distances_along_path_km
        )
        
        agms_with_path_distances.append({
            "name": agm_name,
            "coordinates": agm['coordinates'], # Original (lon, lat, alt)
            "distance_along_path_km": distance_along_path_km_for_agm, # Cumulative 3D distance along centerline
            "shortest_distance_to_path_km": shortest_distance_to_path_km # Direct 3D distance to centerline
        })

    # MODIFIED: Sort AGMs explicitly by their natural name order
    agms_with_path_distances.sort(key=lambda x: natural_sort_key(x['name']))

    # Prepare results in the requested format
    final_results = []
    
    if len(agms_with_path_distances) < 2:
        st.warning("At least two AGMs are required to calculate segments between them.")
        return final_results

    # Calculate segments between consecutive sorted AGMs
    # The 'total distance' for the first segment (000 to 010) should be the distance
    # from the projected point of 000 to the projected point of 010.
    # And then subsequent totals are cumulative from the projected point of 000.

    # Base distance for total calculation: this is the cumulative distance of the first AGM (000)
    # from the start of the centerline.
    base_distance_km = agms_with_path_distances[0]['distance_along_path_km']

    for i in range(len(agms_with_path_distances) - 1):
        from_agm = agms_with_path_distances[i]
        to_agm = agms_with_path_distances[i+1]

        # Segment distance is the difference in their cumulative distances along the centerline
        segment_dist_km = to_agm['distance_along_path_km'] - from_agm['distance_along_path_km']
        
        # Ensure segment distance is not negative (can happen due to floating point or slight deviations)
        segment_dist_km = max(0, segment_dist_km)

        # Total distance for the current 'To AGM' is its cumulative distance from the centerline start
        # minus the baseline (cumulative distance of the first AGM).
        total_distance_km = to_agm['distance_along_path_km'] - base_distance_km
        total_distance_km = max(0, total_distance_km) # Ensure non-negative

        final_results.append({
            "From AGM": from_agm['name'],
            "To AGM": to_agm['name'],
            "Segment Distance (feet)": f"{segment_dist_km * KM_TO_FEET:.2f}",
            "Segment Distance (miles)": f"{segment_dist_km * KM_TO_MILES:.3f}",
            "Total Distance (feet)": f"{total_distance_km * KM_TO_FEET:.2f}",
            "Total Distance (miles)": f"{total_distance_km * KM_TO_MILES:.3f}"
        })
    
    return final_results


# --- Streamlit UI ---

st.title("🗺️ KML/KMZ Terrain-Aware Distance Calculator")

st.markdown("""
Upload your `.kml` or `.kmz` file to calculate the terrain-aware (3D) distances
of "AGMs" (points) along the "CENTERLINE" (linestring) found within the file.
""")

# IMPORTANT: Hardcoding API keys is generally not recommended for security.
# For production apps, consider using Streamlit secrets or environment variables.
google_api_key = "AIzaSyB9HxznAvlGb02e-K1rhld_CPeAm_wvPWU"

# Store API key in session state to avoid re-fetching on every rerun
if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = google_api_key

uploaded_file = st.file_uploader(
    "Choose a .kml or .kmz file",
    type=["kml", "kmz"],
    help="Upload a KML or KMZ file containing linestring and point data."
)

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    kml_content = None

    if file_extension == "kml":
        kml_content = uploaded_file.read().decode("utf-8")
    elif file_extension == "kmz":
        try:
            with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as kmz_file:
                kml_filenames = [name for name in kmz_file.namelist() if name.lower().endswith('.kml')]
                if kml_filenames:
                    kml_content = kmz_file.read(kml_filenames[0]).decode("utf-8")
                else:
                    st.error("No KML file found inside the KMZ archive.")
        except zipfile.BadZipFile:
            st.error("Invalid KMZ file. It might be corrupted or not a valid zip archive.")
        except Exception as e:
            st.error(f"Error processing KMZ file: {e}")

    if kml_content:
        st.subheader("Processing KML/KMZ Data...")
        centerline_linestrings, agm_points = parse_kml_content(kml_content)

        if not centerline_linestrings:
            st.warning("No 'CENTERLINE' linestring or folder containing one found directly under the main document. Please ensure it exists.")
        if not agm_points:
            st.warning("No 'AGMs' (points) found in the uploaded file. Please ensure they exist in a folder named 'AGMs'.")

        if centerline_linestrings and agm_points:
            st.success("Found CENTERLINE and AGMs. Proceeding with calculations.")

            # Assuming we take the first CENTERLINE found (which should be the correct one now)
            centerline_coords = centerline_linestrings[0]['coordinates']

            st.write(f"Number of points in CENTERLINE: {len(centerline_coords)}")
            st.write(f"Number of AGMs found: {len(agm_points)}")
            st.write("Fetching elevation data for AGMs and centerline points...")

            # Fetch elevations for AGMs
            agm_coords_for_elevation_api = [(p['coordinates'][0], p['coordinates'][1]) for p in agm_points]
            agm_elevations = get_elevations(agm_coords_for_elevation_api, st.session_state.google_api_key)

            if agm_elevations:
                # Update AGM points with fetched elevations
                for i, agm in enumerate(agm_points):
                    agm['coordinates'] = (agm['coordinates'][0], agm['coordinates'][1], agm_elevations[i])

                st.success("Elevation data for AGMs fetched successfully!")
                st.write("Calculating terrain-aware distances along CENTERLINE...")

                # Calculate distances
                distance_results = calculate_terrain_aware_distances(centerline_coords, agm_points)

                if distance_results:
                    df = pd.DataFrame(distance_results)
                    st.dataframe(df.set_index("From AGM"))

                    # Export to CSV button
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Export to CSV",
                        data=csv_data,
                        file_name="agm_distances.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No distances calculated. Check if CENTERLINE has enough points or if AGMs have valid data.")
            else:
                st.error("Could not fetch elevation data for AGMs. Please check your API key and network connection.")
        elif kml_content:
            st.info("Please upload a KML/KMZ file containing both 'CENTERLINE' and 'AGMs' data.")
    else:
        st.info("Please upload a .kml or .kmz file to begin.")

st.markdown("""
---
**How Terrain-Aware Distance is Calculated:**
This application now uses the Google Maps Elevation API to fetch the altitude for each point in your linestring and AGMs.
The "terrain-aware" (3D) distance between two points is calculated using a simplified
Pythagorean theorem: $D_{3D} = \sqrt{D_{2D}^2 + (alt_2 - alt_1)^2}$, where $D_{2D}$ is the 2D geodesic distance
between the points on the Earth's surface, and $(alt_2 - alt_1)$ is the difference in their elevations.

The "Distance from Path Start" for each AGM is the cumulative 3D distance along the CENTERLINE path
to the closest *perpendicular point* on that path to the AGM. The "Shortest Distance to Path" is the direct 3D distance
from the AGM to its closest perpendicular point on the CENTERLINE.
""")
