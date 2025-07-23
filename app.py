import streamlit as st
import simplekml
from geopy.distance import geodesic
import zipfile
import io
import pandas as pd
import requests
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="KML/KMZ Linestring Distance Calculator",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Helper Functions ---

def parse_kml_content(kml_data):
    """
    Parses KML content to extract linestring coordinates.
    Supports both KML and KMZ (by extracting KML from KMZ).
    """
    linestrings = []
    try:
        # Corrected: Use simplekml.parse() to parse KML content from a string
        kml = simplekml.parse(kml_data)

        for feature in kml.features():
            if isinstance(feature, simplekml.features.Folder):
                for sub_feature in feature.features():
                    if isinstance(sub_feature, simplekml.features.Linestring):
                        linestrings.append({
                            "name": sub_feature.name,
                            "coordinates": sub_feature.coords
                        })
            elif isinstance(feature, simplekml.features.Linestring):
                linestrings.append({
                    "name": feature.name,
                    "coordinates": feature.coords
                })
    except Exception as e:
        st.error(f"Error parsing KML content: {e}")
        return []
    return linestrings

@st.cache_data(ttl=3600) # Cache results for 1 hour to avoid repeated API calls for same points
def get_elevations(coordinates, api_key):
    """
    Fetches elevation data for a list of coordinates using Google Maps Elevation API.
    Coordinates are expected as (longitude, latitude) tuples from simplekml.
    Returns a list of elevations in meters, or None if API call fails.
    """
    if not api_key:
        st.error("Google Maps API Key is required for terrain-aware distance calculation.")
        return None

    # Google Elevation API accepts locations as "latitude,longitude|latitude,longitude|..."
    # simplekml coords are (longitude, latitude, altitude)
    locations_str = "|".join([f"{coord[1]},{coord[0]}" for coord in coordinates])

    # The API has a URL length limit, so we might need to batch requests for many points.
    # For simplicity, this example sends all at once. For very long linestrings,
    # you'd need to split `locations_str` into smaller chunks.
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
        st.error(f"Network error or API request failed: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing elevation data: {e}")
        return None

def calculate_terrain_aware_distances(linestring_coords, elevations):
    """
    Calculates terrain-aware (3D) distances between consecutive points in a linestring.
    Uses geodesic distance for 2D component and elevation difference for 3D component.
    Elevations are expected in meters.
    Returns a list of dictionaries with point details and segment distances.
    """
    results = []
    total_distance_km = 0.0

    if not linestring_coords or not elevations or len(linestring_coords) != len(elevations):
        st.warning("Cannot calculate terrain-aware distances: missing coordinates or elevations.")
        return results, total_distance_km

    for i, coord in enumerate(linestring_coords):
        lat1, lon1 = coord[1], coord[0] # Current point (lat, lon)
        alt1 = elevations[i] # Current point elevation in meters
        segment_distance_km = 0.0
        segment_distance_3d_km = 0.0

        if i > 0:
            prev_coord = linestring_coords[i-1]
            lat0, lon0 = prev_coord[1], prev_coord[0] # Previous point (lat, lon)
            alt0 = elevations[i-1] # Previous point elevation in meters

            try:
                # Calculate 2D geodesic distance in meters
                distance_2d_m = geodesic((lat0, lon0), (lat1, lon1)).m

                # Calculate elevation difference in meters
                delta_alt_m = alt1 - alt0

                # Calculate 3D distance using Pythagorean theorem
                # D_3D = sqrt(D_2D^2 + Delta_Alt^2)
                segment_distance_3d_m = np.sqrt(distance_2d_m**2 + delta_alt_m**2)
                segment_distance_3d_km = segment_distance_3d_m / 1000.0 # Convert to km
                total_distance_km += segment_distance_3d_km
            except Exception as e:
                st.warning(f"Could not calculate 3D distance between points {i} and {i+1}: {e}")

        results.append({
            "Point Index": i + 1,
            "Latitude": lat1,
            "Longitude": lon1,
            "Elevation (m)": f"{alt1:.2f}",
            "Segment Distance (km)": f"{segment_distance_3d_km:.3f}" if i > 0 else "N/A"
        })
    return results, total_distance_km

# --- Streamlit UI ---

st.title("🗺️ KML/KMZ Linestring Distance Calculator")

st.markdown("""
Upload your `.kml` or `.kmz` file to calculate the terrain-aware (3D) distances
between points along any linestrings found within the file.
""")

# IMPORTANT: Hardcoding API keys is generally not recommended for security.
# For production apps, consider using Streamlit secrets or environment variables.
google_api_key = "AIzaSyB9HxznAvlGb02e-K1rhld_CPeAm_wvPWU"

uploaded_file = st.file_uploader(
    "Choose a .kml or .kmz file",
    type=["kml", "kmz"],
    help="Upload a KML or KMZ file containing linestring data."
)

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    kml_content = None

    if file_extension == "kml":
        kml_content = uploaded_file.read().decode("utf-8")
    elif file_extension == "kmz":
        try:
            # KMZ files are zip archives containing KML
            with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as kmz_file:
                # Look for .kml files inside the KMZ
                kml_filenames = [name for name in kmz_file.namelist() if name.lower().endswith('.kml')]
                if kml_filenames:
                    # Assuming the first KML file is the main one
                    kml_content = kmz_file.read(kml_filenames[0]).decode("utf-8")
                else:
                    st.error("No KML file found inside the KMZ archive.")
        except zipfile.BadZipFile:
            st.error("Invalid KMZ file. It might be corrupted or not a valid zip archive.")
        except Exception as e:
            st.error(f"Error processing KMZ file: {e}")

    if kml_content:
        st.subheader("Processing KML/KMZ Data...")
        linestrings_data = parse_kml_content(kml_content)

        if linestrings_data:
            st.success(f"Found {len(linestrings_data)} linestring(s) in the file.")

            for ls_index, ls in enumerate(linestrings_data):
                st.markdown(f"---")
                st.markdown(f"### Linestring {ls_index + 1}: {ls['name'] if ls['name'] else 'Unnamed Linestring'}")

                if ls['coordinates']:
                    st.write(f"Number of points: {len(ls['coordinates'])}")
                    st.write("Fetching elevation data...")

                    # Fetch elevations for all points in the current linestring
                    # simplekml coords are (longitude, latitude, altitude)
                    # get_elevations expects (longitude, latitude)
                    coords_for_elevation = [(c[0], c[1]) for c in ls['coordinates']]
                    elevations = get_elevations(coords_for_elevation, google_api_key)

                    if elevations:
                        st.success("Elevation data fetched successfully!")
                        st.write("Calculating terrain-aware distances...")
                        segment_results, total_dist = calculate_terrain_aware_distances(ls['coordinates'], elevations)

                        if segment_results:
                            df = pd.DataFrame(segment_results)
                            st.dataframe(df.set_index("Point Index"))
                            st.markdown(f"**Total Terrain-Aware Linestring Distance: {total_dist:.3f} km**")
                        else:
                            st.warning("No segments to calculate distances for (less than 2 points).")
                    else:
                        st.error("Could not fetch elevation data. Please check your API key and network connection.")
                else:
                    st.warning("Linestring has no coordinates.")
        else:
            st.warning("No linestrings found in the uploaded file.")
    else:
        st.info("Please upload a .kml or .kmz file to begin.")

st.markdown("---")
st.markdown("""
**How Terrain-Aware Distance is Calculated:**
This application now uses the Google Maps Elevation API to fetch the altitude for each point in your linestring.
The "terrain-aware" (3D) distance between two consecutive points is then calculated using a simplified
Pythagorean theorem: $D_{3D} = \sqrt{D_{2D}^2 + (alt_2 - alt_1)^2}$, where $D_{2D}$ is the 2D geodesic distance
between the points on the Earth's surface, and $(alt_2 - alt_1)$ is the difference in their elevations.
""")

