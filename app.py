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
import re
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="KML/KMZ Linestring Distance Calculator",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- KML Namespace ---
KML_NAMESPACE = "{http://www.opengis.net/kml/2.2}"

# --- Conversion Factors ---
KM_TO_FEET = 3280.84
KM_TO_MILES = 0.621371

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

def parse_coordinates(coords_text):
    """Parse coordinate string into list of (lon, lat, alt) tuples."""
    coords = []
    if not coords_text:
        return coords
    
    # Clean up the coordinate text
    coords_text = coords_text.strip()
    
    # Split by whitespace or newlines
    coordinate_pairs = coords_text.replace('\n', ' ').replace('\t', ' ').split()
    
    for pair in coordinate_pairs:
        if pair.strip():
            try:
                parts = pair.strip().split(',')
                if len(parts) >= 2:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    alt = float(parts[2]) if len(parts) > 2 else 0.0
                    coords.append((lon, lat, alt))
            except (ValueError, IndexError) as e:
                st.warning(f"Skipping invalid coordinate: {pair} - {e}")
                continue
    
    return coords

def parse_kml_content(kml_data):
    """Parse KML content to extract linestring coordinates and AGM points."""
    centerline_linestrings = []
    agm_points = []
    
    try:
        root = ET.fromstring(kml_data)
        st.write("✅ KML parsed successfully")

        # Find the main folder
        main_folder = find_element_by_name(root, "1GOOGLE EARTH SEED FILE V2.0")
        target_scope = main_folder if main_folder is not None else root
        
        if main_folder is not None:
            st.write("✅ Found main folder: '1GOOGLE EARTH SEED FILE V2.0'")
        else:
            st.write("⚠️ Main folder not found, searching entire document")

        # Find CENTERLINE
        centerline_container = find_element_by_name(target_scope, "CENTERLINE")
        if centerline_container:
            st.write("✅ Found CENTERLINE container")
            
            if centerline_container.tag == f"{KML_NAMESPACE}Placemark":
                linestring_geom = centerline_container.find(f"{KML_NAMESPACE}LineString")
                if linestring_geom is not None:
                    coords_element = linestring_geom.find(f"{KML_NAMESPACE}coordinates")
                    if coords_element is not None:
                        coords_text = coords_element.text
                        coords = parse_coordinates(coords_text)
                        if coords:
                            centerline_linestrings.append({
                                "name": centerline_container.find(f"{KML_NAMESPACE}name").text,
                                "coordinates": coords
                            })
                            st.write(f"✅ Parsed {len(coords)} centerline coordinates")
                        else:
                            st.error("No valid coordinates found in CENTERLINE")
            
            elif centerline_container.tag == f"{KML_NAMESPACE}Folder":
                st.write("CENTERLINE is a folder, looking for linestrings inside...")
                for placemark in centerline_container.iter(f"{KML_NAMESPACE}Placemark"):
                    linestring_geom = placemark.find(f"{KML_NAMESPACE}LineString")
                    if linestring_geom is not None:
                        coords_element = linestring_geom.find(f"{KML_NAMESPACE}coordinates")
                        if coords_element is not None:
                            coords_text = coords_element.text
                            coords = parse_coordinates(coords_text)
                            if coords:
                                centerline_linestrings.append({
                                    "name": placemark.find(f"{KML_NAMESPACE}name").text,
                                    "coordinates": coords
                                })
        else:
            st.error("❌ Could not find CENTERLINE")

        # Find AGMs
        agms_container = find_element_by_name(target_scope, "AGMs")
        if agms_container and agms_container.tag == f"{KML_NAMESPACE}Folder":
            st.write("✅ Found AGMs folder")
            
            for placemark in agms_container.iter(f"{KML_NAMESPACE}Placemark"):
                point_geom = placemark.find(f"{KML_NAMESPACE}Point")
                if point_geom is not None:
                    coords_element = point_geom.find(f"{KML_NAMESPACE}coordinates")
                    if coords_element is not None:
                        coords_text = coords_element.text
                        coords = parse_coordinates(coords_text)
                        if coords:
                            name_element = placemark.find(f"{KML_NAMESPACE}name")
                            name = name_element.text if name_element is not None else "Unknown"
                            agm_points.append({
                                "name": name,
                                "coordinates": coords[0]  # Take first coordinate
                            })
            
            st.write(f"✅ Found {len(agm_points)} AGM points")
        else:
            st.error("❌ Could not find AGMs folder")

    except ET.ParseError as e:
        st.error(f"❌ XML parsing error: {e}")
        return [], []
    except Exception as e:
        st.error(f"❌ Error parsing KML: {e}")
        return [], []
    
    return centerline_linestrings, agm_points

@st.cache_data(ttl=3600)
def get_elevations(coordinates, api_key):
    """Fetch elevation data using Google Maps Elevation API."""
    if not api_key:
        st.error("Google Maps API Key is required")
        return None
    
    # Split into chunks if too many coordinates
    chunk_size = 100  # API limit
    all_elevations = []
    
    for i in range(0, len(coordinates), chunk_size):
        chunk = coordinates[i:i + chunk_size]
        locations_str = "|".join([f"{coord[1]},{coord[0]}" for coord in chunk])
        
        api_url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={locations_str}&key={api_key}"
        
        try:
            time.sleep(0.1)  # Rate limiting
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] == "OK":
                elevations = [result["elevation"] for result in data["results"]]
                all_elevations.extend(elevations)
                st.write(f"✅ Fetched elevations for {len(elevations)} points")
            else:
                st.error(f"API Error: {data['status']} - {data.get('error_message', '')}")
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {e}")
            return None
        except Exception as e:
            st.error(f"Error processing elevation data: {e}")
            return None
    
    return all_elevations

def natural_sort_key(s):
    """Key for natural sorting."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def calculate_terrain_aware_distances(path_coords, agm_points, api_key):
    """Calculate terrain-aware distances between AGMs along the path."""
    if not path_coords or len(path_coords) < 2:
        st.error("CENTERLINE must have at least 2 points")
        return []
    
    if len(agm_points) < 2:
        st.error("Need at least 2 AGM points")
        return []
    
    st.write(f"🔄 Processing {len(path_coords)} centerline points and {len(agm_points)} AGM points")
    
    # Get elevations for centerline
    path_coords_2d = [(c[0], c[1]) for c in path_coords]
    path_elevations = get_elevations(path_coords_2d, api_key)
    
    if not path_elevations:
        st.error("Could not fetch centerline elevations")
        return []
    
    # Create 3D path points
    path_3d = []
    for i, coord in enumerate(path_coords):
        path_3d.append((coord[1], coord[0], path_elevations[i]))  # (lat, lon, alt)
    
    # Calculate cumulative 3D distances along centerline
    cumulative_distances = [0.0]
    for i in range(1, len(path_3d)):
        p1 = path_3d[i-1]
        p2 = path_3d[i]
        
        # 2D distance
        dist_2d_m = geodesic((p1[0], p1[1]), (p2[0], p2[1])).meters
        
        # Altitude difference
        alt_diff_m = p2[2] - p1[2]
        
        # 3D distance
        dist_3d_m = np.sqrt(dist_2d_m**2 + alt_diff_m**2)
        
        cumulative_distances.append(cumulative_distances[-1] + dist_3d_m / 1000.0)  # Convert to km
    
    st.write(f"✅ Calculated centerline cumulative distances (total: {cumulative_distances[-1]:.2f} km)")
    
    # Get elevations for AGM points
    agm_coords_2d = [(agm['coordinates'][0], agm['coordinates'][1]) for agm in agm_points]
    agm_elevations = get_elevations(agm_coords_2d, api_key)
    
    if not agm_elevations:
        st.error("Could not fetch AGM elevations")
        return []
    
    # Update AGM points with elevations
    for i, agm in enumerate(agm_points):
        lon, lat = agm['coordinates'][0], agm['coordinates'][1]
        agm['coordinates'] = (lon, lat, agm_elevations[i])
    
    # Create 2D centerline for projection
    centerline_2d = LineString([(c[0], c[1]) for c in path_coords])
    
    # Project each AGM onto centerline and calculate distances
    agm_distances = []
    for agm in agm_points:
        agm_lon, agm_lat, agm_alt = agm['coordinates']
        agm_point = Point(agm_lon, agm_lat)
        
        # Project AGM onto centerline
        projected_point = nearest_points(centerline_2d, agm_point)[0]
        proj_lon, proj_lat = projected_point.x, projected_point.y
        
        # Get distance along centerline to projected point
        distance_along_line = centerline_2d.project(projected_point)
        
        # Convert to cumulative 3D distance using interpolation
        total_2d_length = centerline_2d.length
        if total_2d_length > 0:
            fraction = distance_along_line / total_2d_length
            distance_along_path_km = fraction * cumulative_distances[-1]
        else:
            distance_along_path_km = 0
        
        agm_distances.append({
            'name': agm['name'],
            'coordinates': agm['coordinates'],
            'distance_along_path_km': distance_along_path_km
        })
        
        st.write(f"AGM {agm['name']}: {distance_along_path_km:.3f} km along path")
    
    #
