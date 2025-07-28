#!/usr/bin/env python3
import sys
import xml.etree.ElementTree as ET
import requests
import json

def parse_kml(kml_file):
    """
    Parse a KML file and extract all (lat, lon) tuples from the first <coordinates> element.
    """
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    tree = ET.parse(kml_file)
    root = tree.getroot()
    coord_text = root.find('.//kml:coordinates', ns).text.strip()
    points = []
    for token in coord_text.split():
        lon, lat, *_ = token.split(',')
        points.append((float(lat), float(lon)))
    return points

def fetch_opentopo(coords):
    """
    Query OpenTopography for a list of (lat, lon) points in one bulk request.
    Raises on HTTP/errors or missing data.
    """
    url = 'https://portal.opentopography.org/API/point'
    params = {
        'demtype': 'SRTMGL1',
        'outputFormat': 'JSON',
    }
    # build repeated latitude/longitude params
    params['latitude']  = [lat for lat, lon in coords]
    params['longitude'] = [lon for lat, lon in coords]
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if 'data' not in payload:
        raise ValueError("OpenTopography returned no 'data' field")
    return [pt['elevation'] for pt in payload['data']]

def fetch_open_elevation(coords):
    """
    Fallback to Open-Elevation bulk lookup via POST JSON.
    Raises on HTTP/errors or missing data.
    """
    url = 'https://api.open-elevation.com/api/v1/lookup'
    body = {'locations': [{'latitude': lat, 'longitude': lon} for lat, lon in coords]}
    resp = requests.post(url, json=body, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if 'results' not in payload:
        raise ValueError("Open-Elevation returned no 'results' field")
    return [pt['elevation'] for pt in payload['results']]

def main():
    if len(sys.argv) != 2:
        print("Usage: diagnose_opentopo_api.py path/to/your.kml", file=sys.stderr)
        sys.exit(1)

    kml_path = sys.argv[1]
    coords   = parse_kml(kml_path)

    try:
        elevations = fetch_opentopo(coords)
        source     = "OpenTopography"
    except Exception as e:
        print(f"⚠ OpenTopography failed ({e}); falling back to Open-Elevation", file=sys.stderr)
        elevations = fetch_open_elevation(coords)
        source     = "Open-Elevation"

    # output CSV to stdout
    print("index,latitude,longitude,elevation_m,source")
    for idx, ((lat, lon), elev) in enumerate(zip(coords, elevations)):
        print(f"{idx},{lat},{lon},{elev},{source}")

if __name__ == '__main__':
    main()
