def parse_kml(kml_bytes):
    k = kml.KML()
    k.from_string(kml_bytes)

    centerline = None
    agms = []

    def recurse_features(features):
        nonlocal centerline, agms
        for f in features:
            if hasattr(f, 'geometry') and isinstance(f.geometry, LineString):
                style = getattr(f, 'styleUrl', '')
                if 'ff0000' in style.lower() or 'red' in style.lower():
                    centerline = f.geometry
            elif hasattr(f, 'geometry') and isinstance(f.geometry, Point):
                name = f.name.strip()
                if name.isdigit() and not name.startswith("SP"):
                    agms.append((int(name), f.geometry))
            # FIXED: check if f.features is callable or list
            elif hasattr(f, 'features'):
                subfeatures = f.features if isinstance(f.features, list) else f.features()
                recurse_features(subfeatures)

    recurse_features(k.features())
    agms.sort(key=lambda x: x[0])
    agm_points = [p for _, p in agms]
    return centerline, agm_points
