def parse_kml(kml_data):
    agms = []
    centerline = []

    try:
        k = kml.KML()
        if isinstance(kml_data, bytes):
            k.from_string(kml_data)
        else:
            k.from_string(kml_data.encode("utf-8"))

        # Defensive navigation of KML levels
        def safe_features(obj):
            try:
                return list(obj.features())
            except TypeError:
                return list(obj.features) if hasattr(obj, 'features') else []

        for doc in safe_features(k):
            for folder in safe_features(doc):
                for feature in safe_features(folder):
                    if hasattr(feature, 'geometry'):
                        if feature.geometry.geom_type == 'LineString':
                            if feature.style_url and 'ff0000' in feature.style_url.lower():
                                centerline = list(feature.geometry.coords)
                            elif feature.description and 'ff0000' in feature.description.lower():
                                centerline = list(feature.geometry.coords)
                        elif feature.geometry.geom_type == 'Point':
                            name = feature.name.strip()
                            if name.isnumeric():
                                coord = list(feature.geometry.coords)[0]
                                agms.append((name, coord))
    except Exception as e:
        st.error(f"Failed to parse KML: {e}")
        return [], []

    agms.sort(key=lambda x: int(x[0]))
    return centerline, agms
