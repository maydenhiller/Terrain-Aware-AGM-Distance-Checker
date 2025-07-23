# [All imports and setup remain the same]
# --snip--

# --- KML Parser with Folder Filtering ---
def parse_kml(kml_data):
    centerline, agms = [], []
    try:
        root = ET.fromstring(kml_data)

        # --- Process each Folder element
        folders = root.findall(f".//{KML_NAMESPACE}Folder")
        for folder in folders:
            name_tag = folder.find(f"{KML_NAMESPACE}name")
            folder_name = name_tag.text.strip().upper() if name_tag is not None and name_tag.text else ""

            if folder_name in ["MAP NOTES", "ACCESS"]:
                continue  # 🚫 Skip these folders

            placemarks = folder.findall(f"{KML_NAMESPACE}Placemark")
            for placemark in placemarks:
                name_tag = placemark.find(f"{KML_NAMESPACE}name")
                name = name_tag.text.strip() if name_tag is not None else "Unnamed"

                point = placemark.find(f"{KML_NAMESPACE}Point")
                line = placemark.find(f"{KML_NAMESPACE}LineString")

                if point is not None:
                    coords = parse_coordinates(point.find(f"{KML_NAMESPACE}coordinates").text)
                    if coords:
                        agms.append({"name": name, "coordinates": coords[0]})
                elif line is not None:
                    coords = parse_coordinates(line.find(f"{KML_NAMESPACE}coordinates").text)
                    centerline.extend(coords)

        # --- Also process standalone Placemarks under <Document> (not nested)
        document = root.find(f"{KML_NAMESPACE}Document")
        if document is not None:
            placemarks = document.findall(f"{KML_NAMESPACE}Placemark")
            for placemark in placemarks:
                name_tag = placemark.find(f"{KML_NAMESPACE}name")
                name = name_tag.text.strip() if name_tag is not None else "Unnamed"

                point = placemark.find(f"{KML_NAMESPACE}Point")
                line = placemark.find(f"{KML_NAMESPACE}LineString")

                if point is not None:
                    coords = parse_coordinates(point.find(f"{KML_NAMESPACE}coordinates").text)
                    if coords:
                        agms.append({"name": name, "coordinates": coords[0]})
                elif line is not None:
                    coords = parse_coordinates(line.find(f"{KML_NAMESPACE}coordinates").text)
                    centerline.extend(coords)

    except ET.ParseError as e:
        st.error(f"❌ KML Parse Error: {e}")
    return centerline, agms
