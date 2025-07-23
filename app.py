def calculate_distances(centerline, agms):
    api_key = "AIzaSyB9HxznAvlGb02e-K1rhld_CPeAm_wvPWU"

    if len(centerline) < 2 or len(agms) < 2:
        st.error("Need at least 2 centerline points and 2 AGMs.")
        return []

    # Elevation for centerline
    cl_2d = [(lon, lat) for lon, lat, _ in centerline]
    cl_elevs = get_elevations(cl_2d, api_key)
    if not cl_elevs or len(cl_elevs) != len(cl_2d):
        st.error("Failed to fetch valid elevation data for centerline.")
        return []

    centerline_3d = [(lat, lon, cl_elevs[i]) for i, (lon, lat) in enumerate(cl_2d)]

    # Cumulative distances
    cumulative = [0.0]
    for i in range(1, len(centerline_3d)):
        p1, p2 = centerline_3d[i-1], centerline_3d[i]
        try:
            d2d = geodesic((p1[0], p1[1]), (p2[0], p2[1])).meters
            d_alt = p2[2] - p1[2]
            d3d = np.sqrt(max(0, d2d)**2 + d_alt**2)
            cumulative.append(cumulative[-1] + d3d / 1000.0)
        except Exception as e:
            st.warning(f"Distance calc error at segment {i}: {e}")
            cumulative.append(cumulative[-1])

    # Elevation for AGMs
    agm_2d = [(a["coordinates"][0], a["coordinates"][1]) for a in agms]
    agm_elevs = get_elevations(agm_2d, api_key)
    if not agm_elevs or len(agm_elevs) != len(agm_2d):
        st.error("Failed to fetch valid elevation data for AGMs.")
        return []

    for i, agm in enumerate(agms):
        agm["coordinates"] = (agm["coordinates"][0], agm["coordinates"][1], agm_elevs[i])

    # Snap AGMs to centerline
    cl_geom = LineString([(lon, lat) for lon, lat, _ in centerline])
    distances = []
    for agm in agms:
        lon, lat, alt = agm["coordinates"]
        pt = Point(lon, lat)
        proj = nearest_points(cl_geom, pt)[0]
        frac = cl_geom.project(proj) / cl_geom.length if cl_geom.length > 0 else 0
        dist_km = max(0, frac * cumulative[-1])
        distances.append({"name": agm["name"], "dist_km": dist_km})

    # Natural sort
    distances.sort(key=lambda d: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', d["name"].lower())])

    # Build result table
    output = []
    for i in range(len(distances)-1):
        seg_km = max(0, distances[i+1]["dist_km"] - distances[i]["dist_km"])
        tot_km = max(0, distances[i+1]["dist_km"] - distances[0]["dist_km"])
        output.append({
            "From AGM": distances[i]["name"],
            "To AGM": distances[i+1]["name"],
            "Segment Distance (feet)": f"{seg_km * KM_TO_FEET:.2f}",
            "Segment Distance (miles)": f"{seg_km * KM_TO_MILES:.3f}",
            "Total Distance (feet)": f"{tot_km * KM_TO_FEET:.2f}",
            "Total Distance (miles)": f"{tot_km * KM_TO_MILES:.3f}"
        })
    return output
