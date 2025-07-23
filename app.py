def calculate_distances(centerline, agms):
    if len(centerline) < 2 or len(agms) < 2:
        st.error("Need at least 2 centerline points and 2 AGMs.")
        return []

    # Use 2D centerline coordinates
    cl_2d = [(lon, lat) for lon, lat, _ in centerline]
    cl_elevs = get_elevations(cl_2d)
    if not cl_elevs or len(cl_elevs) != len(cl_2d):
        st.error("Elevation fetch failed for centerline.")
        return []

    # Build 3D centerline points with consistent orientation
    cl_3d = [(lon, lat, cl_elevs[i]) for i, (lon, lat) in enumerate(cl_2d)]

    # Terrain-aware cumulative distances
    cumulative = [0.0]
    for i in range(1, len(cl_3d)):
        p1, p2 = cl_3d[i-1], cl_3d[i]
        d2d = geodesic((p1[1], p1[0]), (p2[1], p2[0])).meters
        d_alt = p2[2] - p1[2]
        d3d = np.sqrt(max(0, d2d)**2 + d_alt**2)
        cumulative.append(cumulative[-1] + d3d / 1000.0)

    # AGM coordinates and elevation handling
    agm_2d = [(a["coordinates"][0], a["coordinates"][1]) for a in agms]
    agm_elevs = get_elevations(agm_2d)
    if not agm_elevs or len(agm_elevs) != len(agm_2d):
        st.error("Elevation fetch failed for AGMs.")
        return []

    for i, agm in enumerate(agms):
        agm["coordinates"] = (agm["coordinates"][0], agm["coordinates"][1], agm_elevs[i])

    # Build 2D centerline geometry for projections
    cl_geom = LineString(cl_2d)

    # Project AGMs and assign terrain-aware distances
    distances = []
    for agm in agms:
        lon, lat, alt = agm["coordinates"]
        pt = Point(lon, lat)
        proj = nearest_points(cl_geom, pt)[0]
        frac = cl_geom.project(proj) / cl_geom.length if cl_geom.length > 0 else 0
        dist_km = max(0, frac * cumulative[-1])
        distances.append({"name": agm["name"], "dist_km": dist_km})

    # Sort and calculate segment + cumulative distances
    distances.sort(key=lambda d: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', d["name"].lower())])
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
