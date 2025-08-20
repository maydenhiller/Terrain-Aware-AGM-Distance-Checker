MIN_VALID_DISTANCE_FT = 500  # Threshold for flagging short segments

def calculate_agm_distances(agm_indices, centerline):
    results = []
    last_valid_index = None

    for i in range(len(agm_indices) - 1):
        start_idx = agm_indices[i]
        end_idx = agm_indices[i + 1]

        # Ensure minimum separation
        if end_idx <= start_idx:
            end_idx = start_idx + 1  # Force forward movement

        segment_path = centerline[start_idx:end_idx + 1]
        distance_ft = compute_terrain_aware_distance(segment_path)
        distance_mi = distance_ft / 5280

        # Flag short segments
        warning = "⚠️ Too short" if distance_ft < MIN_VALID_DISTANCE_FT else ""

        results.append({
            "Segment": f"{start_idx} to {end_idx}",
            "Distance (ft)": round(distance_ft, 2),
            "Distance (mi)": round(distance_mi, 4),
            "Warning": warning
        })

        last_valid_index = end_idx

    return results
