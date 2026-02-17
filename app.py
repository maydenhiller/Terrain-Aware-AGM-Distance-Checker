def distance_between(p1, p2, line, elevations):

    idx1, t1 = p1
    idx2, t2 = p2

    # Ensure p1 comes first along line
    if (idx1, t1) > (idx2, t2):
        idx1, t1, idx2, t2 = idx2, t2, idx1, t1

    # ---------- SAME SEGMENT CASE ----------
    if idx1 == idx2:
        A = line[idx1]
        B = line[idx1 + 1]

        seg_len = haversine_ft(*A, *B)

        horiz = seg_len * abs(t2 - t1)

        # approximate vertical difference
        v1 = elevations[idx1] + (elevations[idx1+1] - elevations[idx1]) * t1
        v2 = elevations[idx1] + (elevations[idx1+1] - elevations[idx1]) * t2

        vert = v2 - v1

        return math.sqrt(horiz*horiz + vert*vert)

    # ---------- DIFFERENT SEGMENTS ----------

    total = 0.0

    # partial first segment
    A = line[idx1]
    B = line[idx1 + 1]
    seg_len = haversine_ft(*A, *B)
    total += seg_len * (1 - t1)

    # full segments between
    for i in range(idx1 + 1, idx2):
        h = haversine_ft(*line[i], *line[i + 1])
        v = elevations[i + 1] - elevations[i]
        total += math.sqrt(h*h + v*v)

    # partial last segment
    A = line[idx2]
    B = line[idx2 + 1]
    seg_len = haversine_ft(*A, *B)
    total += seg_len * t2

    return total
