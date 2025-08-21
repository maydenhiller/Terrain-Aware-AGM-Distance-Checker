# ---------------------------- Main flow ----------------------------
data = read_uploaded_bytes(uploaded)
if uploaded.name.lower().endswith(".kmz"):
    kml_bytes = extract_kml_from_kmz(data)
    if not kml_bytes:
        st.error("Could not extract KML from KMZ.")
        st.stop()
else:
    kml_bytes = data

agms, centerline_ll = parse_kml_for_agms_and_centerline(kml_bytes)

if not agms:
    st.error("No AGMs found under Folder named 'AGMs'.")
    st.stop()
if not centerline_ll:
    st.error("No CENTERLINE placemarks with styleUrl '#2_0' found.")
    st.stop()

# Prepare CRS
all_lons = [lon for seg in centerline_ll for lon, _ in seg]
all_lats = [lat for seg in centerline_ll for _, lat in seg]
crs_utm = utm_crs_for(all_lats, all_lons)
to_utm = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
to_wgs84 = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)

# Build centerline
line_utm = build_centerline_utm(centerline_ll, to_utm)

# Sort AGMs and project to chainage
agms_sorted = sorted(agms, key=lambda x: parse_station(x[0]))
agm_chain = []
for label, lon, lat in agms_sorted:
    x, y = to_utm.transform(lon, lat)
    s = line_utm.project(Point(x, y))
    agm_chain.append((label, lon, lat, s))

# Rebase to AGM 000
offset = None
for lab, lo, la, s in agm_chain:
    if parse_station(lab)[0] == 0:  # AGM 000
        offset = s
        break
if offset is None:
    st.error("No AGM 000 found to use as starting point.")
    st.stop()
agm_chain = [(lab, lo, la, s - offset) for lab, lo, la, s in agm_chain]

if show_debug:
    st.write("AGM projection (chainage in feet):")
    st.dataframe([
        {"AGM": lab, "lon": lo, "lat": la, "chainage_ft": round(s * METERS_TO_FEET, 2)}
        for lab, lo, la, s in agm_chain
    ])

# Compute AGM → AGM segment distances
rows = []
total_ft = 0.0
for i in range(len(agm_chain) - 1):
    lab1, lo1, la1, s0 = agm_chain[i]
    lab2, lo2, la2, s1 = agm_chain[i + 1]
    pts_xy = densified_points(line_utm, s0 + offset, s1 + offset, step_m)
    d_m = terrain_distance_m(pts_xy, to_wgs84)
    d_ft = d_m * METERS_TO_FEET
    d_mi = d_ft / FEET_PER_MILE
    total_ft += d_ft
    rows.append({
        "Segment": f"{lab1} → {lab2}",
        "Distance (ft)": round(d_ft, 2),
        "Distance (mi)": round(d_mi, 4)
    })

df = pd.DataFrame(rows)

# Display results
st.subheader("AGM Segment Distances (Terrain-aware)")
st.dataframe(df, use_container_width=True)

tot_mi = total_ft / FEET_PER_MILE
st.markdown(f"**Total Distance:** {total_ft:,.2f} ft  |  {tot_mi:.4f} mi")

# CSV download
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="agm_segment_distances.csv",
    mime="text/csv"
)
