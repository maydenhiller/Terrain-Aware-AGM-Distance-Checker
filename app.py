# [All imports and definitions up top stay unchanged]

# --- Upload and Analyze ---
file = st.file_uploader("📤 Upload KMZ or KML", type=["kmz", "kml"])
if file:
    ext = file.name.split('.')[-1].lower()
    kml = None

    if ext == "kml":
        kml = file.read().decode("utf-8")

    elif ext == "kmz":
        with zipfile.ZipFile(io.BytesIO(file.read()), 'r') as zf:
            kml_files = [n for n in zf.namelist() if n.endswith(".kml")]
            st.write("📦 KMZ contents:", kml_files)

            if kml_files:
                kml = zf.read(kml_files[0]).decode("utf-8")
            else:
                st.warning("❌ No .kml file found inside KMZ archive.")

    if kml:
        centerline, agms = parse_kml(kml)
        st.write(f"✅ Parsed AGMs: {len(agms)}, CENTERLINE points: {len(centerline)}")

        if not centerline and not agms:
            st.warning("🕳️ No CENTERLINE or AGMs were found in the file. Double-check folder names or file contents.")
        elif centerline and agms:
            results = calculate_distances(centerline, agms)
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df.set_index("From AGM"))
                st.download_button("📥 Export CSV", df.to_csv(index=False), "agm_distances.csv")
            else:
                st.error("📉 No distances calculated.")
        else:
            st.error("❌ Missing required CENTERLINE or AGM data.")
else:
    st.info("👀 Upload a KMZ or KML file with valid AGMs and CENTERLINE to begin.")
