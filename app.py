import streamlit as st
import pandas as pd

# --- CONFIG ---
MIN_VALID_DISTANCE_FT = 500  # threshold for short segment flag

# --- CORE LOGIC ---
def compute_terrain_aware_distance(path_rows):
    """
    Placeholder distance calculation.
    Replace with your actual terrain-aware logic.
    For now, if a 'Distance (ft)' column exists, sum it.
    """
    if 'Distance (ft)' in path_rows.columns:
        return path_rows['Distance (ft)'].sum()
    return 0.0

def calculate_agm_distances(df):
    results = []

    for i in range(len(df) - 1):
        start_seg = df.iloc[i]['Segment']
        end_seg = df.iloc[i + 1]['Segment']

        # Simulate "snap index" progression
        # (In real GIS logic, you'd be using indices, not these text labels)
        segment_path = df.iloc[i:i+2]  # placeholder for actual path slicing
        distance_ft = compute_terrain_aware_distance(segment_path)
        distance_mi = distance_ft / 5280

        warning = "⚠️ Too short" if distance_ft < MIN_VALID_DISTANCE_FT else ""

        results.append({
            "Segment": f"{start_seg} → {end_seg}",
            "Distance (ft)": round(distance_ft, 2),
            "Distance (mi)": round(distance_mi, 4),
            "Warning": warning
        })

    return pd.DataFrame(results)

# --- STREAMLIT UI ---
st.set_page_config(page_title="AGM Distance Checker", layout="wide")
st.title("📏 AGM Distance Calculator with Short-Segment Flag")

uploaded_file = st.file_uploader("Upload AGM distances CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Ensure numeric distances
        if 'Distance (ft)' in df.columns:
            df['Distance (ft)'] = pd.to_numeric(df['Distance (ft)'], errors='coerce').fillna(0)

        results_df = calculate_agm_distances(df)

        st.subheader("Processed AGM Segments")
        st.dataframe(results_df, use_container_width=True)

        # Optional: download button
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download results as CSV",
            data=csv,
            file_name="agm_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("👆 Upload your AGM distances CSV to begin.")
