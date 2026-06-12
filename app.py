
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from PIL import Image
from datetime import datetime
import sqlite3

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Spray Inspection Dashboard",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: white;
}

[data-testid="metric-container"] {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("Spray Inspection Dashboard")

st.markdown(
    "Cloud-Based Multi Image Spray Analysis System"
)

# =========================================================
# DATABASE
# =========================================================

conn = sqlite3.connect(
    "spray_inspection.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name TEXT,
    spray_angle REAL,
    peak_count INTEGER,
    timestamp TEXT
)
""")

conn.commit()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Dashboard Controls")

show_database = st.sidebar.checkbox(
    "Show Database",
    value=True
)

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_files = st.file_uploader(
    "Upload Spray Images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# =========================================================
# PROCESSING FUNCTION
# =========================================================

def process_image(image):

    image_np = np.array(image)

    gray = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5,5),
        0
    )

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    edges = cv2.Canny(
        thresh,
        50,
        150
    )

    h, w = edges.shape

    crop_start = int(h * 0.20)

    cropped_edges = edges[crop_start:, :]

    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=100,
        maxLineGap=20
    )

    line_image = np.zeros_like(cropped_edges)

    overlay = image_np.copy()

    left_angles = []
    right_angles = []

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]

            angle = np.degrees(
                np.arctan2(
                    (y2-y1),
                    (x2-x1)
                )
            )

            if -80 < angle < -10:

                left_angles.append(angle)

                cv2.line(
                    line_image,
                    (x1,y1),
                    (x2,y2),
                    255,
                    2
                )

                cv2.line(
                    overlay,
                    (x1, y1 + crop_start),
                    (x2, y2 + crop_start),
                    (0,255,0),
                    3
                )

            elif 10 < angle < 80:

                right_angles.append(angle)

                cv2.line(
                    line_image,
                    (x1,y1),
                    (x2,y2),
                    255,
                    2
                )

                cv2.line(
                    overlay,
                    (x1, y1 + crop_start),
                    (x2, y2 + crop_start),
                    (0,255,0),
                    3
                )

    spray_angle = None

    if left_angles and right_angles:

        left_mean = np.mean(left_angles)

        right_mean = np.mean(right_angles)

        spray_angle = 180 - abs(
            right_mean - left_mean
        )

    # =====================================================
    # CENTERLINE
    # =====================================================

    center_x = w // 2

    cv2.line(
        overlay,
        (center_x,0),
        (center_x,h),
        (255,0,0),
        2
    )

    # =====================================================
    # SPRAY ANGLE TEXT
    # =====================================================

    if spray_angle is not None:

        cv2.putText(
            overlay,
            f"Angle: {spray_angle:.2f} deg",
            (40,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    # =====================================================
    # SPRAY PROFILE
    # =====================================================

    sample_y = int(h * 0.55)

    profile = gray[sample_y, :]

    peaks, _ = find_peaks(
        profile,
        prominence=15,
        distance=20
    )

    # =====================================================
    # PEAK MARKERS
    # =====================================================

    for peak in peaks:

        cv2.circle(
            overlay,
            (peak, sample_y),
            8,
            (255,0,255),
            -1
        )

    # =====================================================
    # HEATMAP
    # =====================================================

    heatmap = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_INFERNO
    )

    return {
        "image": image_np,
        "gray": gray,
        "binary": thresh,
        "edges": edges,
        "boundary": line_image,
        "overlay": overlay,
        "heatmap": heatmap,
        "angle": spray_angle,
        "profile": profile,
        "peaks": peaks,
        "peak_count": len(peaks)
    }

# =========================================================
# MAIN PROCESSING
# =========================================================

results = []

if uploaded_files:

    for uploaded_file in uploaded_files:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        result = process_image(image)

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        results.append({
            "Image Name": uploaded_file.name,
            "Spray Angle": round(
                result["angle"], 2
            ) if result["angle"] else None,
            "Peak Count": result["peak_count"],
            "Timestamp": timestamp
        })

        cursor.execute("""
        INSERT INTO inspections (
            image_name,
            spray_angle,
            peak_count,
            timestamp
        )
        VALUES (?, ?, ?, ?)
        """, (
            uploaded_file.name,
            float(result["angle"])
            if result["angle"] else None,
            result["peak_count"],
            timestamp
        ))

        conn.commit()

    df = pd.DataFrame(results)

    # =====================================================
    # KPI CARDS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Images",
        len(df)
    )

    col2.metric(
        "Average Angle",
        f"{df['Spray Angle'].mean():.2f}°"
    )

    col3.metric(
        "Maximum Angle",
        f"{df['Spray Angle'].max():.2f}°"
    )

    col4.metric(
        "Minimum Angle",
        f"{df['Spray Angle'].min():.2f}°"
    )

    # =====================================================
    # RESULTS TABLE
    # =====================================================

    st.subheader("Inspection Results")

    st.data_editor(
        df,
        use_container_width=True
    )

    # =====================================================
    # ANGLE DISTRIBUTION
    # =====================================================

    st.subheader("Angle Distribution")

    fig, ax = plt.subplots(
        figsize=(10,4)
    )

    ax.hist(
        df["Spray Angle"].dropna(),
        bins=10
    )

    ax.set_title(
        "Spray Angle Histogram"
    )

    ax.set_xlabel(
        "Spray Angle"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.grid(True)

    st.pyplot(fig)

    # =====================================================
    # DETAILED VISUALIZATION
    # =====================================================

    st.subheader("Detailed Spray Visualization")

    for uploaded_file in uploaded_files:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        result = process_image(image)

        with st.expander(uploaded_file.name):

            col1, col2 = st.columns(2)

            # =============================================
            # LEFT COLUMN
            # =============================================

            with col1:

                st.image(
                    result["image"],
                    caption="Original Spray Image",
                    use_container_width=True
                )

                st.image(
                    result["overlay"],
                    caption="Spray Boundary Overlay",
                    use_container_width=True
                )

                st.image(
                    result["heatmap"],
                    caption="Spray Intensity Heatmap",
                    use_container_width=True
                )

            # =============================================
            # RIGHT COLUMN
            # =============================================

            with col2:

                st.image(
                    result["binary"],
                    caption="Binary Spray Mask",
                    use_container_width=True
                )

                st.image(
                    result["edges"],
                    caption="Canny Edge Detection",
                    use_container_width=True
                )

                st.image(
                    result["boundary"],
                    caption="Detected Spray Boundary",
                    use_container_width=True
                )

                st.metric(
                    "Spray Angle",
                    f"{result['angle']:.2f}°"
                    if result["angle"]
                    else "N/A"
                )

                st.metric(
                    "Peak Count",
                    result["peak_count"]
                )

                # =========================================
                # PROFILE GRAPH
                # =========================================

                fig2, ax2 = plt.subplots(
                    figsize=(8,3)
                )

                ax2.plot(
                    result["profile"]
                )

                ax2.plot(
                    result["peaks"],
                    result["profile"][
                        result["peaks"]
                    ],
                    "rx"
                )

                ax2.set_title(
                    "Spray Distribution Profile"
                )

                ax2.grid(True)

                st.pyplot(fig2)

# =========================================================
# DATABASE VIEW
# =========================================================

if show_database:

    st.subheader("Inspection Database")

    db_df = pd.read_sql_query(
        "SELECT * FROM inspections",
        conn
    )

    st.dataframe(
        db_df,
        use_container_width=True
    )

# =========================================================
# CSV EXPORT
# =========================================================

if uploaded_files:

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Inspection CSV",
        data=csv,
        file_name="spray_inspection_data.csv",
        mime="text/csv"
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    "Dipendu Mondal | Open Source Spray Inspection Dashboard | Streamlit + OpenCV + SQLite"
)

