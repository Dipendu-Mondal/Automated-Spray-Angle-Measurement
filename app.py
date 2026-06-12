
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from PIL import Image
from datetime import datetime
import sqlite3

st.set_page_config(page_title="Spray Inspection Dashboard", layout="wide")

st.title("Spray Inspection Dashboard")

conn = sqlite3.connect("spray_inspection.db", check_same_thread=False)
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

uploaded_files = st.file_uploader(
    "Upload Spray Images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

def process_image(image):

    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    edges = cv2.Canny(thresh, 50, 150)

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

            elif 10 < angle < 80:

                right_angles.append(angle)

                cv2.line(
                    line_image,
                    (x1,y1),
                    (x2,y2),
                    255,
                    2
                )

    spray_angle = None

    if left_angles and right_angles:

        left_mean = np.mean(left_angles)
        right_mean = np.mean(right_angles)

        spray_angle = 180 - abs(
            right_mean - left_mean
        )

    sample_y = int(h * 0.55)

    profile = gray[sample_y, :]

    peaks, _ = find_peaks(
        profile,
        prominence=15,
        distance=20
    )

    return {
        "image": image_np,
        "edges": line_image,
        "angle": spray_angle,
        "profile": profile,
        "peaks": peaks,
        "peak_count": len(peaks)
    }

results = []

if uploaded_files:

    for uploaded_file in uploaded_files:

        image = Image.open(uploaded_file).convert("RGB")

        result = process_image(image)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        results.append({
            "Image Name": uploaded_file.name,
            "Spray Angle": round(result["angle"], 2) if result["angle"] else None,
            "Peak Count": result["peak_count"],
            "Timestamp": timestamp
        })

        cursor.execute(
            """
            INSERT INTO inspections (
                image_name,
                spray_angle,
                peak_count,
                timestamp
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                uploaded_file.name,
                float(result["angle"]) if result["angle"] else None,
                result["peak_count"],
                timestamp
            )
        )

        conn.commit()

    df = pd.DataFrame(results)

    st.subheader("Inspection Results")

    st.dataframe(df, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Angle",
        f"{df['Spray Angle'].mean():.2f}°"
    )

    col2.metric(
        "Maximum Angle",
        f"{df['Spray Angle'].max():.2f}°"
    )

    col3.metric(
        "Minimum Angle",
        f"{df['Spray Angle'].min():.2f}°"
    )

st.markdown("---")
st.markdown("Open Source Spray Inspection Dashboard")
