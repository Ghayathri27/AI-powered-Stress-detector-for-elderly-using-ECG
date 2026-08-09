import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump, load


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ECG Stress Detector",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #fff5f7, #f8f9ff);
}

/* Main title */
.main-title {
    font-size: 45px;
    font-weight: 800;
    color: #7b1e3a;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

/* Cards */
.metric-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.08);
    text-align: center;
    border: 1px solid #f0dfe4;
}

.metric-title {
    font-size: 15px;
    color: #777;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #7b1e3a;
}

/* Stress result */
.stress-card {
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    margin-top: 25px;
    background: white;
    box-shadow: 0px 5px 25px rgba(0,0,0,0.1);
}

.stress-title {
    font-size: 20px;
    color: #666;
}

.stress-value {
    font-size: 42px;
    font-weight: 800;
    color: #7b1e3a;
}

/* Upload box */
[data-testid="stFileUploader"] {
    background-color: white;
    border-radius: 18px;
    padding: 15px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 17px;
    font-weight: 600;
    background-color: #7b1e3a;
    color: white;
    border: none;
}

.stButton > button:hover {
    background-color: #a83255;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# ECG GENERATION
# ============================================================

def generate_fake_ecg(duration=10, fs=250, bpm=70, noise=0.02):

    t = np.linspace(0, duration, duration * fs)

    hr = bpm / 60

    ecg = np.sin(2 * np.pi * hr * t)

    ecg += noise * np.random.randn(len(ecg))

    return ecg


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(signal, fs=250):

    peaks, _ = find_peaks(
        signal,
        distance=fs * 0.5
    )

    if len(peaks) < 2:
        return None

    rr_intervals = np.diff(peaks) / fs * 1000

    mean_hr = 60000 / np.mean(rr_intervals)

    sdnn = np.std(rr_intervals)

    rmssd = np.sqrt(
        np.mean(
            np.square(
                np.diff(rr_intervals)
            )
        )
    )

    return mean_hr, sdnn, rmssd


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    data = []

    for _ in range(400):

        bpm = np.random.randint(60, 100)

        stress = np.random.randint(0, 100)

        ecg = generate_fake_ecg(
            bpm=bpm
        )

        features = extract_features(ecg)

        if features:

            data.append(
                [*features, stress]
            )

    df = pd.DataFrame(
        data,
        columns=[
            "mean_hr",
            "sdnn",
            "rmssd",
            "stress"
        ]
    )

    X = df[
        [
            "mean_hr",
            "sdnn",
            "rmssd"
        ]
    ]

    y = df["stress"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    dump(
        model,
        "models/stress_model.pkl"
    )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():

    model_path = "models/stress_model.pkl"

    if os.path.exists(model_path):

        return load(model_path)

    else:

        return train_model()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">❤️ ECG Stress Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered ECG analysis for stress-level assessment'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2966/2966486.png",
        width=100
    )

    st.title("ECG Analyzer")

    st.write(
        "Upload an ECG CSV dataset and analyze "
        "heart-rate variability to estimate stress level."
    )

    st.divider()

    st.subheader("How it works")

    st.write("1️⃣ Upload ECG CSV")

    st.write("2️⃣ Process ECG signal")

    st.write("3️⃣ Extract HRV features")

    st.write("4️⃣ Predict stress")

    st.write("5️⃣ View results")

    st.divider()

    st.caption(
        "⚠️ This system is intended for "
        "educational and research purposes."
    )


# ============================================================
# UPLOAD SECTION
# ============================================================

st.subheader("📁 Upload ECG Dataset")

uploaded_file = st.file_uploader(
    "Choose your ECG CSV file",
    type=["csv"],
    help="Upload a CSV containing ECG signal values."
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(
            uploaded_file
        )

        st.success(
            "ECG dataset uploaded successfully!"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.info(
                f"📊 Rows: {df.shape[0]}"
            )

        with col2:

            st.info(
                f"📋 Columns: {df.shape[1]}"
            )


        # ====================================================
        # ECG COLUMN SELECTION
        # ====================================================

        st.subheader("🔎 Select ECG Signal")

        if df.shape[1] == 1:

            selected_column = df.columns[0]

        else:

            selected_column = st.selectbox(
                "Select the ECG signal column:",
                df.columns
            )

        signal = pd.to_numeric(
            df[selected_column],
            errors="coerce"
        ).dropna().values


        if len(signal) < 10:

            st.error(
                "Not enough ECG data found."
            )

            st.stop()


        # ====================================================
        # ECG GRAPH
        # ====================================================

        st.subheader("📈 ECG Signal")

        fig, ax = plt.subplots(
            figsize=(12, 4)
        )

        display_length = min(
            len(signal),
            2500
        )

        ax.plot(
            signal[:display_length]
        )

        ax.set_xlabel(
            "Samples"
        )

        ax.set_ylabel(
            "ECG Amplitude"
        )

        ax.set_title(
            "Uploaded ECG Signal"
        )

        ax.grid(
            alpha=0.2
        )

        st.pyplot(fig)


        # ====================================================
        # FEATURE EXTRACTION
        # ====================================================

        features = extract_features(
            signal
        )

        if features is None:

            st.error(
                "Unable to detect sufficient ECG peaks. "
                "Please upload a valid ECG dataset."
            )

            st.stop()


        mean_hr, sdnn, rmssd = features


        # ====================================================
        # METRICS
        # ====================================================

        st.subheader(
            "❤️ ECG Health Metrics"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        Mean Heart Rate
                    </div>
                    <div class="metric-value">
                        {mean_hr:.2f}
                    </div>
                    <div>bpm</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        SDNN
                    </div>
                    <div class="metric-value">
                        {sdnn:.2f}
                    </div>
                    <div>ms</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        RMSSD
                    </div>
                    <div class="metric-value">
                        {rmssd:.2f}
                    </div>
                    <div>ms</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # PREDICTION
        # ====================================================

        st.subheader(
            "🧠 AI Stress Analysis"
        )

        model = get_model()

        stress_value = float(
            model.predict(
                [[
                    mean_hr,
                    sdnn,
                    rmssd
                ]]
            )[0]
        )


        # Keep value within 0-100

        stress_value = max(
            0,
            min(
                100,
                stress_value
            )
        )


        # ====================================================
        # STRESS LEVEL
        # ====================================================

        if stress_value <= 40:

            stress_level = "LOW"
            message = "Your ECG indicates a low stress level."

        elif stress_value <= 70:

            stress_level = "MEDIUM"
            message = "Your ECG indicates a moderate stress level."

        else:

            stress_level = "HIGH"
            message = "Your ECG indicates a high stress level."


        # ====================================================
        # RESULT CARD
        # ====================================================

        st.markdown(
            f"""
            <div class="stress-card">

                <div class="stress-title">
                    Predicted Stress Level
                </div>

                <div class="stress-value">
                    {stress_level}
                </div>

                <p>
                    Stress Score:
                    <b>{stress_value:.2f}%</b>
                </p>

                <p>
                    {message}
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # PROGRESS BAR
        # ====================================================

        st.write(
            "### Stress Intensity"
        )

        st.progress(
            int(stress_value)
        )


        # ====================================================
        # INTERPRETATION
        # ====================================================

        if stress_level == "LOW":

            st.success(
                "🟢 LOW STRESS\n\n"
                "The analyzed ECG features indicate "
                "a relatively low stress level."
            )

        elif stress_level == "MEDIUM":

            st.warning(
                "🟡 MEDIUM STRESS\n\n"
                "The analyzed ECG features indicate "
                "a moderate level of stress. "
                "Relaxation and rest may be beneficial."
            )

        else:

            st.error(
                "🔴 HIGH STRESS\n\n"
                "The analyzed ECG features indicate "
                "a high stress level. Consider further "
                "evaluation by a qualified healthcare professional."
            )


        # ====================================================
        # SUMMARY
        # ====================================================

        st.subheader(
            "📋 Analysis Summary"
        )

        result_df = pd.DataFrame({

            "Parameter": [
                "Mean Heart Rate",
                "SDNN",
                "RMSSD",
                "Stress Score",
                "Stress Level"
            ],

            "Result": [
                f"{mean_hr:.2f} bpm",
                f"{sdnn:.2f} ms",
                f"{rmssd:.2f} ms",
                f"{stress_value:.2f}%",
                stress_level
            ]

        })

        st.table(
            result_df
        )


    except Exception as e:

        st.error(
            f"Error processing ECG file: {e}"
        )


else:

    # ========================================================
    # WELCOME SCREEN
    # ========================================================

    st.markdown(
        """
        <div style="
            background:white;
            padding:40px;
            border-radius:25px;
            text-align:center;
            box-shadow:0px 5px 20px rgba(0,0,0,0.08);
        ">

        <h2> Welcome to ECG Stress Detector</h2>

        <p style="font-size:18px;color:#666;">
        Upload an ECG CSV dataset to begin your
        stress-level analysis.
        </p>

        <br>

        <p>
        <b>ECG CSV</b>
        →
        <b>Signal Processing</b>
        →
        <b>HRV Features</b>
        →
        <b>AI Prediction</b>
        →
        <b>Low / Medium / High</b>
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )