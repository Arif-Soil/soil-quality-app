import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import openai
import os
from dotenv import load_dotenv

# -------------------- CONFIG -------------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="SOIL QUALITY ANALYTICS PLATFORM",
    layout="centered",
    page_icon="🌱"
)

# -------------------- PROFESSIONAL STYLING --------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .header {
        border-bottom: 2px solid #BB0000;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
    }

    .sidebar .stRadio > div {
        font-size: 1.2rem !important;
        padding: 0.8rem 1rem !important;
        text-transform: uppercase !important;
        background-color: #004D40 !important;
        color: white !important;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    .sidebar .stRadio > div:hover {
        background-color: #00382E !important;
    }

    .sidebar .stRadio > div[data-baseweb="radio"]:has(input:checked) {
        background-color: #002C24 !important;
        border-left: 4px solid #BB0000;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #EAECF0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.1);
    }

    .section-title {
        color: #2C3E50;
        font-weight: 600;
        font-size: 1.4rem;
        margin: 1.5rem 0;
    }

    .footer {
        text-align: center;
        color: #667085;
        padding: 1.5rem 0;
        border-top: 1px solid #EAECF0;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<div class="header">
    <div style="text-align: center">
        <h1 style="color: #2C3E50; margin-bottom: 0.5rem; font-weight: 700">
            Soil Quality Analytics Platform
        </h1>
        <div style="color: #BB0000; font-size: 1.3rem; font-weight: 700">
            The Ohio State University • Soil, Water & Bioenergy Research Group
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------- DATA + MODEL --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Unified_Soil_Quality_App_Input.csv")
    return df.dropna(subset=['H', 'S', 'Manual Color', 'SQI Score'])

df = load_data()

color_to_sqi = {
    "Dark Purple": 15, "Purple": 25, "Light Purple": 40,
    "Pink": 60, "Light Pink": 75, "Colorless": 90
}

def train_color_model(data):
    X = data[['H', 'S']]
    y = LabelEncoder().fit_transform(data['Manual Color'])
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression())
    ])
    model.fit(X, y)
    return model

color_model = train_color_model(df)
label_encoder = LabelEncoder().fit(df['Manual Color'])

# -------------------- ANALYTICS FUNCTIONS --------------------
def extract_hs(image):
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return round(np.mean(hsv[:, :, 0])), round(np.mean(hsv[:, :, 1]))

def get_soil_rating(score):
    ratings = [
        (20, "Very Poor", "#DC2626"),
        (30, "Poor", "#EA580C"),
        (50, "Fair", "#F59E0B"),
        (70, "Good", "#10B981"),
        (85, "Very Good", "#059669"),
        (101, "Excellent", "#047857")
    ]
    for limit, label, color in ratings:
        if score < limit:
            return label, color
    return "Unknown", "#6B7280"

def ask_ai(question, sqi):
    prompt = f"""
    Soil Quality Index: {sqi}/100\n\n
    Question: {question}\n\n
    Provide a detailed, scientific response suitable for agricultural professionals.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# -------------------- SIDEBAR CONTROLS --------------------
with st.sidebar:
    st.markdown("""
    <div style="color: #004D40; font-size: 1.3rem; font-weight: 700; margin-bottom: 1rem; text-transform: uppercase;">
        ANALYSIS METHOD
    </div>
    """, unsafe_allow_html=True)

    input_method = st.radio(
        "",
        ["CAMERA IMAGE", "MANUAL COLOR", "ABSORBANCE", "REDOX POTENTIAL"],
        label_visibility="collapsed"
    )

# -------------------- MAIN CONTENT --------------------
if input_method == "CAMERA IMAGE":
    with st.container():
        st.markdown('<div class="section-title">Image Analysis</div>', unsafe_allow_html=True)
        image = st.camera_input("Capture Solution Sample")

        if image:
            with st.spinner("Processing image..."):
                img = Image.open(image).convert("RGB")
                h, s = extract_hs(img)
                X_test = pd.DataFrame([[h, s]], columns=['H', 'S'])
                pred_label = label_encoder.inverse_transform(color_model.predict(X_test))[0]
                sqi_score = color_to_sqi.get(pred_label, 0)

elif input_method == "MANUAL COLOR":
    with st.container():
        st.markdown('<div class="section-title">Color Analysis</div>', unsafe_allow_html=True)
        color = st.selectbox("Select Solution Color", list(color_to_sqi.keys()))
        sqi_score = color_to_sqi[color]

elif input_method == "ABSORBANCE":
    with st.container():
        st.markdown('<div class="section-title">Spectroscopic Analysis</div>', unsafe_allow_html=True)
        absorbance = st.slider("Absorbance at 550 nm", 0.0, 3.5, 1.2, 0.1)
        sqi_score = round(25 * absorbance + 5)

elif input_method == "REDOX POTENTIAL":
    with st.container():
        st.markdown('<div class="section-title">Electrochemical Analysis</div>', unsafe_allow_html=True)
        redox = st.slider("Redox Potential (mV)", 0.0, 30.0, 15.0, 0.1)
        sqi_score = round(-2.1 * redox + 91.7)

# -------------------- RESULTS PANEL --------------------
if 'sqi_score' in locals():
    with st.container():
        st.markdown('<div class="section-title">Analysis Report</div>', unsafe_allow_html=True)

        rating_text, rating_color = get_soil_rating(sqi_score)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #667085; font-size: 0.95rem">Soil Quality Index</div>
                <div style="color: {rating_color}; font-size: 2rem; font-weight: 600">{sqi_score}/100</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #667085; font-size: 0.95rem">Quality Rating</div>
                <div style="color: {rating_color}; font-size: 1.5rem; font-weight: 600">{rating_text}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            recommendation = "Requires Amendment" if sqi_score < 50 else "Optimal Condition"
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #667085; font-size: 0.95rem">Recommendation</div>
                <div style="color: #2C3E50; font-size: 1.2rem; font-weight: 500">{recommendation}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <style>
            .stProgress > div > div > div {{
                background-color: {rating_color} !important;
            }}
        </style>
        """, unsafe_allow_html=True)
        st.progress(sqi_score / 100)

        # AI Assistant
        with st.container():
            st.markdown('<div class="section-title">Expert Analysis</div>', unsafe_allow_html=True)
            query = st.text_input("Submit inquiry to agricultural AI:")
            if query:
                with st.spinner("Generating professional analysis..."):
                    response = ask_ai(query, sqi_score)
                    st.markdown(f"""
                    <div style="background: #F9FAFB; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #3B82F6">
                        {response}
                    </div>
                    """, unsafe_allow_html=True)

# -------------------- PROFESSIONAL FOOTER --------------------
st.markdown("""
<div class="footer">
    <div style="font-size: 0.9rem">
        © 2025 Dr. Arif Rahman & Khandakar Islam, OSU Southcenters<br>
    </div>
</div>
""", unsafe_allow_html=True)



