import streamlit as st
import joblib
import time

# --- ULTIMATE STYLING & ANIMATIONS ---
st.set_page_config(page_title="CardioCare AI", page_icon="🧪", layout="wide")

st.markdown("""
    <style>
    /* Gradient Background for Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    
    /* Glassmorphism Card Effect */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border: 1px solid #00d2ff;
    }

    /* Professional Title Animation */
    @keyframes slideDown {
        0% { transform: translateY(-50px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    .main-header {
        animation: slideDown 1.5s ease-out;
        text-align: center;
        color: #00d2ff;
        font-family: 'Trebuchet MS', sans-serif;
    }

    /* Pulse for Run Button */
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(231, 76, 60, 0); }
        100% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0); }
    }
    .stButton>button {
        background: linear-gradient(45deg, #e74c3c, #c0392b);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 50px;
        font-weight: bold;
        width: 100%;
        animation: pulse-red 2s infinite;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 👨‍🔬 Student")
    st.write("**Limbaime Axel Jonas**")
    st.markdown("---")
    st.markdown("### ⚙️ Engine")
    st.code("SVM Kernel: RBF\nOptimization: GridSearch\nDuplicates: Removed")

# --- MAIN UI ---
st.markdown("<h1 class='main-header'>🏥 CARDIOCARE DIAGNOSTICS</h1>", unsafe_allow_html=True)
st.write("---")

# Organized Grid Layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧬 Vital Signs")
    age = st.number_input("Patient Age", 40, 100, 60)
    ejection = st.slider("Ejection Fraction (%)", 10, 80, 38)
    creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.5, 9.5, 1.1)
    time_follow = st.number_input("Follow-up period (Days)", 4, 285, 100)

with col2:
    st.markdown("### 📋 Medical History")
    hbp = st.selectbox("Hypertension", ["No", "Yes"])
    anaemia = st.selectbox("Anaemia", ["No", "Yes"])
    sodium = st.number_input("Serum Sodium (mEq/L)", 110, 150, 137)
    smoke = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker"])

st.markdown("<br>", unsafe_allow_html=True)

# --- PREDICTION LOGIC ---
if st.button("RUN CLINICAL ANALYSIS"):
    with st.spinner('🧬 Analyzing biomarkers and patterns...'):
        time.sleep(2) # Visual effect
        
        # Load assets
        model = joblib.load('model_svm.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # [age, anaemia, cpk, diabetes, ejection, hbp, platelets, creatinine, sodium, sex, smoke, time]
        features = [age, 1 if anaemia=="Yes" else 0, 582, 0, ejection, 
                    1 if hbp=="Yes" else 0, 263358, creatinine, sodium, 1, 
                    1 if smoke=="Smoker" else 0, time_follow]
        
        scaled_data = scaler.transform([features])
        prob = model.predict_proba(scaled_data)[0][1]
        
        st.markdown("---")
        if prob > 0.5:
            st.markdown(f"""
                <div style="background-color:rgba(231, 76, 60, 0.2); padding:20px; border-radius:10px; border-left: 8px solid #e74c3c;">
                    <h2 style="color:#e74c3c; margin:0;">⚠️ HIGH RISK DETECTED</h2>
                    <p style="font-size:24px;">Mortality Probability: <b>{prob:.1%}</b></p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.balloons()
            st.markdown(f"""
                <div style="background-color:rgba(46, 204, 113, 0.2); padding:20px; border-radius:10px; border-left: 8px solid #2ecc71;">
                    <h2 style="color:#2ecc71; margin:0;">✅ LOW RISK STABLE</h2>
                    <p style="font-size:24px;">Survival Probability: <b>{1-prob:.1%}</b></p>
                </div>
            """, unsafe_allow_html=True)