import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Injection (Theme & Styling)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CKD Diagnostic Workspace",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deep clean, flat medical UI style override
st.markdown("""
    <style>
    /* Main body background adjustments */
    .stApp {
        background-color: #fcfdfe;
    }
    /* Metric container styling */
    div[data-testid="stMetricContainer"] {
        background-color: #ffffff;
        border: 1px solid #eef2f6;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    /* Headers custom styling */
    h1, h2, h3 {
        color: #1e293b !important;
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 700;
    }
    /* Custom info boxes layout */
    .med-card {
        background-color: #f8fafc;
        border-left: 5px solid #0ea5e9;
        padding: 12px 16px;
        border-radius: 4px 8px 8px 4px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Heavy Resource Loading (Cached to prevent structural lag)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load("best_ckd_model.pkl")
        scaler = joblib.load("scaler.pkl")
        features = joblib.load("selected_features.pkl")
        return model, scaler, features
    except FileNotFoundError as e:
        st.error(f"⚠️ Core asset load failure! Verify that the serialized pipeline components match current workspace directory rules. Details: {e}")
        return None, None, None

model, scaler, selected_features = load_ml_assets()

# -----------------------------------------------------------------------------
# 3. Application Workflow & Architecture Mapping
# -----------------------------------------------------------------------------
def main():
    if model is None or scaler is None or selected_features is None:
        st.warning("Application execution halted. Please resolve dependencies specified above to activate UI pipeline.")
        return

    # Sidebar Navigation Context
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2868/2868725.png", width=90)
    st.sidebar.title("Clinical Interface")
    st.sidebar.markdown("---")
    app_mode = st.sidebar.radio("Navigation", ["Diagnostic Dashboard", "About the Research Framework"])

    if app_mode == "About the Research Framework":
        show_research_info()
        return

    # Master UI Title System
    st.title("🩺 Chronic Kidney Disease (CKD) Intelligence Workspace")
    st.markdown("An advanced ensemble architecture workflow constructed dynamically for patient stratification and screening assistance.")
    st.markdown("---")

    # Diagnostic input segmentation strategy
    st.subheader("📋 Patient Diagnostic Panel Entry")
    st.info("Input valid clinical parameters evaluated from recent lab metrics to generate instantaneous inference estimates.")

    # Initialize container to split controls logically based on features detected inside `selected_features`
    input_data = {}
    
    # We build layout dynamically according to what RFE (Recursive Feature Elimination) locked down.
    # Grouping into Columns for clean horizontal utilization
    col1, col2, col3 = st.columns(3)

    # Dictionary containing configurations for medical variables
    # (Mappings default intelligently based on standard data specifications of chronic kidney disease tracking sets)
    feature_config = {
        'age': ('Numerical', col1, 'Age (Years)', 1.0, 120.0, 50.0, 1.0, "Patient chronologic lifespan assessment."),
        'bp': ('Numerical', col2, 'Blood Pressure (mm/Hg)', 50.0, 180.0, 80.0, 10.0, "Diastolic/Systolic trend parameters."),
        'sg': ('Numerical', col3, 'Specific Gravity', 1.005, 1.025, 1.015, 0.005, "Urine density compared to distilled water metrics."),
        'al': ('Numerical', col1, 'Albumin Level', 0.0, 5.0, 0.0, 1.0, "Protein concentrations observed in urinalysis evaluation."),
        'su': ('Numerical', col2, 'Sugar Level', 0.0, 5.0, 0.0, 1.0, "Glucose level footprint track within urine extract."),
        'bgr': ('Numerical', col3, 'Blood Glucose Random (mg/dl)', 50.0, 500.0, 120.0, 1.0, "Random blood glucose marker analysis."),
        'bu': ('Numerical', col1, 'Blood Urea (mg/dl)', 1.0, 400.0, 40.0, 1.0, "Waste metrics remaining post dietary nitrogen consumption processing."),
        'sc': ('Numerical', col2, 'Serum Creatinine (mg/dl)', 0.1, 15.0, 1.2, 0.1, "Kidney filtering efficiency indexing component values."),
        'sod': ('Numerical', col3, 'Sodium (mEq/L)', 100.0, 185.0, 138.0, 1.0, "Electrolyte fluid balancing parameter."),
        'pot': ('Numerical', col1, 'Potassium (mEq/L)', 2.0, 8.0, 4.2, 0.1, "Intracellular critical cation regulation monitoring matrix."),
        'hemo': ('Numerical', col2, 'Hemoglobin (gms)', 3.0, 18.0, 12.5, 0.1, "Oxygen transporting red blood cell structural profiling index."),
        'pcv': ('Numerical', col3, 'Packed Cell Volume', 15.0, 55.0, 40.0, 1.0, "Percentage content mapping of red blood volumes directly."),
        'wbcc': ('Numerical', col1, 'White Blood Cell Count (cells/cumm)', 2000.0, 20000.0, 8000.0, 100.0, "Immune structural count matrix monitoring metric."),
        'rbcc': ('Numerical', col2, 'Red Blood Cell Count (millions/cm)', 2.0, 8.0, 4.5, 0.1, "Total systemic transport cell aggregation mapping metrics."),
        
        # Categorical maps (0/1 designations map dynamically using nominal encoders logic from code tracking files)
        'rbc': ('Categorical', col3, 'Red Blood Cells', ['normal', 'abnormal'], "Abnormal signals localized structural challenges."),
        'pc': ('Categorical', col1, 'Pus Cell', ['normal', 'abnormal'], "Urinary tract or structural renal localized irritation flags."),
        'pcc': ('Categorical', col2, 'Pus Cell Clumps', ['notpresent', 'present'], "Clumping flags active infection indicators."),
        'ba': ('Categorical', col3, 'Bacteria', ['notpresent', 'present'], "Presence values track systemic pathogen replication."),
        'htn': ('Categorical', col1, 'Hypertension', ['no', 'yes'], "Systemic cardiovascular background loading pressure markers."),
        'dm': ('Categorical', col2, 'Diabetes Mellitus', ['no', 'yes'], "Metabolic profiling validation records index data."),
        'cad': ('Categorical', col3, 'Coronary Artery Disease', ['no', 'yes'], "Ischemic arterial stress historical tracking marker data."),
        'appet': ('Categorical', col1, 'Appetite Status', ['good', 'poor'], "Systemic uremic toxins metrics manifest directly in appetite shifts."),
        '
