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
    .stApp { background-color: #fcfdfe; }
    div[data-testid="stMetricContainer"] {
        background-color: #ffffff;
        border: 1px solid #eef2f6;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    h1, h2, h3 {
        color: #1e293b !important;
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 700;
    }
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
# 2. Heavy Resource Loading (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load("best_ckd_model.pkl")
        scaler = joblib.load("scaler.pkl")
        features = joblib.load("selected_features.pkl")
        return model, scaler, features
    except FileNotFoundError as e:
        st.error(f"⚠️ Asset load failure! Please ensure 'best_ckd_model.pkl', 'scaler.pkl', and 'selected_features.pkl' are in the repository. Details: {e}")
        return None, None, None

model, scaler, selected_features = load_ml_assets()

# -----------------------------------------------------------------------------
# 3. Application Workflow & Architecture
# -----------------------------------------------------------------------------
def main():
    if model is None or scaler is None or selected_features is None:
        st.warning("Application halted. Waiting for ML files to be uploaded...")
        return

    # Sidebar
    st.sidebar.title("Clinical Interface")
    st.sidebar.markdown("---")
    app_mode = st.sidebar.radio("Navigation", ["Diagnostic Dashboard", "About Framework"])

    if app_mode == "About Framework":
        show_research_info()
        return

    # Main UI Title
    st.title("🩺 Chronic Kidney Disease (CKD) Intelligence Workspace")
    st.markdown("Advanced ensemble architecture for patient stratification and screening assistance.")
    st.markdown("---")
    st.subheader("📋 Patient Diagnostic Panel Entry")

    input_data = {}
    col1, col2, col3 = st.columns(3)

    # Feature Configurations for the UI
    feature_config = {
        'age': ('Numerical', col1, 'Age (Years)', 1.0, 120.0, 50.0, 1.0, "Patient chronologic lifespan."),
        'bp': ('Numerical', col2, 'Blood Pressure (mm/Hg)', 50.0, 180.0, 80.0, 10.0, "Diastolic/Systolic trend."),
        'sg': ('Numerical', col3, 'Specific Gravity', 1.005, 1.025, 1.015, 0.005, "Urine density."),
        'al': ('Numerical', col1, 'Albumin Level', 0.0, 5.0, 0.0, 1.0, "Protein concentrations."),
        'su': ('Numerical', col2, 'Sugar Level', 0.0, 5.0, 0.0, 1.0, "Glucose level footprint."),
        'bgr': ('Numerical', col3, 'Blood Glucose Random', 50.0, 500.0, 120.0, 1.0, "Random blood glucose."),
        'bu': ('Numerical', col1, 'Blood Urea (mg/dl)', 1.0, 400.0, 40.0, 1.0, "Waste metrics."),
        'sc': ('Numerical', col2, 'Serum Creatinine', 0.1, 15.0, 1.2, 0.1, "Kidney filtering efficiency."),
        'sod': ('Numerical', col3, 'Sodium (mEq/L)', 100.0, 185.0, 138.0, 1.0, "Electrolyte fluid balancing."),
        'pot': ('Numerical', col1, 'Potassium (mEq/L)', 2.0, 8.0, 4.2, 0.1, "Intracellular critical cation."),
        'hemo': ('Numerical', col2, 'Hemoglobin (gms)', 3.0, 18.0, 12.5, 0.1, "Oxygen transporting red blood cell."),
        'pcv': ('Numerical', col3, 'Packed Cell Volume', 15.0, 55.0, 40.0, 1.0, "Percentage mapping of red blood volumes."),
        'wbcc': ('Numerical', col1, 'White Blood Cell Count', 2000.0, 20000.0, 8000.0, 100.0, "Immune structural count."),
        'rbcc': ('Numerical', col2, 'Red Blood Cell Count', 2.0, 8.0, 4.5, 0.1, "Total systemic transport cell."),
        'rbc': ('Categorical', col3, 'Red Blood Cells', ['normal', 'abnormal'], "Structural challenges."),
        'pc': ('Categorical', col1, 'Pus Cell', ['normal', 'abnormal'], "Urinary tract irritation."),
        'pcc': ('Categorical', col2, 'Pus Cell Clumps', ['notpresent', 'present'], "Active infection indicators."),
        'ba': ('Categorical', col3, 'Bacteria', ['notpresent', 'present'], "Systemic pathogen replication."),
        'htn': ('Categorical', col1, 'Hypertension', ['no', 'yes'], "Cardiovascular background."),
        'dm': ('Categorical', col2, 'Diabetes Mellitus', ['no', 'yes'], "Metabolic validation."),
        'cad': ('Categorical', col3, 'Coronary Artery Disease', ['no', 'yes'], "Ischemic arterial stress."),
        'appet': ('Categorical', col1, 'Appetite Status', ['good', 'poor'], "Systemic uremic toxins manifest."),
        'pe': ('Categorical', col2, 'Pedal Edema', ['no', 'yes'], "Distal lower limb swelling."),
        'ane': ('Categorical', col3, 'Anemia Presence', ['no', 'yes'], "Anemic statuses.")
    }

    # Generate layout iteratively matching EXACTLY what feature selection dictated
    for feature in selected_features:
        if feature in feature_config:
            cfg = feature_config[feature]
            if cfg[0] == 'Numerical':
                _, column, label, min_v, max_v, def_v, step_v, help_t = cfg
                input_data[feature] = column.number_input(
                    label=label, min_value=min_v, max_value=max_v, value=def_v, step=step_v, help=help_t
                )
            elif cfg[0] == 'Categorical':
                _, column, label, options, help_t = cfg
                choice = column.selectbox(label=label, options=options, help=help_t)
                input_data[feature] = options.index(choice)
        else:
            input_data[feature] = st.sidebar.number_input(f"Raw Input: {feature}", value=0.0)

    # -----------------------------------------------------------------------------
    # 4. Pipeline Scaling & Predictive Inference
    # -----------------------------------------------------------------------------
    st.markdown("---")
    
    if st.button("🔬 Execute Diagnostic Prediction Sequence", type="primary", use_container_width=True):
        
        raw_df = pd.DataFrame([input_data])[selected_features]
        numerical_cols_src = ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc", "sg", "al", "su"]
        active_numerical_features = [feat for feat in selected_features if feat in numerical_cols_src]
        processed_df = raw_df.copy()
        
        if len(active_numerical_features) > 0:
            # Recreate structural array matching the exact 14 features the standard scaler was trained on
            dummy_full_row = np.zeros((1, len(numerical_cols_src)))
            for sub_idx, f_name in enumerate(numerical_cols_src):
                if f_name in input_data:
                    dummy_full_row[0, sub_idx] = input_data[f_name]
            
            scaled_full_matrix = scaler.transform(dummy_full_row)
            for sub_idx, f_name in enumerate(numerical_cols_src):
                if f_name in selected_features:
                    processed_df[f_name] = scaled_full_matrix[0, sub_idx]

        # Generate prediction
        prediction = model.predict(processed_df)[0]
        has_proba = hasattr(model, "predict_proba")
        probability = model.predict_proba(processed_df)[0][1] if has_proba else None

        # -----------------------------------------------------------------------------
        # 5. UI Reporting Output
        # -----------------------------------------------------------------------------
        st.subheader("📊 Analytical Diagnostic Summary Output")
        res_col1, res_col2 = st.columns([2, 3])

        with res_col1:
            if prediction == 1:
                st.error("### 🟥 Finding: CKD Positivity Detected")
                st.markdown("The architecture evaluated structural risk indicators matching classifications for **Chronic Kidney Disease (CKD)**. Immediate clinical review is recommended.")
            else:
                st.success("### 🟩 Finding: No CKD Signatures Detected")
                st.markdown("The processed biometric structures align strongly within the **Non-Chronic Kidney Disease (Not CKD)** range parameters.")

        with res_col2:
            st.metric(
                label="Target Result Classification", 
                value=f"{'CKD Positive [Class 1]' if prediction == 1 else 'Not CKD Normal [Class 0]'}"
            )
            if has_proba:
                st.write("**Model Classification Confidence Metric:**")
                st.progress(float(probability))
                st.caption(f"Calculated certitude matching active risk class distribution: {probability * 100:.2f}%")

def show_research_info():
    st.subheader("🔬 Framework Architecture Details")
    st.markdown("""
    * **Data Preprocessing Execution:** Iterative Multivariable MICE architecture.
    * **Imbalance Corrections Layer:** SMOTE matching classes.
    * **Feature Filtering:** Recursive Feature Elimination (RFE) established a streamlined footprint.
    * **Core Model Ecosystem:** Includes Random Forest, XGBoost, LightGBM, and Voting structures.
    """)

if __name__ == "__main__":
    main()
