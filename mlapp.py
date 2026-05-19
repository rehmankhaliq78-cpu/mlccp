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
        'pe': ('Categorical', col2, 'Pedal Edema', ['no', 'yes'], "Fluid distribution changes showing up via distal lower limb swelling."),
        'ane': ('Categorical', col3, 'Anemia Presence', ['no', 'yes'], "Erythropoietin production downfalls display tracking anemic statuses.")
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
                # LabelEncoder mapping translation layer
                # 'normal'/'notpresent'/'no'/'good' -> typically maps to lower index integers in pipeline configurations. 
                # To maintain safety matching structural patterns from source lines we force index matching:
                input_data[feature] = options.index(choice)
        else:
            # Fallback configuration handling for anomalous variables missing default indexing configuration lists
            st.sidebar.warning(f"Feature config structural profile missing for variant: '{feature}'")
            input_data[feature] = st.sidebar.number_input(f"Raw Input: {feature}", value=0.0)

    # -----------------------------------------------------------------------------
    # 4. Pipeline Scaling Transformation & Predictive Inference Process
    # -----------------------------------------------------------------------------
    st.markdown("---")
    action_col, spacer_col = st.columns([1, 2])
    
    if action_col.button("🔬 Execute Diagnostic Prediction Sequence", type="primary", use_container_width=True):
        
        # Step A: Transform raw user dictionary context directly into structural DataFrame row matching original layout arrays
        raw_df = pd.DataFrame([input_data])
        
        # Ensure strict index sequence alignment to avoid structural orientation issues inside the ML framework matrix 
        raw_df = raw_df[selected_features]

        # Step B: Segment processing types between continuous scaling and integer structures
        # In your pipeline configuration notebook scripts, continuous features are isolated via numerical_cols lists
        numerical_cols_src = ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc", "sg", "al", "su"]
        active_numerical_features = [feat for feat in selected_features if feat in numerical_cols_src]
        
        # Create a deep localized processing workspace
        processed_df = raw_df.copy()
        
        # Execute scaling transformation calculations exclusively atop continuous attributes active inside feature subset vectors
        if len(active_numerical_features) > 0:
            # We recreate spatial tracking vectors matching exact array dimensionality parameters expected by original `scaler.fit_transform` actions
            # A protective abstraction strategy mimics base feature environments to yield optimal standard normal metrics
            dummy_full_row = np.zeros((1, 24)) # 24 Base clinical parameters before selection
            for sub_idx, f_name in enumerate(numerical_cols_src):
                if f_name in input_data:
                    dummy_full_row[0, sub_idx] = input_data[f_name]
            
            # Extract standard transformed metrics out from calculated scalar array matrices
            scaled_full_matrix = scaler.transform(dummy_full_row)
            for sub_idx, f_name in enumerate(numerical_cols_src):
                if f_name in selected_features:
                    processed_df[f_name] = scaled_full_matrix[0, sub_idx]

        # Step C: Generate predictions via production system file
        prediction = model.predict(processed_df)[0]
        
        # Extract evaluation tracking metrics safe flags to account for multi-model configurations
        has_proba = hasattr(model, "predict_proba")
        probability = model.predict_proba(processed_df)[0][1] if has_proba else None

        # -----------------------------------------------------------------------------
        # 5. UI Reporting and Classification Output Panels
        # -----------------------------------------------------------------------------
        st.subheader("📊 Analytical Diagnostic Summary Output")
        
        res_col1, res_col2 = st.columns([2, 3])

        with res_col1:
            if prediction == 1:
                st.error("### 🟥 Diagnostic Finding: CKD Positivity Detected")
                st.markdown("""
                The ensemble architecture workflow evaluated structural risk indicators matching classifications for **Chronic Kidney Disease (CKD)**. 
                Immediate clinical review and secondary diagnostic confirmations are highly recommended.
                """)
            else:
                st.success("### 🟩 Diagnostic Finding: No CKD Signatures Detected")
                st.markdown("""
                The processed biometric structures and patient attributes align strongly within the **Non-Chronic Kidney Disease (Not CKD)** range parameters.
                """)

        with res_col2:
            st.metric(
                label="Target Categorical Result Code Extraction", 
                value=f"{'CKD Positive [Class 1]' if prediction == 1 else 'Not CKD Normal [Class 0]'}"
            )
            if has_proba:
                st.write("**Model Classification Confidence Metric:**")
                st.progress(float(probability))
                st.caption(f"Calculated computational certitude matching active risk class distribution: {probability * 100:.2f}%")
            else:
                st.info("💡 Confidence metrics disabled for hard-voting ensemble strategy structures.")


def show_research_info():
    st.subheader("🔬 Framework Architecture & Modeling Environment Details")
    st.markdown("""
    This digital system layer maps directly over structured data training sessions built using an optimized **Ensemble Stack**.
    
    #### Pipeline Blueprint Overview:
    * **Data Preprocessing Execution:** Implements handling via an Iterative Multivariable MICE architecture platform layout module.
    * **Imbalance Corrections Layer:** Utilized Synthetic Minority Over-sampling Technique (`SMOTE`) matching classes across optimal boundaries.
    * **Feature Filtering Dimension Reduction:** Handled via Recursive Feature Elimination (`RFE`) to establish a streamlined, 12-factor operational diagnostic footprint.
    * **Core Model Ecosystem Optimized File:** Select models include Random Forest architectures, Extreme Gradient Boosting (`XGBoost`), LightGBM variants, Stacked Classifier arrays, and general Voting structures.
    """)
    st.info("💡 For optimization modifications, adjust configuration values in the primary `ml_ccp_092.py` structural execution files.")

if __name__ == "__main__":
    main()