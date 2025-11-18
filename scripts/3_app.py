import streamlit as st
import polars as pl
import joblib
import pandas as pd
from pathlib import Path
from PIL import Image

# --- 1. SETUP PATHS ---
try:
    PROJECT_ROOT = Path(__file__).parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()

PROCESSED_DATA_DIR = PROJECT_ROOT / "3_processed_data"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "4_model_outputs"

MODEL_PATH = MODEL_OUTPUT_DIR / "lgbm_model.joblib"
SHAP_PLOT_PATH = MODEL_OUTPUT_DIR / "shap_summary_plot.png"
PARQUET_PATH = PROCESSED_DATA_DIR / "final_features.parquet"

# --- 2. CONFIGURATION ---
st.set_page_config(
    page_title="NYC 311 Fairness Audit",
    page_icon="⚖️",
    layout="wide"
)

# --- 3. LOAD RESOURCES (CACHED) ---
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Run Layer 2 first!")
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_reference_data():
    """
    Load a tiny sample of data just to get the column names
    and categories for the dropdown menus.
    """
    if not PARQUET_PATH.exists():
        st.error("Data not found. Run Layer 1 first!")
        return None
    
    # Scan and fetch unique values for dropdowns (efficiently)
    lazy_df = pl.scan_parquet(PARQUET_PATH)
    
    # We need lists of options for our dropdowns
    options = {}
    categorical_cols = ['Complaint Type', 'Borough', 'Agency', 'Descriptor']
    
    for col in categorical_cols:
        # fetch top 50 most common values for dropdowns
        options[col] = lazy_df.select(col).group_by(col).count().sort("count", descending=True).limit(50).collect().get_column(col).to_list()
        
    return options

# --- 4. MAIN APP UI ---

st.title("⚖️ AI Audit: NYC 311 Service Equity")
st.markdown("""
**Objective:** This system audits NYC 311 response times for systemic socioeconomic bias.
It uses **Explainable AI (SHAP)** to isolate the impact of neighborhood income on government service speed.
""")

# Load resources
model = load_model()
options = load_reference_data()

if model and options:
    
    # Create Tabs
    tab1, tab2 = st.tabs(["Bias Audit Report", "What-If Simulator"])

    # --- TAB 1: STATIC AUDIT REPORT ---
    with tab1:
        st.header("Systemic Bias Analysis")
        st.markdown("The plot below shows the **SHAP values** for the model. It reveals the 'marginal contribution' of each feature to the predicted resolution time.")
        
        if SHAP_PLOT_PATH.exists():
            image = Image.open(SHAP_PLOT_PATH)
            st.image(image, caption="Global Feature Importance (SHAP Summary)", use_container_width=True)
            
            st.info("""
            **How to read this:**
            * **Red dots** = High feature value (e.g., High Income).
            * **Blue dots** = Low feature value (e.g., Low Income).
            * **Left side** = Faster resolution time.
            * **Right side** = Slower resolution time.
            
            **The Smoking Gun:** If Red dots for 'Median_Income' are to the left, it proves wealthy areas get faster service.
            """)
        else:
            st.warning("Audit plot not found. Run Layer 2 to generate it.")

    # --- TAB 2: INTERACTIVE SIMULATOR ---
    with tab2:
        st.header("Real-Time Prediction Simulator")
        st.markdown("Adjust the parameters to see how the AI predicts service times changes based on demographics.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Location & Demographics")
            borough = st.selectbox("Borough", options['Borough'])
            median_income = st.slider("Median Household Income ($)", 15_000, 200_000, 65_000, step=5000)
            num_households = st.slider("Population Density (Households)", 100, 5000, 1500)
            
        with col2:
            st.subheader("The Complaint")
            agency = st.selectbox("Agency", options['Agency'])
            complaint = st.selectbox("Complaint Type", options['Complaint Type'])
            descriptor = st.selectbox("Descriptor", options['Descriptor'])
            
        with col3:
            st.subheader("Time of Incident")
            day_of_week = st.selectbox("Day of Week (0=Mon, 6=Sun)", [0,1,2,3,4,5,6], index=2)
            hour = st.slider("Hour of Day (0-23)", 0, 23, 14)
            month = st.selectbox("Month", range(1,13), index=5)

        # Create the Input DataFrame
        # MUST match the order and columns used in Layer 2 training!
        input_data = pd.DataFrame({
            'Latitude': [40.7],   # Default centroid
            'Longitude': [-73.9], # Default centroid
            'Created_DayOfWeek': [day_of_week],
            'Created_Month': [month],
            'Created_Hour': [hour],
            'Median_Income': [median_income],
            'Mean_Income': [median_income * 1.2], # Approximate logic
            'Num_Households': [num_households],
            'Agency': [agency],
            'Complaint Type': [complaint],
            'Descriptor': [descriptor],
            'City': ['NEW YORK'], # Default
            'Community Board': ['01 MANHATTAN'], # Default
            'Borough': [borough]
        })

        # Cast categoricals just like in training
        cat_cols = ['Agency', 'Complaint Type', 'Descriptor', 'City', 'Community Board', 'Borough']
        for c in cat_cols:
            input_data[c] = input_data[c].astype('category')

        # Prediction Button
        if st.button("Predict Resolution Time", type="primary"):
            prediction = model.predict(input_data)[0]
            
            st.success(f"### Predicted Time to Close: {prediction:.2f} Hours")
            
            if prediction < 24:
                st.caption("Very Fast Response (Less than 1 day)")
            elif prediction > 100:
                st.caption("Slow Response (More than 4 days)")