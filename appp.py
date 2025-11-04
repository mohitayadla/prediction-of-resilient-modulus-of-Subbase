import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Resilient Modulus Predictor",
    page_icon="🏗️",
    layout="centered"
)

# Title and description
st.title("🏗️ Subgrade Resilient Modulus Predictor")
st.markdown("Predict the resilient modulus of subgrade soils based on soil properties")
st.markdown("---")

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'model.pkl' not found. Please ensure the file is in the same directory as this app.")
        return None

model = load_model()

# Input form
st.header("📋 Enter Soil Parameters")

col1, col2 = st.columns(2)

with col1:
    no_200_passing = st.number_input(
        "No. 200 Passing (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=50.0,
        help="Percentage passing No. 200 sieve"
    )
    
    liquid_limit = st.number_input(
        "Liquid Limit (LL)", 
        min_value=0.0, 
        max_value=100.0, 
        value=30.0,
        help="Liquid limit of soil"
    )
    
    plastic_limit = st.number_input(
        "Plastic Limit (PL)", 
        min_value=0.0, 
        max_value=100.0, 
        value=20.0,
        help="Plastic limit of soil"
    )
    
    plasticity_index = st.number_input(
        "Plasticity Index (PI)", 
        min_value=0.0, 
        max_value=100.0, 
        value=10.0,
        help="Plasticity Index (LL - PL)"
    )
    
    aashto_soil_class = st.selectbox(
        "AASHTO Soil Class",
        options=["A-1-a", "A-1-b", "A-2-4", "A-2-5", "A-2-6", "A-2-7", 
                 "A-3", "A-4", "A-5", "A-6", "A-7-5", "A-7-6"],
        help="AASHTO soil classification"
    )

with col2:
    aashto_soil_class_exp = st.text_input(
        "AASHTO Soil Class (Expanded)", 
        value="A-2-4(0)",
        help="Expanded AASHTO classification with group index"
    )
    
    spec_gravity = st.number_input(
        "Specific Gravity", 
        min_value=1.5, 
        max_value=3.5, 
        value=2.65,
        help="Specific gravity of soil solids"
    )
    
    max_lab_dry_density = st.number_input(
        "Max Lab Dry Density (pcf)", 
        min_value=80.0, 
        max_value=150.0, 
        value=120.0,
        help="Maximum dry density from lab compaction test"
    )
    
    optimum_lab_moisture = st.number_input(
        "Optimum Lab Moisture (%)", 
        min_value=0.0, 
        max_value=40.0, 
        value=12.0,
        help="Optimum moisture content from lab test"
    )
    
    hydraulic_conductivity = st.number_input(
        "Hydraulic Conductivity (cm/s)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.001,
        format="%.6f",
        help="Hydraulic conductivity of soil"
    )

st.markdown("---")

# Prediction button
if st.button("🔍 Predict Resilient Modulus", type="primary", use_container_width=True):
    if model is not None:
        # Create input dataframe with exact feature names
        input_data = pd.DataFrame({
            'NO_200_PASSING': [no_200_passing],
            'LIQUID_LIMIT': [liquid_limit],
            'PLASTIC_LIMIT': [plastic_limit],
            'PLASTICITY_INDEX': [plasticity_index],
            'AASHTO_SOIL_CLASS': [aashto_soil_class],
            'AASHTO_SOIL_CLASS_EXP': [aashto_soil_class_exp],
            'SPEC_GRAVITY': [spec_gravity],
            'MAX_LAB_DRY_DENSITY': [max_lab_dry_density],
            'OPTIMUM_LAB_MOISTURE': [optimum_lab_moisture],
            'HYDRAULIC_CONDUCTIVITY': [hydraulic_conductivity]
        })
        
        try:
            # Make prediction
            prediction = model.predict(input_data)
            
            # Display result
            st.success("✅ Prediction Complete!")
            st.metric(
                label="Predicted Resilient Modulus", 
                value=f"{prediction[0]:,.2f} psi"
            )
            
            # Additional info
            st.info(f"📊 Equivalent: {prediction[0] * 0.00689476:.2f} MPa")
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("💡 Make sure your model was trained with these exact feature names and order")
    else:
        st.warning("⚠️ Please upload the model file first")

# Footer
st.markdown("---")
st.markdown("### 📝 Instructions for Deployment")
st.markdown("""
1. Save this code as `app.py`
2. Place your `model.pkl` file in the same folder
3. Create a `requirements.txt` file with:
   ```
   streamlit
   pandas
   scikit-learn
   numpy
   ```
4. Upload to GitHub and deploy on Streamlit Cloud!
""")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app predicts the **Resilient Modulus** of subgrade soils based on standard soil properties.
    
    **Input Parameters:**
    - Gradation properties
    - Atterberg limits
    - AASHTO classification
    - Compaction characteristics
    - Hydraulic properties
    
    **Model:** Machine Learning predictor trained on laboratory test data
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")