import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Cache the model and preprocessor loading
@st.cache_resource
def load_model_and_preprocessor():
    with open('churn_model_xgb.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

# Load model and preprocessor
model, preprocessor = load_model_and_preprocessor()

# Define feature lists
categorical_cols = ['REGION', 'TENURE', 'MRG', 'TOP_PACK']
numerical_cols = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 
                 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'ZONE1', 'ZONE2', 
                 'REGULARITY', 'FREQ_TOP_PACK']

# Sample options for categorical features (adjust based on your dataset)
region_options = ['DAKAR', 'THIES', 'FATICK', 'KAOLACK', 'missing']
tenure_options = ['K > 24 month', 'I 18-21 month', 'J 21-24 month', 'missing']
mrg_options = ['NO', 'YES', 'missing']
top_pack_options = ['On net 200F=Unlimited _call24H', 'Data:1000F=5GB,7d', 
                    'Mixt 250F=Unlimited_call24H', 'On-net 1000F=10MilF;10d', 'missing']

# Sidebar
st.sidebar.markdown("""
# Expresso Telecom Churn Prediction
This app predicts customer churn for Expresso Telecom using an XGBoost model trained on customer demographics and behavior.
""")

# Main app
st.title("Expresso Telecom Churn Prediction")

# Single Prediction
st.header("Single Customer Prediction")
with st.form("single_prediction"):
    st.subheader("Enter Customer Details")
    
    # Numerical inputs (sliders with reasonable ranges)
    montant = st.slider("MONTANT (Recharge Amount)", 0.0, 50000.0, 5000.0, step=100.0)
    freq_rech = st.slider("FREQUENCE_RECH (Recharge Frequency)", 0.0, 100.0, 10.0, step=1.0)
    revenue = st.slider("REVENUE", 0.0, 50000.0, 5000.0, step=100.0)
    arpu_segment = st.slider("ARPU_SEGMENT", 0.0, 20000.0, 2000.0, step=100.0)
    frequence = st.slider("FREQUENCE (Activity Frequency)", 0.0, 100.0, 10.0, step=1.0)
    data_volume = st.slider("DATA_VOLUME", 0.0, 100000.0, 1000.0, step=100.0)
    on_net = st.slider("ON_NET (On-net Calls)", 0.0, 1000.0, 50.0, step=1.0)
    orange = st.slider("ORANGE (Orange Calls)", 0.0, 1000.0, 50.0, step=1.0)
    tigo = st.slider("TIGO (Tigo Calls)", 0.0, 1000.0, 10.0, step=1.0)
    zone1 = st.slider("ZONE1 (Zone 1 Calls)", 0.0, 100.0, 0.0, step=1.0)
    zone2 = st.slider("ZONE2 (Zone 2 Calls)", 0.0, 100.0, 0.0, step=1.0)
    regularity = st.slider("REGULARITY (Activity Regularity)", 0, 62, 30, step=1)
    freq_top_pack = st.slider("FREQ_TOP_PACK (Top Pack Frequency)", 0.0, 100.0, 5.0, step=1.0)
    
    # Categorical inputs (dropdowns)
    region = st.selectbox("REGION", region_options)
    tenure = st.selectbox("TENURE", tenure_options)
    mrg = st.selectbox("MRG", mrg_options)
    top_pack = st.selectbox("TOP_PACK", top_pack_options)
    
    submit = st.form_submit_button("Predict")
    
    if submit:
        # Create input DataFrame
        input_data = pd.DataFrame({
            'MONTANT': [montant],
            'FREQUENCE_RECH': [freq_rech],
            'REVENUE': [revenue],
            'ARPU_SEGMENT': [arpu_segment],
            'FREQUENCE': [frequence],
            'DATA_VOLUME': [data_volume],
            'ON_NET': [on_net],
            'ORANGE': [orange],
            'TIGO': [tigo],
            'ZONE1': [zone1],
            'ZONE2': [zone2],
            'REGULARITY': [regularity],
            'FREQ_TOP_PACK': [freq_top_pack],
            'REGION': [region],
            'TENURE': [tenure],
            'MRG': [mrg],
            'TOP_PACK': [top_pack]
        })
        
        # Preprocess input
        input_processed = preprocessor.transform(input_data)
        
        # Get feature names from preprocessor
        onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = onehot_encoder.get_feature_names_out(categorical_cols)
        feature_names = numerical_cols + list(cat_feature_names)
        
        # Convert to DataFrame
        input_processed = pd.DataFrame(input_processed, columns=feature_names)
        
        # Ensure all model features are present
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in input_processed.columns:
                input_processed[col] = 0
        input_processed = input_processed[model_features]  # Reorder columns
        
        # Predict
        prediction = model.predict(input_processed)[0]
        confidence = model.predict_proba(input_processed)[0][prediction]
        
        # Display results
        st.subheader("Prediction Result")
        st.write(f"Churn Prediction: {'Churned' if prediction == 1 else 'Not Churned'}")
        st.write(f"Confidence Score: {confidence:.2%}")

# Batch Prediction
st.header("Batch Prediction via CSV Upload")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    
    # Validate required columns
    required_cols = numerical_cols + categorical_cols
    if not all(col in batch_data.columns for col in required_cols):
        st.error(f"CSV must contain columns: {', '.join(required_cols)}")
    else:
        # Preprocess batch data
        batch_processed = preprocessor.transform(batch_data[required_cols])
        
        # Get feature names
        onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = onehot_encoder.get_feature_names_out(categorical_cols)
        feature_names = numerical_cols + list(cat_feature_names)
        
        # Convert to DataFrame
        batch_processed = pd.DataFrame(batch_processed, columns=feature_names)
        
        # Ensure all model features are present
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in batch_processed.columns:
                batch_processed[col] = 0
        batch_processed = batch_processed[model_features]  # Reorder columns
        
        # Predict
        predictions = model.predict(batch_processed)
        confidences = model.predict_proba(batch_processed)[:, 1]
        
        # Add predictions to DataFrame
        batch_data['PREDICTION'] = predictions
        batch_data['CONFIDENCE'] = confidences
        
        # Display results
        st.subheader("Batch Prediction Results")
        st.dataframe(batch_data)
        
        # Download link
        csv = batch_data.to_csv(index=False)
        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )