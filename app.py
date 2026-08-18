import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .churn-high {
        color: #d62728;
        font-weight: bold;
    }
    .churn-low {
        color: #2ca02c;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and encoders with caching
@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and preprocessing objects with caching."""
    try:
        model = tf.keras.models.load_model('model.h5')
        
        with open('label_encoder_gender.pkl', 'rb') as file:
            label_encoder_gender = pickle.load(file)
        
        with open('onehot_encoder_geo.pkl', 'rb') as file:
            onehot_encoder_geo = pickle.load(file)
        
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        
        return model, label_encoder_gender, onehot_encoder_geo, scaler
    except FileNotFoundError as e:
        st.error(f" Error: Could not find model or encoder files. {str(e)}")
        st.stop()

# Load resources
model, label_encoder_gender, onehot_encoder_geo, scaler = load_model_and_encoders()

# App Title and Description
st.title("Customer Churn Prediction")
st.markdown("""
    This application predicts the likelihood of a customer churning (leaving the bank) 
    based on their profile and activity. Enter customer details below to get a prediction.
    """)

# Sidebar for inputs
st.sidebar.header("Customer Information")
st.sidebar.markdown("---")

# Organize inputs in the sidebar
col1_side, col2_side = st.sidebar.columns(2)

with col1_side:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0], 
                            help="Customer's geographic location")
    age = st.slider('Age', 18, 92, 40, help="Customer's age")
    tenure = st.slider('Tenure (years)', 0, 10, 5, help="Years as a customer")

with col2_side:
    gender = st.selectbox('Gender', label_encoder_gender.classes_, 
                         help="Customer's gender")
    num_of_products = st.slider('Number of Products', 1, 4, 1, 
                               help="Number of bank products owned")

st.sidebar.markdown("---")
st.sidebar.subheader("Financial Information")

col1_fin, col2_fin = st.sidebar.columns(2)

with col1_fin:
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, 
                                  value=650, step=10, 
                                  help="Customer's credit score")
    balance = st.number_input('Balance ($)', min_value=0.0, value=50000.0, step=1000.0,
                             help="Account balance")

with col2_fin:
    estimated_salary = st.number_input('Estimated Salary ($)', min_value=0.0, 
                                       value=100000.0, step=5000.0,
                                       help="Estimated annual salary")

st.sidebar.markdown("---")
st.sidebar.subheader("Account Status")

col1_status, col2_status = st.sidebar.columns(2)

with col1_status:
    has_cr_card = st.selectbox('Has Credit Card?', ['Yes', 'No'], 
                              help="Does customer have a credit card?")
    has_cr_card_binary = 1 if has_cr_card == 'Yes' else 0

with col2_status:
    is_active_member = st.selectbox('Active Member?', ['Yes', 'No'],
                                   help="Is customer actively using their account?")
    is_active_member_binary = 1 if is_active_member == 'Yes' else 0

# Main content area
st.markdown("---")

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card_binary],
    'IsActiveMember': [is_active_member_binary],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled, verbose=0)
prediction_proba = prediction[0][0]
churn_probability = prediction_proba * 100

# Display Results in columns
col_pred, col_info = st.columns([2, 1])

with col_pred:
    st.subheader("Prediction Result")
    
    if prediction_proba > 0.5:
        # High churn risk
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 24px; margin-bottom: 10px;">⚠️ High Churn Risk</div>
                <div style="font-size: 48px; color: #d62728; font-weight: bold;">{churn_probability:.1f}%</div>
                <div style="font-size: 14px; color: #666; margin-top: 10px;">
                    The customer is <strong>likely to churn</strong>. Consider retention strategies.
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Low churn risk
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 24px; margin-bottom: 10px;">✅ Low Churn Risk</div>
                <div style="font-size: 48px; color: #2ca02c; font-weight: bold;">{churn_probability:.1f}%</div>
                <div style="font-size: 14px; color: #666; margin-top: 10px;">
                    The customer is <strong>unlikely to churn</strong>. Good customer retention outlook.
                </div>
            </div>
            """, unsafe_allow_html=True)

with col_info:
    st.subheader("Customer Summary")
    st.metric("Age", f"{age} years")
    st.metric("Tenure", f"{tenure} years")
    st.metric("Products", num_of_products)

# Display customer details
st.subheader("Customer Details")

detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)

with detail_col1:
    st.info(f"**Geography:** {geography}")
    st.info(f"**Gender:** {gender}")
    
with detail_col2:
    st.info(f"**Credit Score:** {credit_score}")
    st.info(f"**Has Credit Card:** {has_cr_card}")

with detail_col3:
    st.info(f"**Balance:** ${balance:,.2f}")
    st.info(f"**Active Member:** {is_active_member}")

with detail_col4:
    st.info(f"**Estimated Salary:** ${estimated_salary:,.2f}")

# Additional insights
st.markdown("---")
st.subheader("Insights")

insights = []
if age > 60:
    insights.append("**Age Factor:** Customer is over 60, which is often associated with higher churn risk.")
if balance == 0:
    insights.append(" **Balance:** Zero balance may indicate disengagement with the bank.")
if num_of_products < 2:
    insights.append(" **Cross-sell Opportunity:** Customer has less than 2 products. Consider cross-selling.")
if tenure < 2:
    insights.append("**New Customer:** Less than 2 years tenure. New customers need extra engagement.")
if is_active_member_binary == 0:
    insights.append(" **Activity:** Customer is not actively using their account. May need re-engagement.")

if insights:
    for insight in insights:
        st.markdown(f"- {insight}")
else:
    st.success("No specific risk factors identified.")
