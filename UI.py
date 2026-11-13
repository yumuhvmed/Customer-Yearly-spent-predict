import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="E-commerce Model",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 E-commerce Prediction")
st.write("Model that predict Yearly amount that customer spent")
st.write("Enter data then click \"Predict\"")

@st.cache_resource
def load_model():
    model = joblib.load('e-commerce.joblib')
    return model

model = load_model()

st.sidebar.header("🎛️ Model Inputs")

def user_input_features():
    avg_session_length = st.sidebar.number_input("Avg. Session Length", value=0.0)
    time_on_app = st.sidebar.number_input("Time on App", value=0.0)
    time_on_website = st.sidebar.number_input("Time on Website", value=0.0)
    length_of_membership = st.sidebar.number_input("Length of Membership", value=0.0)

    data = {
        'Avg. Session Length': avg_session_length,
        'Time on App': time_on_app,
        'Time on Website': time_on_website,
        'Length of Membership': length_of_membership
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("📋 Inputs:")
st.dataframe(input_df)

if st.button("🚀 Predict", type="primary"):
    prediction = model.predict(input_df)
    
    st.success("✅ Success!")
    
    st.metric(label="Predicted Value:", value=f"${prediction[0]:,.2f}")