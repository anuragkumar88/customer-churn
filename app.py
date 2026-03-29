import streamlit as st

st.title("Customer Churn Prediction")

age = st.slider("Age", 18, 80)
balance = st.number_input("Balance")

if st.button("Predict"):
    st.write("Prediction: Likely to Churn")