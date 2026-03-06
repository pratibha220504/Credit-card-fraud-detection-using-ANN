import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("fraud_detection_project.h5")

# Load dataset
df = pd.read_csv("creditcard.csv")

st.title("💳 Credit Card Fraud Detection System")

st.subheader("Enter Transaction Details")

# User Inputs (Only Important Ones)
time = st.number_input("Transaction Time (seconds)", min_value=0.0)
amount = st.number_input("Transaction Amount", min_value=0.0)

if st.button("Predict Transaction Status"):

    # Take random row from dataset
    random_row = df.sample(1)

    # Replace user entered values
    random_row["Time"] = time
    random_row["Amount"] = amount

    # Drop target column
    input_data = random_row.drop("Class", axis=1)

    # Convert to numpy
    input_array = input_data.values

    # Prediction
    prediction = model.predict(input_array)

    if prediction[0][0] > 0.5:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")
