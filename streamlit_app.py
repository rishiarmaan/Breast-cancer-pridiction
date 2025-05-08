import streamlit as st
import numpy as np
import pickle

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🔬 Breast Cancer Prediction App")
st.markdown("Enter the required medical values below to check the prediction.")

mean_perimeter = st.number_input("Mean Perimeter", format="%.4f")
mean_concave_points = st.number_input("Mean Concave Points", format="%.4f")
worst_radius = st.number_input("Worst Radius", format="%.4f")
worst_perimeter = st.number_input("Worst Perimeter", format="%.4f")
worst_area = st.number_input("Worst Area", format="%.4f")
worst_concave_points = st.number_input("Worst Concave Points", format="%.4f")

if st.button("Predict"):
    input_data = np.array([
        mean_perimeter,
        mean_concave_points,
        worst_radius,
        worst_perimeter,
        worst_area,
        worst_concave_points
    ]).reshape(1, -1)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Prediction: Cancerous")
    else:
        st.success("Prediction: Not Cancerous")