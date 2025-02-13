import streamlit as st
import pickle
import numpy as np

diabetes_model = pickle.load(open("diabetes_prediction_model.pkl", "rb"))
heart_model = pickle.load(open("heart_disease_model.pkl", "rb"))
parkinsons_model = pickle.load(open("parkinsons_model.pkl", "rb"))

st.title("Health Prediction System")
disease = st.sidebar.selectbox("Select Disease to Predict", ["Diabetes", "Heart Disease", "Parkinson's Disease"])

if disease == "Diabetes":
    st.subheader("Diabetes Prediction")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, step=1)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, step=1)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, step=1)

    if st.button("Predict"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        prediction = diabetes_model.predict(input_data)
        st.success("Diabetes Detected" if prediction[0] == 1 else "No Diabetes Detected")

elif disease == "Heart Disease":
    st.subheader("Heart Disease Prediction")
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.number_input("Chest Pain Type", min_value=0, max_value=3, step=1)
    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, step=1)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, step=1)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, step=1)
    exang = st.selectbox("Exercise-Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.number_input("Slope", min_value=0, max_value=2, step=1)
    ca = st.number_input("Major Vessels Colored", min_value=0, max_value=4, step=1)
    thal = st.number_input("Thalassemia", min_value=0, max_value=3, step=1)

    if st.button("Predict"):
        input_data = np.array([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = heart_model.predict(input_data)
        st.success("Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected")

elif disease == "Parkinson's Disease":
    st.subheader("Parkinson's Disease Prediction")
    
    fo = st.number_input("MDVP:Fo(Hz)")
    fhi = st.number_input("MDVP:Fhi(Hz)")
    flo = st.number_input("MDVP:Flo(Hz)")
    jitter_perc = st.number_input("MDVP:Jitter(%)")
    jitter_abs = st.number_input("MDVP:Jitter(Abs)")
    rap = st.number_input("MDVP:RAP")
    ppq = st.number_input("MDVP:PPQ")
    ddp = st.number_input("Jitter:DDP")
    shimmer = st.number_input("MDVP:Shimmer")
    shimmer_db = st.number_input("MDVP:Shimmer(dB)")
    apq3 = st.number_input("Shimmer:APQ3")
    apq5 = st.number_input("Shimmer:APQ5")
    apq = st.number_input("MDVP:APQ")
    dda = st.number_input("Shimmer:DDA")
    nhr = st.number_input("NHR")
    hnr = st.number_input("HNR")
    rpde = st.number_input("RPDE")
    dfa = st.number_input("DFA")
    spread1 = st.number_input("Spread1")
    spread2 = st.number_input("Spread2")
    d2 = st.number_input("D2")
    ppe = st.number_input("PPE")

    if st.button("Predict"):
        input_data = np.array([[fo, fhi, flo, jitter_perc, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
        prediction = parkinsons_model.predict(input_data)
        st.success("Parkinson's Disease Detected" if prediction[0] == 1 else "No Parkinson's Disease Detected")
