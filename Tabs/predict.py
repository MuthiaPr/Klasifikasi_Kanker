import streamlit as st
from web_function import predict 

def app(df, x, y) :
    st.title("Prediksi")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Gender = st.selectbox('GENDER', ['Male', 'Female'])
        Smoking = st.selectbox('SMOKING', ['No', 'Yes'])
        Yellow_Fingers = st.selectbox('YELLOW_FINGERS', ['No', 'Yes'])
        Anxiety = st.selectbox('ANXIETY', ['No', 'Yes'])
        Peer_Pressure = st.selectbox('PEER_PRESSURE', ['No', 'Yes'])
    
    with col2:
        Chronic_Disease = st.selectbox('CHRONIC_DISEASE', ['No', 'Yes'])
        Fatigue = st.selectbox('FATIGUE ', ['No', 'Yes'])
        Allergy = st.selectbox('ALLERGY ', ['No', 'Yes'])
        Wheezing = st.selectbox('WHEEZING', ['No', 'Yes'])
        Alcohol_Consuming = st.selectbox('ALCOHOL_CONSUMING', ['No', 'Yes'])
    
    with col3:
        Coughing = st.selectbox('COUGHING', ['No', 'Yes'])
        Shortness_Of_Breath = st.selectbox('SHORTNESS_OF_BREATH', ['No', 'Yes'])
        Swallowing_Difficulty = st.selectbox('SWALLOWING_DIFFICULTY', ['No', 'Yes'])
        Chest_Pain = st.selectbox('CHEST_PAIN', ['No', 'Yes'])

    Gender = 1 if Gender == 'Male' else 0
    Smoking = 1 if Smoking == 'Yes' else 0
    Yellow_Fingers = 1 if Yellow_Fingers == 'Yes' else 0
    Anxiety = 1 if Anxiety == 'Yes' else 0
    Peer_Pressure = 1 if Peer_Pressure == 'Yes' else 0
    Chronic_Disease = 1 if Chronic_Disease == 'Yes' else 0
    Fatigue = 1 if Fatigue == 'Yes' else 0
    Allergy = 1 if Allergy == 'Yes' else 0
    Wheezing = 1 if Wheezing == 'Yes' else 0
    Alcohol_Consuming = 1 if Alcohol_Consuming == 'Yes' else 0
    Coughing = 1 if Coughing == 'Yes' else 0
    Shortness_Of_Breath = 1 if Shortness_Of_Breath == 'Yes' else 0
    Swallowing_Difficulty = 1 if Swallowing_Difficulty == 'Yes' else 0
    Chest_Pain = 1 if Chest_Pain == 'Yes' else 0
    
    features = [Gender, Smoking, Yellow_Fingers, Anxiety, Peer_Pressure, Chronic_Disease, Fatigue, Allergy,
                Wheezing, Alcohol_Consuming, Coughing, Shortness_Of_Breath, Swallowing_Difficulty, Chest_Pain]
    
    if st.button("Prediksi"):
        prediction, score = predict(x, y, features)
        score = score
        st.info("Prediksi Sukses...")

        if prediction == 1:
            st.warning("Orang tersebut rentan terkena penyakit lung cancer")
        else:
            st.success("Orang tersebut relatif aman dari penyakit lung cancer")
    
    st.write("Model yang digunakan memiliki tingkat akurasi " + str(score * 100) + "%")