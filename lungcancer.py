import pickle
import pandas as pd
from matplotlib import pyplot as plt
import streamlit as st
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import tree
import seaborn as sns

def predict(x, y, features):
    model = pickle.load(open('klasifikasi_kanker.sav', 'rb'))
    prediction = model.predict([features])
    return prediction[0]

def plot_confusion_matrix(model, x, y):
    labels = np.unique(y)
    cm = confusion_matrix(y, model.predict(x), labels=labels)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

def app(df, x, y):
    st.title("Prediksi Kanker Paru-paru")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'])
        smoking = st.selectbox('Smoking', ['No', 'Yes'])
        yellow_fingers = st.selectbox('Yellow Fingers', ['No', 'Yes'])
        anxiety = st.selectbox('Anxiety', ['No', 'Yes'])
        peer_pressure = st.selectbox('Peer Pressure', ['No', 'Yes'])
        chronic_disease = st.selectbox('Chronic Disease', ['No', 'Yes'])
        fatigue = st.selectbox('Fatigue', ['No', 'Yes'])
    with col2:
        allergy = st.selectbox('Allergy', ['No', 'Yes'])
        wheezing = st.selectbox('Wheezing', ['No', 'Yes'])
        alcohol_consuming = st.selectbox('Alcohol Consuming', ['No', 'Yes'])
        coughing = st.selectbox('Coughing', ['No', 'Yes'])
        shortness_of_breath = st.selectbox('Shortness of Breath', ['No', 'Yes'])
        swallowing_difficulty = st.selectbox('Swallowing Difficulty', ['No', 'Yes'])
        chest_pain = st.selectbox('Chest Pain', ['No', 'Yes'])

    if st.button('Prediksi'):
        features = [
            gender, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy,
            wheezing, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, chest_pain
        ]
        features_encoded = [1 if f == 'Yes' else 0 for f in features]
        prediction = predict(x, y, features_encoded)

        if prediction == 1:
            st.warning("Orang tersebut rentan terkena penyakit kanker paru-paru")
        else:
            st.success("Orang tersebut relatif aman dari penyakit kanker paru-paru")
        
        model = pickle.load(open('klasifikasi_kanker.sav', 'rb'))
        score = model.score(x, y)
        st.write("Model yang digunakan memiliki tingkat akurasi " + str(score * 100) + "%")

    if st.checkbox("Plot Confusion Matrix"):
        model = pickle.load(open('klasifikasi_kanker.sav', 'rb'))
        plot_confusion_matrix(model, x, y)

    if st.checkbox("Plot Decision Tree"):
        model = pickle.load(open('klasifikasi_kanker.sav', 'rb'))
        dot_data = tree.export_graphviz(
            decision_tree=model, max_depth=5, out_file=None, filled=True, rounded=True,
            feature_names=x.columns, class_names=['notcancer', 'cancer']
        )
        st.graphviz_chart(dot_data)

df = pd.read_csv('lung_cancer.csv')

x = df[["GENDER", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC_DISEASE", "FATIGUE ",
            "ALLERGY ", "WHEEZING", "ALCOHOL_CONSUMING", "COUGHING", "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY",	
            "CHEST_PAIN"]]
y = df[['LUNG_CANCER']]

app(df, x, y)