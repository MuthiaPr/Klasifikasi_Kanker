import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import tree
import streamlit as st
from web_function import train_model
import seaborn as sns
import numpy as np

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
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Visualisasi Prediksi Lung Cancer")

    if st.checkbox("Plot Confusion Matrix"):
        model, score = train_model(x, y)
        plot_confusion_matrix(model, x, y)

    if st.checkbox("Plot Decision Tree"):
        model, score = train_model(x, y)
        dot_data = tree.export_graphviz(
            decision_tree=model, max_depth=5, out_file=None, filled=True, rounded=True,
            feature_names=x.columns, class_names=['notcancer', 'cancer']
        )

        st.graphviz_chart(dot_data)
