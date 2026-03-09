import streamlit as st
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier


# --------- TITLE ----------
st.markdown("---")
st.title("NLP Model Deployment App 🚀")
st.write("Clasificación de texto usando dataset 20 Newsgroups")


# --------- SIDEBAR ----------
st.sidebar.title("Opciones")

methods = ["Naive Bayes", "SVM"]
selected_method = st.sidebar.selectbox("Elige el modelo:", methods)


# --------- INPUT TEXT ----------
st.write("## ✍️ Escribe un texto para clasificar")

user_text = st.text_area("Texto:", "This computer is fast and powerful")


# --------- BUTTON ----------
if st.button("Clasificar texto"):

    # Cargar dataset
    train_data = fetch_20newsgroups(subset="train", shuffle=True)

    # Pipeline según modelo
    if selected_method == "Naive Bayes":

        pipeline = Pipeline([
            ("bow", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("classifier", MultinomialNB())
        ])

    else:

        pipeline = Pipeline([
            ("bow", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("classifier", SGDClassifier(loss="hinge",
                                         penalty="l1",
                                         alpha=0.0005,
                                         l1_ratio=0.17))
        ])

    # Entrenar
    pipeline.fit(train_data.data, train_data.target)

    # Predecir texto del usuario
    prediction = pipeline.predict([user_text])

    # Mostrar resultado
    st.success("Categoría predicha:")
    st.write(train_data.target_names[prediction[0]])
    st.markdown("---")
    st.caption("App creada con Streamlit • NLP Deployment • MDC AI")
