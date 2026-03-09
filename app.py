import streamlit as st
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier


st.title("NLP Model Deployment App")

st.write("Clasificación de texto usando dataset 20 Newsgroups")

st.sidebar.title("Opciones")

methods = ["Naive Bayes", "SVM"]
selected_method = st.sidebar.selectbox("Elige el modelo:", methods)

if st.sidebar.button("Classify"):

    st.write("## Resultados")

    train_data = fetch_20newsgroups(subset="train", shuffle=True)
    test_data = fetch_20newsgroups(subset="test", shuffle=True)

    if selected_method == "Naive Bayes":

        st.write("Modelo seleccionado: Naive Bayes")

        pipeline = Pipeline([
            ("bow", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("classifier", MultinomialNB())
        ])

    else:

        st.write("Modelo seleccionado: SVM")

        pipeline = Pipeline([
            ("bow", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("classifier", SGDClassifier(loss="hinge",
                                         penalty="l1",
                                         alpha=0.0005,
                                         l1_ratio=0.17))
        ])

    pipeline.fit(train_data.data, train_data.target)

    predictions = pipeline.predict(test_data.data)

    accuracy = np.mean(predictions == test_data.target)

    st.success(f"Accuracy: {accuracy:.4f}")
