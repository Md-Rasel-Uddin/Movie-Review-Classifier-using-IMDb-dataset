# importing library

import streamlit as st
import joblib
from transformers import pipeline

# Setting the title of the application

st.title("IMDB Movie Review Sentiment")

review = st.text_area("Enter a movie review:", height=200)
model_type = st.selectbox("Choose model:", ["Logistic Regression", "DistilBERT"])

if st.button("Predict"):
    if model_type == "Logistic Regression":
        # Load saved classifier and TF-IDF
        clf = joblib.load('best_model_lr.pkl')
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        vect = tfidf.transform([review])
        pred = clf.predict(vect)[0]
        sentiment = "Positive" if pred == 1 else "Negative"
    else:
        # Use HuggingFace pipeline for DistilBERT
        classifier = pipeline("sentiment-analysis", model="distilbert_model", tokenizer="distilbert_tokenizer")
        result = classifier(review[:512])[0]  # Truncate if needed
        sentiment = result['label']  # e.g. 'POSITIVE' or 'NEGATIVE'
    
    st.write(f"**Sentiment:** {sentiment}")
