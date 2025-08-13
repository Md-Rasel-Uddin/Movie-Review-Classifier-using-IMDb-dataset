# demo.py
import joblib
import argparse
from transformers import pipeline

parser = argparse.ArgumentParser(description="Sentiment Analysis Demo")
parser.add_argument("--model", choices=["ml", "dl"], default="ml", help="Use classical ML or DistilBERT")
args = parser.parse_args()

review = input("Enter movie review: ")

if args.model == "ml":
    clf = joblib.load('best_model_lr.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    vect = tfidf.transform([review])
    pred = clf.predict(vect)[0]
    print("Prediction (Logistic Regression):", "Positive" if pred == 1 else "Negative")
else:
    classifier = pipeline("sentiment-analysis", model="distilbert_model", tokenizer="distilbert_tokenizer")
    result = classifier(review[:512])[0]
    print("Prediction (DistilBERT):", result['label'])
