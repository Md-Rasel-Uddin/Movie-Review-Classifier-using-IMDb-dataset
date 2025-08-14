IMDB Movie Reviews Sentiment Analysis - 
AI/ML Engineer Task Submission - 
Author: Md Rasel Uddin - 
Date: August 2025

<img width="1070" height="672" alt="image" src="https://github.com/user-attachments/assets/3f3ef619-13a1-4c36-b215-af5aa44bba9f" />

Got to this deployed version link: https://rasel-movie-review-classifier-using-imdb-dataset.streamlit.app/


This project demonstrates **sentiment analysis**  for IMDB movie reviews, classifying them as either Positive or Negative on movie reviews using two approaches:

1. **Classical Machine Learning (ML)** — Logistic Regression with TF-IDF features
2. **Deep Learning (DL)** — Fine-tuned DistilBERT model from Hugging Face Transformers

The solution includes traditional machine learning models and deep learning approaches, with hyperparameter optimization and interactive demo capabilities.
The application allows you to **choose the model (ML/DL)** interactively and test multiple reviews in one session.

---

🎯 Task Requirements Completed
✅ Step 1: Data Preparation

- Load IMDB dataset using Pandas
- Text cleaning: Remove HTML tags, punctuation, numbers
- Convert to lowercase
- Remove English stopwords using NLTK
- Split data: 80% train, 20% test

✅ Step 2: Model Training

Implemented multiple ML models:

- Logistic Regression
- Naive Bayes (MultinomialNB)
- Random Forest

- TF-IDF feature extraction with n-grams
- Hyperparameter optimization using GridSearchCV

✅ Step 3: Evaluation

- Comprehensive metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrix visualization with matplotlib/seaborn
- Model saved as .pkl files
- Parameter optimization for improved accuracy

✅ Step 4: Bonus - Deep Learning

- DistilBERT fine-tuning for sentiment classification
- Transformer-based approach with HuggingFace
- Model comparison: Traditional ML vs Deep Learning

🛠️ Technologies Used

Core Libraries:
+ pandas, numpy - Data manipulation
+ scikit-learn - Machine learning models and evaluation
+ nltk - Natural language processing
+ matplotlib, seaborn - Data visualization
+ joblib - Model persistence

Deep Learning:
+ transformers - HuggingFace transformers library
+ datasets - HuggingFace datasets
+ torch - PyTorch backend

---


## 📂 Project Structure

MovieReviewClassifier/
├── best_model_lr.pkl    # Trained Logistic Regression model
├── tfidf_vectorizer.pkl # Fitted TF-IDF vectorizer
├── distilbert_model/    # Fine-tuned DistilBERT model folder
├── distilbert_tokenizer/ # Tokenizer folder for the DistilBERT model
├── streamlit_app.py    # Main sentiment analysis interactive script
├── README.md            # This file
├── requirements.txt    # List of Python dependencies
├── streamlit_app.py    # Main sentiment analysis interactive script
├── Code-Md-Rasel-Uddin-MovieReviewClassifier.ipynb # the main code file
└── DemoScript.py        # Interactive script to test with input 

---

## 🛠 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Md-Rasel-Uddin/Movie-Review-Classifier-using-IMDb-dataset.git
   cd Movie-Review-Classifier-using-IMDb-dataset

2. Install Dependencies
   pip install -r requirements.txt


📊 Results

Traditional ML Models Performance:
Model	Accuracy	      Precision	Recall	F1-Score
Logistic Regression	0.8642	0.8580	0.8720	0.8649
Naive Bayes	         0.8421	0.8654	0.8115	0.8376
Random Forest	      0.8398	0.8456	0.8312	0.8383

Deep Learning Model Performance:
Model	      Accuracy	Precision	Recall	F1-Score
DistilBERT	0.9123	0.9087	0.9165	0.9126

Key Findings
Best Traditional ML Model: Logistic Regression (86.4% accuracy)
Best Overall Model: Fine-tuned DistilBERT (91.2% accuracy)

Optimal TF-IDF Configuration:
N-grams: (1, 2)
Max features: 20,000
Min document frequency: 2


🚀 Usage
Run the main script:

python sentiment_app.py

The script will:

-Ask you to choose ML or DL at the start

-Let you enter movie reviews repeatedly

-Show predictions until you type quit

*Example Interaction*

Choose model (ml/dl): ml
Enter 'quit' to stop reviewing.

Enter movie review: I absolutely loved this film!
Prediction (Logistic Regression): Positive

Choose model (ml/dl): dl
Enter movie review: This was terrible and boring.
Prediction (DistilBERT): NEGATIVE (score: 0.9987)

Enter movie review: quit
Exiting...

📜 Scripts

DemoScript.py

Interactive sentiment analysis using ML or DL.

Key features:

Preloads model and vectorizer/tokenizer

Interactive loop for multiple reviews

Model can be changed after each review

Shows both label and confidence score (for DL model)




📦 requirements.txt

datasets
joblib

🧠 Training Notes
-ML model trained with Logistic Regression on TF-IDF features from movie reviews dataset
-DL model is a fine-tuned distilbert-base-uncased on the same dataset for binary classification (POSITIVE, NEGATIVE)
