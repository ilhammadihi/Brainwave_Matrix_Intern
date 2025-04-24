import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load data
try:
    df = pd.read_csv("C:/Users/ILHAM/Brainwave_Matrix_Intern/data/news.csv")
except FileNotFoundError:
    print("Error: File 'fake_or_real_news.csv' not found.")
    exit()

# Basic text cleaning
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\d+', '', text)  
    return text.strip()

df['clean_text'] = df['text'].apply(clean_text)

# Split data
X = df['clean_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Prediction function
def predict_news(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0][1]
    return {
        'prediction': prediction,
        'confidence': proba if prediction == "REAL" else 1 - proba
    }

# Example usage
test_text = "This is a sample news article to classify."
print(predict_news(test_text))