import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download NLTK stopwords
nltk.download('stopwords')

import os

# Print all files in the directory to check if 'sentiment_dataset.csv' is present
print(os.listdir("C:/Users/SIRI LASYA/OneDrive/Desktop/projects"))

# Load dataset
df = pd.read_csv("C:/Users/SIRI LASYA/OneDrive/Desktop/projects/sentiment_analysis.csv")

# Load dataset (ensure you have a CSV with text & sentiment columns)
df = pd.read_excel("sentiment_dataset.xlsx")

# Check dataset structure
print(df.head())

# Preprocessing function
def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(clean_text)
# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text']).toarray()
y = df['sentiment'].map({'positive': 1, 'negative': 0, 'neutral': 2})  # Convert sentiment to numeric labels

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

def predict_sentiment(text):
    cleaned_text = clean_text(text)
    transformed_text = vectorizer.transform([cleaned_text]).toarray()
    prediction = model.predict(transformed_text)
    sentiment_label = {1: "Positive", 0: "Negative", 2: "Neutral"}
    return sentiment_label[prediction[0]]

# Example Test
print(predict_sentiment("I love this product!"))
print(predict_sentiment("This is terrible"))