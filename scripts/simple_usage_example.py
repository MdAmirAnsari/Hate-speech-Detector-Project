#!/usr/bin/env python3
"""
Simple Usage Example for Hate Speech Detection
This shows how to use the trained models in your own code.
"""

import pickle
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_text(text):
    """Simple text cleaning function."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    """Preprocess text for prediction."""
    # Clean text
    cleaned = clean_text(text)
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(cleaned)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)

def predict_hate_speech(text):
    """Predict hate speech for a given text."""
    # Load the best model (Task 1 - Binary classification)
    with open('../models/best_model_task_1.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Transform using the model's vectorizer
    text_vector = model_data['vectorizer'].transform([processed_text])
    
    # Make prediction
    prediction = model_data['model'].predict(text_vector)[0]
    probabilities = model_data['model'].predict_proba(text_vector)[0]
    
    # Get confidence scores
    prob_dict = dict(zip(model_data['labels'], probabilities))
    
    return {
        'text': text,
        'prediction': prediction,
        'confidence': max(probabilities),
        'probabilities': prob_dict
    }

# Example usage
if __name__ == "__main__":
    # Test with some examples
    test_texts = [
        "I love this movie!",
        "This is terrible, I hate it",
        "You people are disgusting",
        "Have a great day everyone!",
        "This politician is an idiot"
    ]
    
    print("HATE SPEECH DETECTION - SIMPLE USAGE")
    print("=" * 50)
    
    for text in test_texts:
        result = predict_hate_speech(text)
        print(f"\nText: {text}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Probabilities: {result['probabilities']}")
        print("-" * 50)
