#!/usr/bin/env python3
"""
Hate Speech Detection Prediction Script
This script loads the trained models and provides predictions for new text samples.
"""

import pandas as pd
import numpy as np
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

class HateSpeechPredictor:
    """
    A class for predicting hate speech using trained models.
    """
    
    def __init__(self):
        self.models = {}
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.load_models()
    
    def load_models(self):
        """Load all trained models."""
        tasks = ['task_1', 'task_2', 'task_3']
        
        for task in tasks:
            try:
                with open(f'../models/best_model_{task}.pkl', 'rb') as f:
                    self.models[task] = pickle.load(f)
                print(f"Loaded model for {task}")
            except FileNotFoundError:
                print(f"Model file for {task} not found. Please run training first.")
    
    def clean_text(self, text):
        """Clean and preprocess text."""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove punctuation except for basic sentence structure
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove single characters
        text = re.sub(r'\b\w{1}\b', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_filter(self, text):
        """Tokenize text and remove stopwords."""
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter out stopwords and short tokens
        filtered_tokens = [
            token for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return filtered_tokens
    
    def preprocess_text(self, text):
        """Complete text preprocessing pipeline."""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and filter
        tokens = self.tokenize_and_filter(cleaned_text)
        
        # Return as string
        return ' '.join(tokens)
    
    def predict_single(self, text):
        """Predict hate speech for a single text sample."""
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        results = {}
        
        for task, model_data in self.models.items():
            # Transform text using the model's vectorizer
            text_vector = model_data['vectorizer'].transform([processed_text])
            
            # Make prediction
            prediction = model_data['model'].predict(text_vector)[0]
            
            # Get prediction probabilities if available
            if hasattr(model_data['model'], 'predict_proba'):
                probabilities = model_data['model'].predict_proba(text_vector)[0]
                prob_dict = dict(zip(model_data['labels'], probabilities))
            else:
                prob_dict = None
            
            results[task] = {
                'prediction': prediction,
                'probabilities': prob_dict,
                'model_name': model_data['model_name']
            }
        
        return results
    
    def predict_batch(self, texts):
        """Predict hate speech for multiple text samples."""
        results = []
        
        for text in texts:
            result = self.predict_single(text)
            result['original_text'] = text
            results.append(result)
        
        return results
    
    def explain_prediction(self, text):
        """Provide detailed explanation of the prediction."""
        results = self.predict_single(text)
        
        print(f"Original text: {text}")
        print(f"Processed text: {self.preprocess_text(text)}")
        print("\nPredictions:")
        print("-" * 50)
        
        task_names = {
            'task_1': 'Hate/Offensive Detection',
            'task_2': 'Hate Speech Classification',
            'task_3': 'Target Identification'
        }
        
        for task, result in results.items():
            print(f"\n{task_names[task]} ({result['model_name']}):")
            print(f"  Prediction: {result['prediction']}")
            
            if result['probabilities']:
                print("  Probabilities:")
                for label, prob in result['probabilities'].items():
                    print(f"    {label}: {prob:.4f}")
        
        return results

def main():
    """Main function to demonstrate the prediction system."""
    
    print("HATE SPEECH DETECTION - PREDICTION SYSTEM")
    print("=" * 60)
    
    # Initialize predictor
    predictor = HateSpeechPredictor()
    
    if not predictor.models:
        print("No models loaded. Please run training first.")
        return
    
    # Test examples
    test_examples = [
        "I love spending time with my friends and family.",
        "This politician is completely incompetent and should resign.",
        "All people from that country are criminals and should be deported.",
        "The movie was terrible, what a waste of time.",
        "Women are inferior to men and shouldn't have equal rights.",
        "Great job on the project, keep up the good work!",
        "I hate this stupid government policy.",
        "Those people are animals and deserve to be punished.",
        "Looking forward to the weekend with my family.",
        "This group of people is ruining our country."
    ]
    
    print("\nTesting with sample texts...")
    print("-" * 60)
    
    for i, text in enumerate(test_examples, 1):
        print(f"\nExample {i}:")
        predictor.explain_prediction(text)
        print("-" * 60)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Enter text to classify (or 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\\nEnter text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                print("Please enter some text.")
                continue
            
            predictor.explain_prediction(user_input)
            
        except KeyboardInterrupt:
            print("\\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\\nThank you for using the hate speech detection system!")

def create_prediction_demo():
    """Create a demonstration of the prediction system."""
    
    predictor = HateSpeechPredictor()
    
    # Create a comprehensive test dataset
    test_data = [
        {"text": "I love spending time with my friends and family.", "expected": "NOT"},
        {"text": "This politician is completely incompetent.", "expected": "NOT"},
        {"text": "All immigrants are criminals and should be deported.", "expected": "HOF"},
        {"text": "The movie was terrible, what a waste of time.", "expected": "NOT"},
        {"text": "Women shouldn't have equal rights.", "expected": "HOF"},
        {"text": "Great job on the project, keep up the good work!", "expected": "NOT"},
        {"text": "I hate this stupid government policy.", "expected": "NOT"},
        {"text": "Those people are animals and deserve punishment.", "expected": "HOF"},
        {"text": "Looking forward to the weekend.", "expected": "NOT"},
        {"text": "This group is ruining our country.", "expected": "HOF"}
    ]
    
    # Run predictions
    results = []
    for item in test_data:
        prediction = predictor.predict_single(item["text"])
        results.append({
            'text': item["text"],
            'expected': item["expected"],
            'predicted_task1': prediction['task_1']['prediction'],
            'predicted_task2': prediction['task_2']['prediction'],
            'predicted_task3': prediction['task_3']['prediction'],
            'task1_probs': prediction['task_1']['probabilities'],
            'task2_probs': prediction['task_2']['probabilities'],
            'task3_probs': prediction['task_3']['probabilities']
        })
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results
    df_results.to_csv('../data/prediction_demo_results.csv', index=False)
    print("Prediction demo results saved to '../data/prediction_demo_results.csv'")
    
    # Print summary
    print("\\nPrediction Demo Summary:")
    print("-" * 40)
    
    # Task 1 accuracy
    task1_correct = sum(1 for r in results if r['predicted_task1'] == r['expected'])
    print(f"Task 1 (Hate/Offensive) accuracy: {task1_correct}/{len(results)} ({task1_correct/len(results)*100:.1f}%)")
    
    # Show some examples
    print("\\nSample predictions:")
    for i, result in enumerate(results[:5]):
        print(f"\\n{i+1}. Text: {result['text'][:50]}...")
        print(f"   Expected: {result['expected']}")
        print(f"   Task 1: {result['predicted_task1']}")
        print(f"   Task 2: {result['predicted_task2']}")
        print(f"   Task 3: {result['predicted_task3']}")
    
    return df_results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        create_prediction_demo()
    else:
        main()
