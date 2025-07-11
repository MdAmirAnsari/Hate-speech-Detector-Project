#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for Hate Speech Detection
This script performs comprehensive text preprocessing including cleaning, tokenization, and feature extraction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class TextPreprocessor:
    """
    A comprehensive text preprocessing class for hate speech detection.
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
    def clean_text(self, text):
        """
        Clean and normalize text data.
        """
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
        """
        Tokenize text and remove stopwords.
        """
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
    
    def stem_tokens(self, tokens):
        """
        Apply stemming to tokens.
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess_text(self, text, stem=False):
        """
        Complete text preprocessing pipeline.
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and filter
        tokens = self.tokenize_and_filter(cleaned_text)
        
        # Apply stemming if requested
        if stem:
            tokens = self.stem_tokens(tokens)
        
        # Return as string
        return ' '.join(tokens)

def load_and_explore_data():
    """
    Load the dataset and perform initial exploration.
    """
    print("Loading dataset...")
    df = pd.read_csv('../data/english_dataset.tsv', sep='\t')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Display basic statistics
    print("\nDataset info:")
    print(df.info())
    
    return df

def analyze_text_characteristics(df):
    """
    Analyze text characteristics and distributions.
    """
    print("\n" + "="*60)
    print("TEXT CHARACTERISTICS ANALYSIS")
    print("="*60)
    
    # Calculate text length statistics
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    print(f"\nText length statistics:")
    print(f"Mean: {df['text_length'].mean():.2f}")
    print(f"Median: {df['text_length'].median():.2f}")
    print(f"Min: {df['text_length'].min()}")
    print(f"Max: {df['text_length'].max()}")
    
    print(f"\nWord count statistics:")
    print(f"Mean: {df['word_count'].mean():.2f}")
    print(f"Median: {df['word_count'].median():.2f}")
    print(f"Min: {df['word_count'].min()}")
    print(f"Max: {df['word_count'].max()}")
    
    # Analyze by class
    print(f"\nText length by class (Task 1):")
    for class_label in df['task_1'].unique():
        class_data = df[df['task_1'] == class_label]
        print(f"{class_label}: Mean={class_data['text_length'].mean():.2f}, "
              f"Median={class_data['text_length'].median():.2f}")
    
    return df

def preprocess_dataset(df):
    """
    Apply preprocessing to the entire dataset.
    """
    print("\n" + "="*60)
    print("PREPROCESSING DATASET")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Apply preprocessing
    print("Applying text preprocessing...")
    df['text_cleaned'] = df['text'].apply(preprocessor.preprocess_text)
    
    # Also create stemmed version
    print("Creating stemmed version...")
    df['text_stemmed'] = df['text'].apply(lambda x: preprocessor.preprocess_text(x, stem=True))
    
    # Show examples
    print("\nPreprocessing examples:")
    print("-" * 40)
    for i in range(3):
        print(f"Original: {df['text'].iloc[i]}")
        print(f"Cleaned:  {df['text_cleaned'].iloc[i]}")
        print(f"Stemmed:  {df['text_stemmed'].iloc[i]}")
        print("-" * 40)
    
    # Calculate new text statistics
    df['cleaned_length'] = df['text_cleaned'].str.len()
    df['cleaned_word_count'] = df['text_cleaned'].str.split().str.len()
    
    print(f"\nCleaned text statistics:")
    print(f"Mean length: {df['cleaned_length'].mean():.2f}")
    print(f"Mean word count: {df['cleaned_word_count'].mean():.2f}")
    
    return df

def analyze_vocabulary(df):
    """
    Analyze vocabulary and word frequencies.
    """
    print("\n" + "="*60)
    print("VOCABULARY ANALYSIS")
    print("="*60)
    
    # Combine all cleaned text
    all_text = ' '.join(df['text_cleaned'].dropna())
    words = all_text.split()
    
    print(f"Total words: {len(words)}")
    print(f"Unique words: {len(set(words))}")
    
    # Most common words
    word_freq = Counter(words)
    print(f"\nTop 20 most common words:")
    for word, freq in word_freq.most_common(20):
        print(f"  {word}: {freq}")
    
    # Analyze by class
    print(f"\nVocabulary by class (Task 1):")
    for class_label in df['task_1'].unique():
        class_text = ' '.join(df[df['task_1'] == class_label]['text_cleaned'].dropna())
        class_words = class_text.split()
        class_word_freq = Counter(class_words)
        print(f"\n{class_label} - Top 10 words:")
        for word, freq in class_word_freq.most_common(10):
            print(f"  {word}: {freq}")
    
    return word_freq

def create_features(df):
    """
    Create TF-IDF features for machine learning.
    """
    print("\n" + "="*60)
    print("FEATURE EXTRACTION")
    print("="*60)
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    
    # TF-IDF with different parameters
    tfidf_configs = {
        'basic': {'max_features': 5000, 'ngram_range': (1, 1)},
        'bigrams': {'max_features': 10000, 'ngram_range': (1, 2)},
        'trigrams': {'max_features': 15000, 'ngram_range': (1, 3)}
    }
    
    features = {}
    
    for config_name, config in tfidf_configs.items():
        print(f"Creating {config_name} TF-IDF features...")
        
        vectorizer = TfidfVectorizer(
            max_features=config['max_features'],
            ngram_range=config['ngram_range'],
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        # Fit on cleaned text
        tfidf_matrix = vectorizer.fit_transform(df['text_cleaned'].fillna(''))
        
        features[config_name] = {
            'vectorizer': vectorizer,
            'matrix': tfidf_matrix,
            'feature_names': vectorizer.get_feature_names_out()
        }
        
        print(f"  Shape: {tfidf_matrix.shape}")
        print(f"  Sparsity: {1 - (tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])):.4f}")
    
    return features

def prepare_datasets(df, features):
    """
    Prepare train/test splits for all tasks.
    """
    print("\n" + "="*60)
    print("PREPARING DATASETS")
    print("="*60)
    
    datasets = {}
    
    # Prepare datasets for each task
    tasks = ['task_1', 'task_2', 'task_3']
    
    for task in tasks:
        print(f"\nPreparing {task} dataset...")
        
        # Get labels
        y = df[task]
        
        # Print label distribution
        print(f"Label distribution: {dict(y.value_counts())}")
        
        # For each feature type
        for feature_name, feature_data in features.items():
            X = feature_data['matrix']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            datasets[f"{task}_{feature_name}"] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'vectorizer': feature_data['vectorizer']
            }
            
            print(f"  {feature_name}: Train={X_train.shape}, Test={X_test.shape}")
    
    return datasets

def main():
    """
    Main preprocessing pipeline.
    """
    print("HATE SPEECH DETECTION - DATA PREPROCESSING")
    print("=" * 60)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Analyze text characteristics
    df = analyze_text_characteristics(df)
    
    # Preprocess the dataset
    df = preprocess_dataset(df)
    
    # Analyze vocabulary
    word_freq = analyze_vocabulary(df)
    
    # Create features
    features = create_features(df)
    
    # Prepare datasets
    datasets = prepare_datasets(df, features)
    
    # Save processed data
    print("\nSaving processed data...")
    df.to_csv('../data/processed_dataset.csv', index=False)
    print("Processed dataset saved as '../data/processed_dataset.csv'")
    
    print("\n" + "="*60)
    print("🎉 PREPROCESSING COMPLETE!")
    print("="*60)
    
    print(f"\nProcessed dataset contains:")
    print(f"- Original text")
    print(f"- Cleaned text")
    print(f"- Stemmed text")
    print(f"- Text statistics")
    print(f"- TF-IDF features (basic, bigrams, trigrams)")
    print(f"- Train/test splits for all tasks")
    
    print(f"\nReady for model training!")
    
    return df, features, datasets

if __name__ == "__main__":
    df, features, datasets = main()
