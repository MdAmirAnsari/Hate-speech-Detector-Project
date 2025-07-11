#!/usr/bin/env python3
"""
Model Training and Evaluation Pipeline for Hate Speech Detection
This script trains and evaluates multiple machine learning models for hate speech detection tasks.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.multiclass import OneVsRestClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class HateSpeechModelTrainer:
    """
    A comprehensive model training and evaluation class for hate speech detection.
    """
    
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(random_state=42, probability=True),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
        }
        
        self.results = {}
        self.best_models = {}
        
    def train_and_evaluate_model(self, model_name, model, X_train, X_test, y_train, y_test, task_name):
        """
        Train and evaluate a single model.
        """
        print(f"  Training {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Store results
        results = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Calculate AUC for binary classification
        if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
            results['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        return results
    
    def train_all_models(self, X_train, X_test, y_train, y_test, task_name, feature_type):
        """
        Train and evaluate all models for a given task and feature type.
        """
        print(f"\\nTraining models for {task_name} with {feature_type} features...")
        print("-" * 60)
        
        task_results = {}
        
        for model_name, model in self.models.items():
            try:
                results = self.train_and_evaluate_model(
                    model_name, model, X_train, X_test, y_train, y_test, task_name
                )
                task_results[model_name] = results
                
                print(f"    {model_name}: Accuracy={results['accuracy']:.4f}, "
                      f"F1={results['f1']:.4f}, Precision={results['precision']:.4f}, "
                      f"Recall={results['recall']:.4f}")
                
            except Exception as e:
                print(f"    {model_name}: Error - {str(e)}")
                continue
        
        return task_results
    
    def find_best_model(self, task_results):
        """
        Find the best performing model based on F1 score.
        """
        best_model_name = None
        best_f1 = 0
        
        for model_name, results in task_results.items():
            if results['f1'] > best_f1:
                best_f1 = results['f1']
                best_model_name = model_name
        
        return best_model_name, task_results[best_model_name]
    
    def create_results_summary(self, all_results):
        """
        Create a summary of all model results.
        """
        summary_data = []
        
        for task_feature, task_results in all_results.items():
            for model_name, results in task_results.items():
                summary_data.append({
                    'Task_Feature': task_feature,
                    'Model': model_name,
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1': results['f1'],
                    'AUC': results.get('auc', 'N/A')
                })
        
        return pd.DataFrame(summary_data)
    
    def plot_confusion_matrix(self, cm, labels, title, save_path=None):
        """
        Plot confusion matrix.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(self, summary_df, metric='F1', save_path=None):
        """
        Plot model comparison across tasks.
        """
        plt.figure(figsize=(12, 8))
        
        # Create pivot table for plotting
        pivot_data = summary_df.pivot(index='Model', columns='Task_Feature', values=metric)
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd', 
                   cbar_kws={'label': metric})
        plt.title(f'Model Performance Comparison ({metric})')
        plt.xlabel('Task and Feature Type')
        plt.ylabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def hyperparameter_tuning(self, model_name, X_train, y_train, task_name):
        """
        Perform hyperparameter tuning for the best model.
        """
        print(f"\\nPerforming hyperparameter tuning for {model_name} on {task_name}...")
        
        param_grids = {
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'saga']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        }
        
        if model_name not in param_grids:
            print(f"  No hyperparameter grid defined for {model_name}")
            return None
        
        # Get the base model
        base_model = self.models[model_name]
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, 
            param_grids[model_name],
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_

def load_preprocessed_data():
    """
    Load the preprocessed data and recreate the datasets.
    """
    print("Loading preprocessed data...")
    df = pd.read_csv('../data/processed_dataset.csv')
    
    # Recreate features
    features = {}
    tfidf_configs = {
        'basic': {'max_features': 5000, 'ngram_range': (1, 1)},
        'bigrams': {'max_features': 10000, 'ngram_range': (1, 2)},
        'trigrams': {'max_features': 15000, 'ngram_range': (1, 3)}
    }
    
    for config_name, config in tfidf_configs.items():
        vectorizer = TfidfVectorizer(
            max_features=config['max_features'],
            ngram_range=config['ngram_range'],
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        tfidf_matrix = vectorizer.fit_transform(df['text_cleaned'].fillna(''))
        features[config_name] = {
            'vectorizer': vectorizer,
            'matrix': tfidf_matrix
        }
    
    # Recreate datasets
    datasets = {}
    tasks = ['task_1', 'task_2', 'task_3']
    
    for task in tasks:
        y = df[task]
        
        for feature_name, feature_data in features.items():
            X = feature_data['matrix']
            
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
    
    return df, datasets

def main():
    """
    Main training and evaluation pipeline.
    """
    print("HATE SPEECH DETECTION - MODEL TRAINING AND EVALUATION")
    print("=" * 70)
    
    # Load preprocessed data
    df, datasets = load_preprocessed_data()
    
    # Initialize trainer
    trainer = HateSpeechModelTrainer()
    
    # Train and evaluate all models
    all_results = {}
    
    # Process each task and feature combination
    for dataset_name, dataset in datasets.items():
        task_results = trainer.train_all_models(
            dataset['X_train'], dataset['X_test'],
            dataset['y_train'], dataset['y_test'],
            dataset_name.split('_')[0], dataset_name.split('_')[1]
        )
        all_results[dataset_name] = task_results
    
    # Create results summary
    print("\\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    summary_df = trainer.create_results_summary(all_results)
    print(summary_df.to_string(index=False))
    
    # Save results summary
    summary_df.to_csv('../data/model_results_summary.csv', index=False)
    print("\\nResults saved to '../data/model_results_summary.csv'")
    
    # Find and display best models for each task
    print("\\n" + "=" * 70)
    print("BEST MODELS BY TASK")
    print("=" * 70)
    
    best_models = {}
    tasks = ['task_1', 'task_2', 'task_3']
    
    for task in tasks:
        task_results = {}
        
        # Collect results for this task across all feature types
        for dataset_name, results in all_results.items():
            if dataset_name.startswith(task):
                for model_name, model_results in results.items():
                    key = f"{model_name}_{dataset_name.split('_')[1]}"
                    task_results[key] = model_results
        
        # Find best model
        best_model_key, best_results = trainer.find_best_model(task_results)
        best_models[task] = {
            'name': best_model_key,
            'results': best_results
        }
        
        print(f"\\n{task.upper()}:")
        print(f"  Best Model: {best_model_key}")
        print(f"  F1 Score: {best_results['f1']:.4f}")
        print(f"  Accuracy: {best_results['accuracy']:.4f}")
        print(f"  Precision: {best_results['precision']:.4f}")
        print(f"  Recall: {best_results['recall']:.4f}")
        
        # Show classification report
        print(f"\\n  Classification Report:")
        print(best_results['classification_report'])
    
    # Create visualizations
    print("\\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    # Plot model comparison
    trainer.plot_model_comparison(summary_df, 'F1', '../visualizations/model_comparison_f1.png')
    trainer.plot_model_comparison(summary_df, 'Accuracy', '../visualizations/model_comparison_accuracy.png')
    
    # Plot confusion matrices for best models
    for task, best_model_info in best_models.items():
        cm = best_model_info['results']['confusion_matrix']
        labels = np.unique(best_model_info['results']['y_test'])
        
        trainer.plot_confusion_matrix(
            cm, labels, 
            f'Confusion Matrix - {task.upper()} - {best_model_info["name"]}',
            f'../visualizations/confusion_matrix_{task}.png'
        )
    
    # Save best models
    print("\\n" + "=" * 70)
    print("SAVING BEST MODELS")
    print("=" * 70)
    
    for task, best_model_info in best_models.items():
        model_filename = f'../models/best_model_{task}.pkl'
        
        # Save model and vectorizer
        model_data = {
            'model': best_model_info['results']['model'],
            'model_name': best_model_info['name'],
            'performance': {
                'accuracy': best_model_info['results']['accuracy'],
                'f1': best_model_info['results']['f1'],
                'precision': best_model_info['results']['precision'],
                'recall': best_model_info['results']['recall']
            }
        }
        
        with open(model_filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Saved {task} model: {model_filename}")
    
    print("\\n" + "=" * 70)
    print("🎉 MODEL TRAINING AND EVALUATION COMPLETE!")
    print("=" * 70)
    
    print("\\nFiles created:")
    print("- model_results_summary.csv: Complete results summary")
    print("- model_comparison_f1.png: F1 score comparison heatmap")
    print("- model_comparison_accuracy.png: Accuracy comparison heatmap")
    print("- confusion_matrix_*.png: Confusion matrices for best models")
    print("- best_model_*.pkl: Saved best models for each task")
    
    print("\\nNext steps:")
    print("1. Fine-tune hyperparameters for best models")
    print("2. Implement ensemble methods")
    print("3. Deploy models for real-time prediction")
    print("4. Create a web interface for testing")
    
    return all_results, best_models, summary_df

if __name__ == "__main__":
    all_results, best_models, summary_df = main()
