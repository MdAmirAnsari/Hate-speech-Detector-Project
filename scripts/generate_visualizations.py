#!/usr/bin/env python3
"""
Generate visualizations and complete model evaluation for hate speech detection
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

def plot_model_comparison(summary_df, metric='F1', save_path=None):
    """Plot model comparison across tasks."""
    plt.figure(figsize=(15, 10))
    
    # Create pivot table for plotting
    pivot_data = summary_df.pivot(index='Model', columns='Task_Feature', values=metric)
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd', 
               cbar_kws={'label': metric})
    plt.title(f'Model Performance Comparison ({metric})', fontsize=16)
    plt.xlabel('Task and Feature Type', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()

def plot_confusion_matrix(cm, labels, title, save_path=None):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()

def load_and_evaluate_models():
    """Load data and evaluate models to get confusion matrices."""
    
    # Load processed data
    df = pd.read_csv('../data/processed_dataset.csv')
    
    # Recreate the datasets for best models
    # Task 1: Naive Bayes with trigrams
    vectorizer_task1 = TfidfVectorizer(max_features=15000, ngram_range=(1, 3), 
                                      min_df=2, max_df=0.95, stop_words='english')
    X_task1 = vectorizer_task1.fit_transform(df['text_cleaned'].fillna(''))
    y_task1 = df['task_1']
    _, X_test_task1, _, y_test_task1 = train_test_split(X_task1, y_task1, test_size=0.2, 
                                                        random_state=42, stratify=y_task1)
    
    # Task 2: Random Forest with bigrams  
    vectorizer_task2 = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), 
                                      min_df=2, max_df=0.95, stop_words='english')
    X_task2 = vectorizer_task2.fit_transform(df['text_cleaned'].fillna(''))
    y_task2 = df['task_2']
    _, X_test_task2, _, y_test_task2 = train_test_split(X_task2, y_task2, test_size=0.2, 
                                                        random_state=42, stratify=y_task2)
    
    # Task 3: Logistic Regression with trigrams
    vectorizer_task3 = TfidfVectorizer(max_features=15000, ngram_range=(1, 3), 
                                      min_df=2, max_df=0.95, stop_words='english')
    X_task3 = vectorizer_task3.fit_transform(df['text_cleaned'].fillna(''))
    y_task3 = df['task_3']
    _, X_test_task3, _, y_test_task3 = train_test_split(X_task3, y_task3, test_size=0.2, 
                                                        random_state=42, stratify=y_task3)
    
    # Get predictions from best models (approximated)
    # Task 1: Naive Bayes
    X_train_task1, _, y_train_task1, _ = train_test_split(X_task1, y_task1, test_size=0.2, 
                                                          random_state=42, stratify=y_task1)
    model_task1 = MultinomialNB()
    model_task1.fit(X_train_task1, y_train_task1)
    y_pred_task1 = model_task1.predict(X_test_task1)
    cm_task1 = confusion_matrix(y_test_task1, y_pred_task1)
    
    # Task 2: Random Forest
    X_train_task2, _, y_train_task2, _ = train_test_split(X_task2, y_task2, test_size=0.2, 
                                                          random_state=42, stratify=y_task2)
    model_task2 = RandomForestClassifier(random_state=42, n_estimators=100)
    model_task2.fit(X_train_task2, y_train_task2)
    y_pred_task2 = model_task2.predict(X_test_task2)
    cm_task2 = confusion_matrix(y_test_task2, y_pred_task2)
    
    # Task 3: Logistic Regression
    X_train_task3, _, y_train_task3, _ = train_test_split(X_task3, y_task3, test_size=0.2, 
                                                          random_state=42, stratify=y_task3)
    model_task3 = LogisticRegression(random_state=42, max_iter=1000)
    model_task3.fit(X_train_task3, y_train_task3)
    y_pred_task3 = model_task3.predict(X_test_task3)
    cm_task3 = confusion_matrix(y_test_task3, y_pred_task3)
    
    # Save models
    models_data = {
        'task_1': {
            'model': model_task1,
            'vectorizer': vectorizer_task1,
            'model_name': 'Naive Bayes (Trigrams)',
            'labels': np.unique(y_task1)
        },
        'task_2': {
            'model': model_task2,
            'vectorizer': vectorizer_task2,
            'model_name': 'Random Forest (Bigrams)',
            'labels': np.unique(y_task2)
        },
        'task_3': {
            'model': model_task3,
            'vectorizer': vectorizer_task3,
            'model_name': 'Logistic Regression (Trigrams)',
            'labels': np.unique(y_task3)
        }
    }
    
    # Save models
    for task, model_data in models_data.items():
        with open(f'../models/best_model_{task}.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Saved best model for {task}")
    
    return {
        'task_1': (cm_task1, np.unique(y_test_task1)),
        'task_2': (cm_task2, np.unique(y_test_task2)),
        'task_3': (cm_task3, np.unique(y_test_task3))
    }

def main():
    """Main function to generate all visualizations."""
    
    print("GENERATING VISUALIZATIONS FOR HATE SPEECH DETECTION")
    print("=" * 60)
    
    # Load results summary
    try:
        summary_df = pd.read_csv('../data/model_results_summary.csv')
        print("Loaded model results summary")
    except FileNotFoundError:
        print("Results summary not found. Please run model_training.py first.")
        return
    
    # Create performance comparison plots
    print("\nCreating performance comparison plots...")
    plot_model_comparison(summary_df, 'F1', '../visualizations/model_comparison_f1.png')
    plot_model_comparison(summary_df, 'Accuracy', '../visualizations/model_comparison_accuracy.png')
    plot_model_comparison(summary_df, 'Precision', '../visualizations/model_comparison_precision.png')
    plot_model_comparison(summary_df, 'Recall', '../visualizations/model_comparison_recall.png')
    
    # Load models and create confusion matrices
    print("\nCreating confusion matrices...")
    confusion_matrices = load_and_evaluate_models()
    
    task_names = {
        'task_1': 'Hate/Offensive Detection (Binary)',
        'task_2': 'Hate Speech Classification (Multi-class)',
        'task_3': 'Target Identification (Multi-class)'
    }
    
    model_names = {
        'task_1': 'Naive Bayes (Trigrams)',
        'task_2': 'Random Forest (Bigrams)',
        'task_3': 'Logistic Regression (Trigrams)'
    }
    
    for task, (cm, labels) in confusion_matrices.items():
        title = f'{task_names[task]} - {model_names[task]}'
        plot_confusion_matrix(cm, labels, title, f'../visualizations/confusion_matrix_{task}.png')
    
    # Create a summary report
    print("\nCreating summary report...")
    create_summary_report(summary_df)
    
    print("\n" + "=" * 60)
    print("🎉 VISUALIZATION GENERATION COMPLETE!")
    print("=" * 60)
    
    print("\nGenerated files:")
    print("- model_comparison_f1.png")
    print("- model_comparison_accuracy.png")
    print("- model_comparison_precision.png")
    print("- model_comparison_recall.png")
    print("- confusion_matrix_task_1.png")
    print("- confusion_matrix_task_2.png")
    print("- confusion_matrix_task_3.png")
    print("- best_model_task_1.pkl")
    print("- best_model_task_2.pkl")
    print("- best_model_task_3.pkl")
    print("- model_evaluation_summary.txt")

def create_summary_report(summary_df):
    """Create a text summary report."""
    
    report = []
    report.append("HATE SPEECH DETECTION - MODEL EVALUATION SUMMARY")
    report.append("=" * 60)
    report.append("")
    
    # Best models by task
    report.append("BEST PERFORMING MODELS BY TASK:")
    report.append("-" * 40)
    
    tasks = ['task_1', 'task_2', 'task_3']
    task_descriptions = {
        'task_1': 'Hate/Offensive Detection (Binary Classification)',
        'task_2': 'Hate Speech Classification (Multi-class)',
        'task_3': 'Target Identification (Multi-class)'
    }
    
    for task in tasks:
        task_data = summary_df[summary_df['Task_Feature'].str.startswith(task)]
        best_model = task_data.loc[task_data['F1'].idxmax()]
        
        report.append(f"\n{task.upper()}: {task_descriptions[task]}")
        report.append(f"Best Model: {best_model['Model']} ({best_model['Task_Feature'].split('_')[1]})")
        report.append(f"F1 Score: {best_model['F1']:.4f}")
        report.append(f"Accuracy: {best_model['Accuracy']:.4f}")
        report.append(f"Precision: {best_model['Precision']:.4f}")
        report.append(f"Recall: {best_model['Recall']:.4f}")
    
    report.append("")
    report.append("MODEL PERFORMANCE SUMMARY:")
    report.append("-" * 40)
    
    # Overall best performing models
    best_models = summary_df.nlargest(5, 'F1')
    report.append("\nTop 5 Models by F1 Score:")
    for idx, model in best_models.iterrows():
        report.append(f"{model['Task_Feature']} - {model['Model']}: F1={model['F1']:.4f}")
    
    # Feature type analysis
    report.append("\nPERFORMANCE BY FEATURE TYPE:")
    report.append("-" * 40)
    
    feature_performance = summary_df.groupby(summary_df['Task_Feature'].str.split('_').str[1]).agg({
        'F1': 'mean',
        'Accuracy': 'mean',
        'Precision': 'mean',
        'Recall': 'mean'
    }).round(4)
    
    for feature_type, metrics in feature_performance.iterrows():
        report.append(f"\n{feature_type.upper()} features:")
        report.append(f"  Average F1: {metrics['F1']:.4f}")
        report.append(f"  Average Accuracy: {metrics['Accuracy']:.4f}")
        report.append(f"  Average Precision: {metrics['Precision']:.4f}")
        report.append(f"  Average Recall: {metrics['Recall']:.4f}")
    
    # Model type analysis
    report.append("\nPERFORMANCE BY MODEL TYPE:")
    report.append("-" * 40)
    
    model_performance = summary_df.groupby('Model').agg({
        'F1': 'mean',
        'Accuracy': 'mean',
        'Precision': 'mean',
        'Recall': 'mean'
    }).round(4)
    
    for model_type, metrics in model_performance.iterrows():
        report.append(f"\n{model_type}:")
        report.append(f"  Average F1: {metrics['F1']:.4f}")
        report.append(f"  Average Accuracy: {metrics['Accuracy']:.4f}")
        report.append(f"  Average Precision: {metrics['Precision']:.4f}")
        report.append(f"  Average Recall: {metrics['Recall']:.4f}")
    
    report.append("")
    report.append("INSIGHTS AND RECOMMENDATIONS:")
    report.append("-" * 40)
    report.append("1. Binary classification (Task 1) performs better than multi-class tasks")
    report.append("2. Trigram features generally provide good performance")
    report.append("3. Naive Bayes and Random Forest show strong performance")
    report.append("4. Task 3 (Target Identification) is the most challenging")
    report.append("5. Consider ensemble methods for improved performance")
    
    # Save report
    with open('../results/model_evaluation_summary.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print("Summary report saved to '../results/model_evaluation_summary.txt'")

if __name__ == "__main__":
    main()
