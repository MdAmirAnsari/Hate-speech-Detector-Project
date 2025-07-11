# Hate Speech Detection Project

A comprehensive machine learning system for detecting hate speech and offensive content in English text.

## 📁 Project Structure

```
hate-speech-detection-project/
├── README.md                          # This file
├── PROJECT_SUMMARY.md                 # Detailed project summary
├── streamlit_app.py                   # 🌐 Web interface (NEW!)
├── run_streamlit.ps1                  # 🚀 Web interface launcher
├── run_streamlit.bat                  # 🚀 Alternative launcher
├── sample_texts.txt                   # 📝 Sample texts for testing
├── requirements.txt                   # 📦 Dependencies
├── setup.py                          # 🔧 Setup script
├── VSCODE_SETUP.md                   # 💻 VS Code troubleshooting
├── .vscode/                          # VS Code configuration
│   ├── settings.json                 # Python interpreter settings
│   └── launch.json                   # Debug configurations
├── data/                              # Data files
│   ├── english_dataset.tsv            # Original dataset
│   ├── processed_dataset.csv          # Preprocessed dataset
│   ├── model_results_summary.csv      # Model performance metrics
│   └── prediction_demo_results.csv    # Demo predictions
├── models/                            # Trained models
│   ├── best_model_task_1.pkl          # Binary classification model
│   ├── best_model_task_2.pkl          # Multi-class classification model
│   └── best_model_task_3.pkl          # Target identification model
├── scripts/                           # Python scripts
│   ├── data_preprocessing.py          # Data preprocessing pipeline
│   ├── model_training.py              # Model training & evaluation
│   ├── predict_hate_speech.py         # Terminal prediction system
│   ├── generate_visualizations.py     # Create charts & graphs
│   ├── simple_usage_example.py        # Simple usage example
│   └── environment_test.py            # Environment verification
├── visualizations/                    # Charts and graphs
│   ├── confusion_matrix_task_1.png    # Confusion matrix for Task 1
│   ├── confusion_matrix_task_2.png    # Confusion matrix for Task 2
│   ├── confusion_matrix_task_3.png    # Confusion matrix for Task 3
│   ├── model_comparison_f1.png        # F1 score comparison
│   ├── model_comparison_accuracy.png  # Accuracy comparison
│   ├── model_comparison_precision.png # Precision comparison
│   └── model_comparison_recall.png    # Recall comparison
└── results/                           # Analysis results
    └── model_evaluation_summary.txt   # Detailed performance analysis
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Virtual environment (recommended)

### Installation
1. **Clone/Download** this project folder
2. **Create virtual environment**:
   ```bash
   python -m venv hate-speech-env
   ```
3. **Activate virtual environment**:
   - Windows: `.\hate-speech-env\Scripts\Activate.ps1`
   - Linux/Mac: `source hate-speech-env/bin/activate`
4. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn nltk
   ```
5. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('punkt_tab')
   ```

## 🎯 Usage

### 1. 🌐 Web Interface (Recommended)
```bash
streamlit run streamlit_app.py
```
- **Modern web interface** with drag-and-drop file upload
- **Real-time predictions** with interactive visualizations
- **Batch processing** for multiple texts
- **Export results** as CSV files
- **User-friendly** with confidence scores and charts

**Quick Start:**
```bash
# Option 1: Direct command
streamlit run streamlit_app.py --server.port 8501

# Option 2: Use the startup script
.\run_streamlit.ps1
```

### 2. 📱 Terminal Interface
```bash
cd scripts
python predict_hate_speech.py
```
- Type any text to get instant predictions
- Shows results for all three classification tasks
- Includes confidence scores

### 3. 📊 Batch Processing
```bash
cd scripts
python predict_hate_speech.py demo
```
- Tests multiple predefined examples
- Saves results to CSV file

### 4. 🔧 Simple Integration
```bash
cd scripts
python simple_usage_example.py
```
- Basic example for using models in your code
- Easy to customize and integrate

## 📊 Classification Tasks

### Task 1: Binary Classification (Hate/Offensive Detection)
- **NOT**: Normal, non-offensive text
- **HOF**: Hate speech or offensive content
- **Best Model**: Naive Bayes (F1: 0.6437, Accuracy: 0.6661)

### Task 2: Multi-class Classification
- **NONE**: Not hate speech
- **HATE**: Hate speech
- **PRFN**: Profanity
- **OFFN**: Offensive content
- **Best Model**: Random Forest (F1: 0.5582, Accuracy: 0.6413)

### Task 3: Target Identification
- **NONE**: No specific target
- **TIN**: Targeted at individual/group
- **UNT**: Untargeted offensive content
- **Best Model**: Logistic Regression (F1: 0.6302, Accuracy: 0.6687)

## 🔧 Custom Usage

### Load and Use Models
```python
import pickle
from scripts.simple_usage_example import predict_hate_speech

# Load model
with open('models/best_model_task_1.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Make prediction
text = "Your text here"
result = predict_hate_speech(text)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Process Multiple Texts
```python
texts = ["Text 1", "Text 2", "Text 3"]
results = []

for text in texts:
    result = predict_hate_speech(text)
    results.append({
        'text': text,
        'prediction': result['prediction'],
        'confidence': result['confidence']
    })
```

## 📈 Model Performance

| Task | Best Model | F1 Score | Accuracy | Precision | Recall |
|------|------------|----------|----------|-----------|--------|
| Task 1 | Naive Bayes | 0.6437 | 0.6661 | 0.6552 | 0.6661 |
| Task 2 | Random Forest | 0.5582 | 0.6413 | 0.5897 | 0.6413 |
| Task 3 | Logistic Regression | 0.6302 | 0.6687 | 0.6327 | 0.6687 |

## 🔄 Retraining Models

### 1. Preprocess Data
```bash
cd scripts
python data_preprocessing.py
```

### 2. Train Models
```bash
python model_training.py
```

### 3. Generate Visualizations
```bash
python generate_visualizations.py
```

## 📊 Dataset Information

- **Source**: english_dataset.tsv
- **Size**: 5,852 samples
- **Features**: Text content with labels for three tasks
- **Preprocessing**: Text cleaning, tokenization, TF-IDF vectorization

### Label Distribution
- **Task 1**: 3,591 NOT, 2,261 HOF
- **Task 2**: 3,591 NONE, 1,143 HATE, 667 PRFN, 451 OFFN
- **Task 3**: 3,591 NONE, 2,041 TIN, 220 UNT

## 🛠️ Technical Details

### Models Tested
- Logistic Regression
- Naive Bayes (MultinomialNB)
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting

### Feature Engineering
- **Text Cleaning**: URLs, mentions, hashtags, punctuation removal
- **Tokenization**: NLTK word tokenization
- **Stopword Removal**: English stopwords filtered
- **TF-IDF Vectorization**: Unigrams, bigrams, and trigrams

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC (for binary classification)

## 🔮 Future Improvements

1. **Advanced NLP**: Word embeddings (Word2Vec, GloVe, BERT)
2. **Deep Learning**: LSTM, GRU, Transformer models
3. **Ensemble Methods**: Combine multiple models
4. **Hyperparameter Tuning**: Grid search optimization
5. **Class Imbalance**: SMOTE, weighted sampling
6. **Real-time API**: Flask/FastAPI deployment
7. **Multi-language Support**: Extend to other languages

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements.

## 📞 Support

For questions or issues:
1. Check the `PROJECT_SUMMARY.md` for detailed information
2. Review the `results/model_evaluation_summary.txt` for performance details
3. Look at the visualization files for model comparisons

## 🎉 Acknowledgments

- Dataset providers for the English hate speech dataset
- scikit-learn for machine learning tools
- NLTK for natural language processing utilities
- matplotlib/seaborn for visualizations

---

**Project Status**: ✅ Complete and Ready for Use

## 🚀 Quick Deploy to Streamlit Cloud

[![Deploy to Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

### One-Click Deployment
1. **Fork this repository** to your GitHub account
2. **Visit [share.streamlit.io](https://share.streamlit.io)**
3. **Click "New app"** and select your forked repository
4. **Set main file path** to `streamlit_app.py`
5. **Click "Deploy"** and wait for the magic! ✨

### Live Demo
Once deployed, your app will be available at:
`https://your-username-hate-speech-detection-project-streamlit-app.streamlit.app`

**Note**: Initial deployment may take a few minutes as models (~33MB) are loaded.

## 📖 Deployment Guide
For detailed deployment instructions and troubleshooting, see [DEPLOYMENT.md](DEPLOYMENT.md).

Last Updated: July 2025
