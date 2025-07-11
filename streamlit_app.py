#!/usr/bin/env python3
"""
Streamlit Web Interface for Hate Speech Detection
Upload text files and get instant hate speech predictions with confidence scores.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Hate Speech Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}
.stAlert {
    padding: 1rem;
    margin: 1rem 0;
}
.prediction-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.metric-container {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 5px;
    border: 1px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)

class HateSpeechDetector:
    """Streamlit-optimized hate speech detection class."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.models = self.load_models()
    
    def load_models(self):
        """Load models for the detector."""
        models = {}
        tasks = ['task_1', 'task_2', 'task_3']
        
        for task in tasks:
            try:
                with open(f'models/best_model_{task}.pkl', 'rb') as f:
                    models[task] = pickle.load(f)
                print(f"✅ Loaded model for {task}")
            except FileNotFoundError as e:
                print(f"❌ Model file for {task} not found: {e}")
                return {}
            except Exception as e:
                print(f"❌ Error loading model for {task}: {e}")
                return {}
        
        print(f"🎉 Successfully loaded {len(models)} models")
        return models
    
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
        if not self.models:
            return None
        
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
                'model_name': model_data['model_name'],
                'confidence': max(probabilities) if probabilities is not None else 0.0
            }
        
        return results
    
    def predict_batch(self, texts):
        """Predict hate speech for multiple text samples."""
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, text in enumerate(texts):
            progress = (i + 1) / len(texts)
            progress_bar.progress(progress)
            status_text.text(f'Processing text {i+1}/{len(texts)}...')
            
            result = self.predict_single(text)
            if result:
                result['original_text'] = text
                results.append(result)
        
        progress_bar.empty()
        status_text.empty()
        
        return results

def create_prediction_visualization(results):
    """Create visualizations for prediction results."""
    if not results:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Task 1: Hate/Offensive Detection', 'Task 2: Hate Speech Classification', 
                       'Task 3: Target Identification', 'Confidence Scores'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Task 1 - Binary Classification
    if 'task_1' in results and results['task_1']['probabilities']:
        probs = results['task_1']['probabilities']
        fig.add_trace(
            go.Bar(x=list(probs.keys()), y=list(probs.values()), 
                   name="Task 1", marker_color='lightblue'),
            row=1, col=1
        )
    
    # Task 2 - Multi-class Classification
    if 'task_2' in results and results['task_2']['probabilities']:
        probs = results['task_2']['probabilities']
        fig.add_trace(
            go.Bar(x=list(probs.keys()), y=list(probs.values()), 
                   name="Task 2", marker_color='lightgreen'),
            row=1, col=2
        )
    
    # Task 3 - Target Identification
    if 'task_3' in results and results['task_3']['probabilities']:
        probs = results['task_3']['probabilities']
        fig.add_trace(
            go.Bar(x=list(probs.keys()), y=list(probs.values()), 
                   name="Task 3", marker_color='lightcoral'),
            row=2, col=1
        )
    
    # Confidence scores
    confidences = [results[task]['confidence'] for task in ['task_1', 'task_2', 'task_3'] if task in results]
    task_names = ['Task 1', 'Task 2', 'Task 3']
    
    fig.add_trace(
        go.Bar(x=task_names, y=confidences, 
               name="Confidence", marker_color='orange'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Prediction Analysis")
    return fig

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🛡️ Hate Speech Detection System")
    st.markdown("### Upload text files or enter text directly to detect hate speech content")
    
    # Initialize detector with caching
    @st.cache_resource
    def load_detector():
        return HateSpeechDetector()
    
    detector = load_detector()
    
    if not detector.models:
        st.error("❌ Models not loaded. Please ensure all model files are present in the 'models' directory.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("📊 Detection Options")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Text Input", "File Upload", "Batch Processing"]
    )
    
    # Model information
    st.sidebar.header("🤖 Model Information")
    st.sidebar.info("""
    **Task 1**: Binary Classification (Hate/Offensive vs Normal)
    **Task 2**: Multi-class Classification (Hate, Profanity, Offensive, None)
    **Task 3**: Target Identification (Targeted, Untargeted, None)
    """)
    
    # Main content area
    if input_method == "Text Input":
        st.header("📝 Single Text Analysis")
        
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste your text here...",
            height=150
        )
        
        if st.button("🔍 Analyze Text", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing text..."):
                    results = detector.predict_single(text_input)
                
                if results:
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    # Create columns for results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🎯 Predictions")
                        
                        # Task 1 - Binary
                        task1_pred = results['task_1']['prediction']
                        task1_conf = results['task_1']['confidence']
                        
                        if task1_pred == 'HOF':
                            st.error(f"⚠️ **Hate/Offensive Content Detected** (Confidence: {task1_conf:.2%})")
                        else:
                            st.success(f"✅ **Normal Content** (Confidence: {task1_conf:.2%})")
                        
                        # Task 2 - Multi-class
                        task2_pred = results['task_2']['prediction']
                        task2_conf = results['task_2']['confidence']
                        
                        labels_map = {
                            'HATE': '🔴 Hate Speech',
                            'PRFN': '🟡 Profanity',
                            'OFFN': '🟠 Offensive',
                            'NONE': '🟢 None'
                        }
                        
                        st.info(f"**Classification**: {labels_map.get(task2_pred, task2_pred)} (Confidence: {task2_conf:.2%})")
                        
                        # Task 3 - Target
                        task3_pred = results['task_3']['prediction']
                        task3_conf = results['task_3']['confidence']
                        
                        target_map = {
                            'TIN': '🎯 Targeted',
                            'UNT': '🌐 Untargeted',
                            'NONE': '⚪ None'
                        }
                        
                        st.info(f"**Target**: {target_map.get(task3_pred, task3_pred)} (Confidence: {task3_conf:.2%})")
                    
                    with col2:
                        st.subheader("📊 Confidence Breakdown")
                        
                        # Display probability distributions
                        for task, task_results in results.items():
                            if task_results['probabilities']:
                                st.write(f"**{task.upper()}** probabilities:")
                                for label, prob in task_results['probabilities'].items():
                                    st.write(f"  {label}: {prob:.2%}")
                                st.write("")
                    
                    # Visualization
                    st.subheader("📈 Prediction Visualization")
                    fig = create_prediction_visualization(results)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show processed text
                    with st.expander("🔍 View Processed Text"):
                        processed = detector.preprocess_text(text_input)
                        st.code(processed, language='text')
                        
                else:
                    st.error("❌ Failed to analyze text. Please check the models.")
            else:
                st.warning("⚠️ Please enter some text to analyze.")
    
    elif input_method == "File Upload":
        st.header("📁 File Upload Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose a text file",
            type=['txt', 'csv'],
            help="Upload a .txt file or .csv file with text content"
        )
        
        if uploaded_file is not None:
            # Read file content
            try:
                if uploaded_file.name.endswith('.txt'):
                    # Read as text file
                    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                    text_content = stringio.read()
                    
                    # Split into sentences or paragraphs
                    texts = [line.strip() for line in text_content.split('\n') if line.strip()]
                    
                elif uploaded_file.name.endswith('.csv'):
                    # Read as CSV
                    df = pd.read_csv(uploaded_file)
                    
                    # Let user select text column
                    text_column = st.selectbox(
                        "Select the text column:",
                        df.columns.tolist()
                    )
                    
                    if text_column:
                        texts = df[text_column].dropna().tolist()
                    else:
                        texts = []
                
                if texts:
                    st.success(f"📄 File loaded successfully! Found {len(texts)} text samples.")
                    
                    # Show preview
                    with st.expander("📖 Preview File Content"):
                        preview_count = min(5, len(texts))
                        for i, text in enumerate(texts[:preview_count]):
                            st.write(f"**{i+1}.** {text[:200]}{'...' if len(text) > 200 else ''}")
                        
                        if len(texts) > preview_count:
                            st.write(f"... and {len(texts) - preview_count} more texts")
                    
                    # Analyze button
                    if st.button("🔍 Analyze All Texts", type="primary"):
                        results = detector.predict_batch(texts)
                        
                        if results:
                            st.success("✅ Analysis Complete!")
                            
                            # Create summary statistics
                            st.subheader("📊 Analysis Summary")
                            
                            # Count predictions
                            task1_counts = {}
                            task2_counts = {}
                            task3_counts = {}
                            
                            for result in results:
                                # Task 1
                                pred1 = result['task_1']['prediction']
                                task1_counts[pred1] = task1_counts.get(pred1, 0) + 1
                                
                                # Task 2
                                pred2 = result['task_2']['prediction']
                                task2_counts[pred2] = task2_counts.get(pred2, 0) + 1
                                
                                # Task 3
                                pred3 = result['task_3']['prediction']
                                task3_counts[pred3] = task3_counts.get(pred3, 0) + 1
                            
                            # Display summary
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Texts", len(results))
                                st.write("**Task 1 (Hate/Offensive):**")
                                for label, count in task1_counts.items():
                                    st.write(f"  {label}: {count} ({count/len(results)*100:.1f}%)")
                            
                            with col2:
                                st.write("**Task 2 (Classification):**")
                                for label, count in task2_counts.items():
                                    st.write(f"  {label}: {count} ({count/len(results)*100:.1f}%)")
                            
                            with col3:
                                st.write("**Task 3 (Target):**")
                                for label, count in task3_counts.items():
                                    st.write(f"  {label}: {count} ({count/len(results)*100:.1f}%)")
                            
                            # Create summary charts
                            st.subheader("📈 Distribution Charts")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                fig1 = px.pie(
                                    values=list(task1_counts.values()),
                                    names=list(task1_counts.keys()),
                                    title="Task 1: Hate/Offensive Detection"
                                )
                                st.plotly_chart(fig1, use_container_width=True)
                            
                            with col2:
                                fig2 = px.pie(
                                    values=list(task2_counts.values()),
                                    names=list(task2_counts.keys()),
                                    title="Task 2: Classification"
                                )
                                st.plotly_chart(fig2, use_container_width=True)
                            
                            with col3:
                                fig3 = px.pie(
                                    values=list(task3_counts.values()),
                                    names=list(task3_counts.keys()),
                                    title="Task 3: Target Identification"
                                )
                                st.plotly_chart(fig3, use_container_width=True)
                            
                            # Detailed results
                            with st.expander("📋 Detailed Results"):
                                # Create results DataFrame
                                results_data = []
                                for i, result in enumerate(results):
                                    results_data.append({
                                        'Text': result['original_text'][:100] + '...' if len(result['original_text']) > 100 else result['original_text'],
                                        'Task 1': result['task_1']['prediction'],
                                        'Task 1 Confidence': f"{result['task_1']['confidence']:.2%}",
                                        'Task 2': result['task_2']['prediction'],
                                        'Task 2 Confidence': f"{result['task_2']['confidence']:.2%}",
                                        'Task 3': result['task_3']['prediction'],
                                        'Task 3 Confidence': f"{result['task_3']['confidence']:.2%}",
                                    })
                                
                                results_df = pd.DataFrame(results_data)
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Download results
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="💾 Download Results as CSV",
                                    data=csv,
                                    file_name="hate_speech_analysis_results.csv",
                                    mime="text/csv"
                                )
                        
                        else:
                            st.error("❌ Failed to analyze texts. Please check the models.")
                else:
                    st.warning("⚠️ No text content found in the file.")
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    elif input_method == "Batch Processing":
        st.header("📊 Batch Text Processing")
        
        st.info("💡 Enter multiple texts separated by line breaks for batch analysis.")
        
        batch_text = st.text_area(
            "Enter multiple texts (one per line):",
            placeholder="Text 1\nText 2\nText 3\n...",
            height=200
        )
        
        if st.button("🔍 Analyze Batch", type="primary"):
            if batch_text.strip():
                texts = [line.strip() for line in batch_text.split('\n') if line.strip()]
                
                if texts:
                    st.success(f"📄 Found {len(texts)} texts to analyze.")
                    
                    results = detector.predict_batch(texts)
                    
                    if results:
                        st.success("✅ Batch Analysis Complete!")
                        
                        # Display results similar to file upload
                        # ... (similar code as file upload section)
                        
                        st.subheader("📋 Results Summary")
                        
                        # Create results DataFrame
                        results_data = []
                        for i, result in enumerate(results):
                            results_data.append({
                                'Text': result['original_text'][:100] + '...' if len(result['original_text']) > 100 else result['original_text'],
                                'Task 1': result['task_1']['prediction'],
                                'Task 1 Confidence': f"{result['task_1']['confidence']:.2%}",
                                'Task 2': result['task_2']['prediction'],
                                'Task 2 Confidence': f"{result['task_2']['confidence']:.2%}",
                                'Task 3': result['task_3']['prediction'],
                                'Task 3 Confidence': f"{result['task_3']['confidence']:.2%}",
                            })
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="💾 Download Results as CSV",
                            data=csv,
                            file_name="batch_hate_speech_analysis.csv",
                            mime="text/csv"
                        )
                    
                    else:
                        st.error("❌ Failed to analyze texts. Please check the models.")
                else:
                    st.warning("⚠️ No valid texts found. Please enter texts separated by line breaks.")
            else:
                st.warning("⚠️ Please enter some texts to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("### 📚 About This System")
    st.markdown("""
    This hate speech detection system uses machine learning models trained on labeled data to identify:
    - **Hate Speech**: Content that attacks or discriminates against individuals or groups
    - **Offensive Language**: Content that may be offensive but not necessarily hateful
    - **Targeted Content**: Content that specifically targets individuals or groups
    
    **Note**: This system is for educational and research purposes. Results should be interpreted carefully and may not be 100% accurate.
    """)

if __name__ == "__main__":
    main()
