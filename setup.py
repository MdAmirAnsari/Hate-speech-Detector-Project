#!/usr/bin/env python3
"""
Setup script for Hate Speech Detection Project
Run this to install all dependencies and download NLTK data
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("✅ NLTK data downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False
    return True

def verify_setup():
    """Verify that everything is working."""
    print("Verifying setup...")
    try:
        # Test imports
        import pandas as pd
        import numpy as np
        import sklearn
        import matplotlib.pyplot as plt
        import seaborn as sns
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        # Test NLTK data
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize("This is a test sentence.")
        
        print("✅ All imports successful!")
        print(f"✅ NLTK working: {len(stop_words)} stopwords, {len(tokens)} tokens")
        
        # Check if models exist
        model_files = [
            "models/best_model_task_1.pkl",
            "models/best_model_task_2.pkl",
            "models/best_model_task_3.pkl"
        ]
        
        models_exist = all(os.path.exists(f) for f in model_files)
        if models_exist:
            print("✅ All trained models found!")
        else:
            print("⚠️  Some trained models missing. You may need to run training scripts.")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Main setup function."""
    print("HATE SPEECH DETECTION PROJECT SETUP")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected!")
    else:
        print("⚠️  Warning: Not in a virtual environment. Consider using one.")
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation.")
        return False
    
    # Download NLTK data
    if not download_nltk_data():
        print("❌ Setup failed at NLTK data download.")
        return False
    
    # Verify setup
    if not verify_setup():
        print("❌ Setup verification failed.")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETE!")
    print("=" * 50)
    print("\nYou can now use the hate speech detection system:")
    print("1. cd scripts")
    print("2. python predict_hate_speech.py")
    print("3. python simple_usage_example.py")
    print("\nFor VS Code users:")
    print("- Open the project folder in VS Code")
    print("- The Python interpreter should be automatically configured")
    print("- Use Ctrl+Shift+P -> 'Python: Select Interpreter' if needed")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
