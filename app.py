#!/usr/bin/env python3
"""
App entry point for Streamlit Cloud deployment
This file ensures compatibility with different deployment methods
"""

import streamlit as st
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Download NLTK data if not already present
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

# Import and run the main app
from streamlit_app import main

if __name__ == "__main__":
    main()
