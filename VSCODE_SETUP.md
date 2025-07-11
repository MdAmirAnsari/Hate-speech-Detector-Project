# VS Code Setup Guide for Hate Speech Detection Project

## 🔧 Fixing Import Issues in VS Code

### Problem
You're getting errors like:
- `Import "sklearn.linear_model" could not be resolved`
- `Import "nltk.corpus" could not be resolved`
- `Import "pandas" could not be resolved`

### Solution

Follow these steps to fix VS Code Python interpreter issues:

## 1. Select the Correct Python Interpreter

### Method 1: Using Command Palette
1. Open VS Code
2. Press `Ctrl + Shift + P` (or `Cmd + Shift + P` on Mac)
3. Type "Python: Select Interpreter"
4. Choose the interpreter that ends with:
   ```
   C:\Users\user\hate-speech-env\Scripts\python.exe
   ```

### Method 2: Using Bottom Status Bar
1. Look at the bottom-left corner of VS Code
2. Click on the Python version (e.g., "Python 3.13.3")
3. Select the correct interpreter from the list

## 2. Verify the Interpreter is Working

1. Open a Python file in VS Code
2. Check the bottom-left corner shows the virtual environment name: `(hate-speech-env)`
3. Open the integrated terminal (`Ctrl + ` ` or `View > Terminal`)
4. The terminal should show `(hate-speech-env)` at the beginning of the prompt

## 3. If Issues Persist

### Method 1: Reload VS Code
1. Press `Ctrl + Shift + P`
2. Type "Developer: Reload Window"
3. Press Enter

### Method 2: Clear VS Code Cache
1. Close VS Code completely
2. Delete the `.vscode` folder in your project (if it exists)
3. Reopen VS Code and select the interpreter again

### Method 3: Manual Configuration
1. Create/edit `.vscode/settings.json` in your project folder:
   ```json
   {
       "python.defaultInterpreterPath": "C:\\Users\\user\\hate-speech-env\\Scripts\\python.exe",
       "python.terminal.activateEnvironment": true
   }
   ```

## 4. Test the Setup

1. Create a new Python file in VS Code
2. Type the following code:
   ```python
   import pandas as pd
   import numpy as np
   import sklearn
   import nltk
   print("All imports successful!")
   ```
3. Run the file - you should see no import errors

## 5. Running Scripts from VS Code

### Method 1: Run Button
1. Open any Python file in the `scripts` folder
2. Click the "Run Python File" button (▶️) in the top-right corner

### Method 2: Integrated Terminal
1. Open the integrated terminal (`Ctrl + ` `)
2. Navigate to the scripts folder:
   ```bash
   cd scripts
   ```
3. Run any script:
   ```bash
   python simple_usage_example.py
   python predict_hate_speech.py
   ```

### Method 3: Debug Configuration
1. Press `F5` or go to `Run > Start Debugging`
2. Select from the predefined configurations:
   - "Python: Hate Speech Predictor"
   - "Python: Simple Usage Example"
   - "Python: Current File"

## 6. Common Issues and Solutions

### Issue: "No module named 'pandas'"
**Solution**: The wrong Python interpreter is selected. Go back to Step 1.

### Issue: Terminal doesn't show "(hate-speech-env)"
**Solution**: 
1. Close the terminal
2. Activate the virtual environment manually:
   ```bash
   .\hate-speech-env\Scripts\Activate.ps1
   ```
3. Or restart VS Code after selecting the correct interpreter

### Issue: Pylance still shows import errors
**Solution**: 
1. Press `Ctrl + Shift + P`
2. Type "Python: Refresh Language Server"
3. Press Enter

### Issue: Can't find the virtual environment
**Solution**: 
1. Make sure you're in the correct project directory
2. Check if the virtual environment exists:
   ```bash
   ls ../hate-speech-env/Scripts/
   ```
3. If it doesn't exist, create it again:
   ```bash
   python -m venv ../hate-speech-env
   ```

## 7. Alternative: Use Python Extension Settings

1. Open VS Code settings (`Ctrl + ,`)
2. Search for "python.defaultInterpreterPath"
3. Set it to: `C:\Users\user\hate-speech-env\Scripts\python.exe`
4. Search for "python.terminal.activateEnvironment"
5. Make sure it's checked (enabled)

## 8. Verification Checklist

✅ VS Code shows the correct Python interpreter in the bottom-left corner
✅ Integrated terminal shows `(hate-speech-env)` in the prompt
✅ No import errors when typing `import pandas`
✅ Can run scripts without "module not found" errors
✅ Pylance/IntelliSense works correctly

## 9. If Nothing Works

### Nuclear Option: Reset Everything
1. Close VS Code
2. Delete the `.vscode` folder in your project
3. Deactivate and delete the virtual environment:
   ```bash
   rm -rf ../hate-speech-env
   ```
4. Recreate the virtual environment:
   ```bash
   python -m venv ../hate-speech-env
   ```
5. Activate it and install requirements:
   ```bash
   ../hate-speech-env/Scripts/Activate.ps1
   pip install -r requirements.txt
   python setup.py
   ```
6. Reopen VS Code and select the interpreter

---

**Need Help?** 
- Check the project's README.md for more information
- Run `python setup.py` to verify your environment
- Make sure you're working in the correct project directory
