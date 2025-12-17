# Installation Guide

## Python Version Requirement

MediaPipe currently supports Python 3.8, 3.9, 3.10, and 3.11. **Python 3.13 is not yet supported.**

## Solution Options

### Option 1: Use Python 3.11 (Recommended)

1. **Install Python 3.11**:
   - Download from [python.org](https://www.python.org/downloads/)
   - Or use pyenv/conda to manage multiple Python versions

2. **Create a virtual environment with Python 3.11**:
   ```bash
   # Using pyenv (if installed)
   pyenv install 3.11.9
   pyenv local 3.11.9
   
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Use Conda (Easier Python Version Management)

1. **Create a conda environment with Python 3.11**:
   ```bash
   conda create -n mediapipe-env python=3.11
   conda activate mediapipe-env
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Option 3: Check MediaPipe Updates

MediaPipe is actively developed. Check the latest version:
```bash
pip install mediapipe --upgrade
```

Or check the official repository: https://github.com/google/mediapipe

## Verification

After installation, verify MediaPipe works:
```python
import mediapipe as mp
print(mp.__version__)
```

## Troubleshooting

### "No matching distribution found"
- Your Python version is not supported
- Use Python 3.8-3.11 instead

### Import errors after installation
- Ensure you're using the correct Python version
- Try reinstalling: `pip uninstall mediapipe && pip install mediapipe`

