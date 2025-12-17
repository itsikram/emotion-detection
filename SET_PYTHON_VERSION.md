# 🔧 Set Python Version in Render Dashboard

## Current Issue
Render is using Python 3.13.4 (default) instead of Python 3.11.9 that we specified.

## ✅ Solution: Set Python Version Manually

The `PYTHON_VERSION` environment variable in `render.yaml` might not be working. Set it manually in the dashboard:

### Step 1: Go to Render Dashboard
1. Open https://dashboard.render.com
2. Click on your service: `mediapipe-emotion-detection`

### Step 2: Go to Settings → Environment
1. Click **"Settings"** tab
2. Scroll to **"Environment Variables"** section

### Step 3: Add/Update Python Version
1. Click **"Add Environment Variable"** (if PYTHON_VERSION doesn't exist)
2. Or click **"Edit"** on existing PYTHON_VERSION
3. Set:
   - **Key**: `PYTHON_VERSION`
   - **Value**: `3.11.9`
4. Click **"Save Changes"**

### Step 4: Alternative - Use Runtime.txt
Create a `runtime.txt` file in your `mediapipe` directory:

```
python-3.11.9
```

Then commit and push:
```bash
cd mediapipe
echo "python-3.11.9" > runtime.txt
git add runtime.txt
git commit -m "Set Python version to 3.11.9"
git push
```

## 🎯 Why This Matters

- **Python 3.13.4** is very new and some packages (like numpy) need to build from source
- **Python 3.11.9** has pre-built wheels for most packages (faster builds)
- **MediaPipe** is better tested with Python 3.11

## ⏳ Current Build Status

Your build is currently:
- ✅ Installing dependencies
- ⏳ Building numpy from source (takes time on Python 3.13)
- ⏳ May take 10-15 minutes total

**You can:**
1. **Wait** for current build to finish (might work, but slower)
2. **Cancel** current build and set Python 3.11.9, then redeploy (recommended)

## 📋 Quick Action

**Option 1: Set in Dashboard (Immediate)**
- Go to Settings → Environment Variables
- Add `PYTHON_VERSION` = `3.11.9`
- Save and redeploy

**Option 2: Use runtime.txt (Permanent)**
- Create `runtime.txt` with `python-3.11.9`
- Commit and push
- Render will use it automatically

