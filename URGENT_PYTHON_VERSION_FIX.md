# ⚠️ URGENT: Set Python Version in Render Dashboard

## The Problem
Render is **ignoring** `runtime.txt` and `PYTHON_VERSION` environment variable, and using Python 3.13.4 (default).

## ✅ IMMEDIATE FIX (Do This Now)

### Step 1: Go to Render Dashboard
1. Open https://dashboard.render.com
2. Click your service: `mediapipe-emotion-detection`

### Step 2: Go to Settings
- Click **"Settings"** tab (left sidebar)

### Step 3: Find "Python Version" Section
- Scroll down to find **"Python Version"** dropdown/field
- OR look for **"Environment Variables"** section

### Step 4: Set Python Version
**Option A: If there's a "Python Version" dropdown:**
- Select: `3.11.9` (or `3.11` if 3.11.9 not available)

**Option B: If using Environment Variables:**
- Add/Edit environment variable:
  - **Key**: `PYTHON_VERSION`
  - **Value**: `3.11.9`
- Click **"Save"**

### Step 5: Save and Redeploy
1. Click **"Save Changes"** at bottom
2. Go to **"Manual Deploy"** tab
3. Click **"Deploy latest commit"**

## 🎯 Why This Is Critical

- **Python 3.13.4** builds numpy from source (takes 10-15 minutes, may fail)
- **Python 3.11.9** uses pre-built wheels (faster, more reliable)
- **MediaPipe** works better with Python 3.11

## 📋 Alternative: Cancel Current Build

If the current build is taking too long:

1. **Cancel** the current build in Render dashboard
2. **Set Python version** to 3.11.9 (as above)
3. **Redeploy** - should be much faster

## ✅ Expected Result

After setting Python 3.11.9:
```
==> Installing Python version 3.11.9...
==> Using Python version 3.11.9
```

Instead of:
```
==> Installing Python version 3.13.4... ❌
==> Using Python version 3.13.4 (default) ❌
```

## 🔍 Where to Find Python Version Setting

In Render dashboard, it might be:
- **Settings → Python Version** (dropdown)
- **Settings → Environment Variables → PYTHON_VERSION**
- **Settings → Build & Deploy → Python Version**

Look for any field related to "Python" or "Runtime" version.








