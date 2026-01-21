# 🔧 FIX: Render Not Using Your Start Command

## The Problem
Render is running: `gunicorn your_application.wsgi` (default template)
Instead of: `gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT --timeout 120 wsgi:application`

## ✅ Solution: Manual Configuration in Render Dashboard

### Step 1: Go to Your Service
1. Open https://dashboard.render.com
2. Click on your service: `mediapipe-emotion-detection`

### Step 2: Go to Settings
- Click the **"Settings"** tab (left sidebar)

### Step 3: Find "Start Command" Section
- Scroll down to find **"Start Command"** field

### Step 4: Clear and Set Start Command
1. **Delete** any existing text in the "Start Command" field
2. **Paste** this exact command:
   ```
   gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT --timeout 120 wsgi:application
   ```

### Step 5: Verify Other Settings

**Root Directory:**
- Should be: `mediapipe`
- If blank or different, change it to: `mediapipe`

**Build Command:**
- Should be: `pip install --upgrade pip && pip install -r requirements.txt`

**Python Version:**
- In Environment Variables section, add/verify:
  - Key: `PYTHON_VERSION`
  - Value: `3.11.0` (or `3.10.0` if 3.11 has issues)

### Step 6: Save Changes
- Click **"Save Changes"** button at the bottom

### Step 7: Manual Deploy
1. Go to **"Manual Deploy"** tab
2. Click **"Deploy latest commit"**
3. Wait for deployment

## ✅ Expected Result

After deployment, you should see in logs:
```
==> Running 'gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT --timeout 120 wsgi:application'
```

Instead of:
```
==> Running 'gunicorn your_application.wsgi'  ❌
```

## 🎯 Quick Copy-Paste Commands

### Start Command:
```bash
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT --timeout 120 wsgi:application
```

### Build Command:
```bash
pip install --upgrade pip && pip install -r requirements.txt
```

### Root Directory:
```
mediapipe
```

## 🔍 Verify Files Are Correct

Make sure these files exist in your `mediapipe` directory:
- ✅ `wsgi.py` - WSGI entry point
- ✅ `Procfile` - Contains start command
- ✅ `requirements.txt` - With gunicorn and eventlet
- ✅ `Main.py` - Your Flask app

## ❓ Still Not Working?

1. **Check Runtime Logs** - See what command is actually being run
2. **Verify Root Directory** - Must be exactly `mediapipe` (case-sensitive)
3. **Check Python Version** - Try `3.10.0` if `3.11.0` has issues
4. **Verify wsgi.py exists** - Should be in `mediapipe/wsgi.py`

## 📝 Why This Happens

Render sometimes doesn't automatically detect `render.yaml` or `Procfile` if:
- Service was created before these files existed
- Service was created through dashboard (not via render.yaml)
- Root directory wasn't set correctly initially

**Solution:** Always manually set the start command in dashboard to ensure it's used.








