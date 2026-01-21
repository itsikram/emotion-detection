# Manual Render Dashboard Configuration

Since Render is not reading the `render.yaml` file automatically, you need to configure it manually in the dashboard.

## Steps to Fix in Render Dashboard

1. **Go to your service in Render Dashboard**
   - Navigate to: https://dashboard.render.com
   - Click on your service

2. **Go to Settings Tab**

3. **Update the following settings:**

   ### Build Command:
   ```
   pip install --upgrade pip && pip install -r requirements.txt
   ```

   ### Start Command:
   ```
   gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT --timeout 120 wsgi:application
   ```

   ### Environment Variables:
   Add these if not already present:
   - `PYTHON_VERSION` = `3.11.0` (or `3.10.0` if 3.11 doesn't work)
   - `PYTHONUNBUFFERED` = `1`
   - `TF_CPP_MIN_LOG_LEVEL` = `2`
   - `GLOG_minloglevel` = `2`
   - `PORT` = `10000` (Render will override this automatically, but set it anyway)

4. **Save Changes**

5. **Manual Deploy**
   - Go to "Manual Deploy" tab
   - Click "Deploy latest commit"

## Alternative: Use Procfile

If the above doesn't work, Render should automatically detect the `Procfile` in your `mediapipe` directory.

The Procfile contains:
```
web: gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT --timeout 120 wsgi:application
```

Make sure:
- Procfile is in the `mediapipe` directory (where your code is)
- Root Directory in Render is set to `mediapipe`

## Verify Files Are Correct

Make sure these files exist in your `mediapipe` directory:
- ✅ `wsgi.py` - WSGI entry point
- ✅ `Procfile` - Start command
- ✅ `requirements.txt` - With gunicorn and eventlet
- ✅ `Main.py` - Your Flask app
- ✅ `templates/index.html` - Your HTML template

## If Still Not Working

1. **Check Build Logs** - Make sure gunicorn installs successfully
2. **Check Runtime Logs** - See what command is actually being run
3. **Try Python 3.10** - Change PYTHON_VERSION to 3.10.0 (more stable)
4. **Verify Root Directory** - Make sure it's set to `mediapipe` not root








