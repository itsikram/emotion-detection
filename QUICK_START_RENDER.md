# Quick Start: Deploy to Render.com

## Fast Deployment (5 minutes)

### 1. Push to GitHub
```bash
cd mediapipe
git add .
git commit -m "Ready for Render deployment"
git push
```

### 2. Deploy on Render
1. Go to https://dashboard.render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `mediapipe-emotion-detection`
   - **Root Directory**: `mediapipe`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `python Main.py`
   - **Plan**: Free
5. Click "Create Web Service"
6. Wait 10-15 minutes for build
7. Done! Your service will be live at `https://your-service.onrender.com`

## If Build Fails (Memory Issues)

Use the optimized requirements instead:

1. In Render dashboard, change **Build Command** to:
   ```
   pip install --upgrade pip && pip install -r requirements-render.txt
   ```
2. This uses lighter dependencies (no TensorFlow, headless OpenCV)

## Important Notes

- ⚠️ **Free tier spins down after 15 min inactivity** - First request will be slow (~30s)
- 💾 **512MB RAM limit** - May be tight with MediaPipe + TensorFlow
- 🐌 **Build takes 10-15 min** - Heavy dependencies (TensorFlow, MediaPipe, OpenCV)

## Keep Service Alive (Optional)

To prevent spin-downs, use a free service like UptimeRobot:
1. Sign up at https://uptimerobot.com
2. Add monitor for your Render URL
3. Set interval to 5 minutes

## Need Help?

See `RENDER_DEPLOYMENT.md` for detailed instructions and troubleshooting.








