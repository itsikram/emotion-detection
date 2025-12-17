# Deploying to Render.com - Step by Step Guide

This guide will help you deploy your MediaPipe emotion detection server to Render.com's free tier.

## Prerequisites

1. A GitHub account
2. A Render.com account (sign up at https://render.com)
3. Your code pushed to a GitHub repository

## Step 1: Prepare Your Repository

1. Make sure all files are committed to your GitHub repository:
   - `Main.py`
   - `requirements.txt`
   - `render.yaml`
   - `templates/index.html`
   - `.gitignore`

2. Push your code to GitHub if you haven't already:
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

## Step 2: Create a New Web Service on Render

1. **Log in to Render.com**
   - Go to https://dashboard.render.com
   - Sign in or create an account

2. **Create New Web Service**
   - Click "New +" button in the top right
   - Select "Web Service"

3. **Connect Your Repository**
   - If this is your first time, click "Connect account" to connect your GitHub account
   - Select your repository from the list
   - Click "Connect"

4. **Configure the Service**
   - **Name**: `mediapipe-emotion-detection` (or any name you prefer)
   - **Region**: Choose closest to you (Oregon is default in render.yaml)
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: `mediapipe` (important!)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `python Main.py`
   - **Plan**: Select "Free" (note: free tier has limitations)

5. **Environment Variables** (Optional - already in render.yaml)
   - `PYTHON_VERSION`: `3.11.0`
   - `PORT`: `10000` (Render will override this automatically)
   - `TF_CPP_MIN_LOG_LEVEL`: `2`
   - `GLOG_minloglevel`: `2`

6. **Click "Create Web Service"**

## Step 3: Wait for Deployment

1. Render will start building your service
2. This may take 10-15 minutes on the free tier (heavy dependencies like TensorFlow and MediaPipe)
3. You can watch the build logs in real-time
4. Once deployed, you'll get a URL like: `https://mediapipe-emotion-detection.onrender.com`

## Step 4: Test Your Deployment

1. Visit your Render URL
2. You should see the web interface
3. Test the emotion detection functionality

## Step 5: Using Optimized Requirements (Optional)

If you encounter memory issues during build or runtime:

1. The `requirements-render.txt` file uses `opencv-python-headless` (lighter version)
2. TensorFlow is commented out (MediaPipe face mesh works without it)
3. To use optimized requirements, update `render.yaml`:
   - Change `requirements.txt` to `requirements-render.txt` in buildCommand
   - Or manually change build command in Render dashboard

## Important Notes for Free Tier

### Limitations:
- **Spins down after 15 minutes of inactivity** - First request after spin-down takes ~30-50 seconds
- **512MB RAM limit** - MediaPipe and TensorFlow are memory-intensive
- **Limited CPU** - Processing may be slower than local
- **Build time limits** - Large dependencies may cause timeouts

### Optimization Tips:

1. **Reduce Dependencies** (if possible):
   - Consider removing TensorFlow if not needed
   - MediaPipe is required for face detection

2. **Memory Management**:
   - The code already uses thread-local storage to prevent memory leaks
   - Monitor memory usage in Render dashboard

3. **Handle Spin-downs**:
   - First request after inactivity will be slow
   - Consider using a service like UptimeRobot to ping your service every 10 minutes

4. **Upgrade Considerations**:
   - If you need better performance, consider Render's paid plans
   - Alternative: Use Railway.app or Fly.io which have different free tier policies

## Troubleshooting

### Build Fails
- **Issue**: Build timeout or memory error
- **Solution**: 
  - Check build logs for specific errors
  - Try reducing dependencies
  - Consider using Python 3.10 instead of 3.11

### Service Crashes
- **Issue**: Service starts but crashes
- **Solution**:
  - Check logs in Render dashboard
  - Verify all dependencies are in requirements.txt
  - Check that PORT environment variable is being used

### WebSocket Issues
- **Issue**: WebSocket connections fail
- **Solution**:
  - Render free tier supports WebSockets
  - Check that CORS is properly configured (already set to "*")
  - Verify gevent is installed (in requirements.txt)

### Memory Issues
- **Issue**: Service runs out of memory
- **Solution**:
  - Reduce concurrent sessions
  - Optimize MediaPipe usage
  - Consider upgrading to paid plan

## Alternative: Manual Deployment (Without render.yaml)

If you prefer to configure manually in Render dashboard:

1. **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
2. **Start Command**: `python Main.py`
3. **Environment Variables**:
   - `PYTHON_VERSION`: `3.11.0`
   - `TF_CPP_MIN_LOG_LEVEL`: `2`
   - `GLOG_minloglevel`: `2`
4. **Root Directory**: `mediapipe` (if your repo root is not mediapipe folder)

## Monitoring

- **Logs**: View real-time logs in Render dashboard
- **Metrics**: Monitor CPU, Memory, and Network usage
- **Events**: Track deployments and service events

## Updating Your Service

1. Push changes to your GitHub repository
2. Render will automatically detect changes and redeploy
3. Or manually trigger a deploy from the Render dashboard

## Support

- Render Documentation: https://render.com/docs
- Render Community: https://community.render.com
- Check Render status: https://status.render.com

