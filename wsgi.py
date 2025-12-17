"""
WSGI entry point for Flask-SocketIO application
Used by gunicorn for production deployment with eventlet workers
"""
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import the app and socketio from Main
# The socketio is already attached to the app, so we just need the app
from Main import app

# For Flask-SocketIO with gunicorn + eventlet workers:
# - The eventlet worker automatically handles WebSocket connections
# - The socketio object attached to app will work automatically
# - socketio.run() is NOT used - gunicorn handles the server
application = app

