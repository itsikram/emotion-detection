import warnings
import os
import sys
from contextlib import redirect_stderr
from io import StringIO

# Suppress protobuf warnings and TensorFlow warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warning messages
os.environ['GLOG_minloglevel'] = '2'  # Suppress MediaPipe/Google Logging (GLOG) messages

import cv2
import numpy as np
import base64
import mediapipe as mp
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from io import BytesIO
from PIL import Image
from collections import deque
import time
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# Enable CORS for all origins on Flask app
CORS(app, origins="*", resources={r"/*": {"origins": "*"}})

# Try to use gevent for better WebSocket support, fallback to threading
try:
    import gevent
    async_mode = 'gevent'
    print("Using gevent async mode for WebSocket support")
except ImportError:
    async_mode = 'threading'
    print("Using threading async mode (gevent not available)")

# Configure Socket.IO with proper async mode and error handling
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    async_mode=async_mode,
    logger=False,  # Reduce logging noise
    engineio_logger=False,  # Reduce engineio logging noise
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=10 * 1024 * 1024,  # 10MB max payload (default is 1MB) - for large base64 images
    allow_upgrades=True,
    transports=['websocket', 'polling'],
    socketio_path='socket.io'  # Explicitly set socket.io path
)

# Initialize MediaPipe Face Mesh for additional features
# Use static_image_mode=True since we're processing individual frames from browser
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Thread-local storage for MediaPipe face_mesh instances
# This prevents cross-contamination between concurrent sessions
_thread_local = threading.local()

def get_face_mesh():
    """Get or create a thread-local MediaPipe FaceMesh instance"""
    if not hasattr(_thread_local, 'face_mesh'):
        _thread_local.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,  # Changed to True for frame-by-frame processing
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return _thread_local.face_mesh

# Using MediaPipe-only emotion detection (FER has protobuf conflicts)
# FER library removed due to protobuf conflicts with MediaPipe
# Using MediaPipe-only detection which supports all emotions
fer_detector = None
print("Using MediaPipe-based emotion detection (all emotions supported)")

# Face mesh landmark indices
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
TOP_LIP = 13
BOTTOM_LIP = 14
LEFT_BROW_INNER = 285
RIGHT_BROW_INNER = 55
NOSE_TIP = 1
# Eye corner landmarks for EAR calculation
LEFT_EYE_OUTER = 33   # Left eye outer corner
LEFT_EYE_INNER = 133  # Left eye inner corner (alternative: 39)
RIGHT_EYE_OUTER = 263 # Right eye outer corner
RIGHT_EYE_INNER = 362 # Right eye inner corner (alternative: 42)
CHIN = 18

# Session state management for multi-user support
# Each client session has isolated state to prevent interference
session_states = {}  # Dictionary mapping session_id -> state dict
DISTANCE_CHANGE_THRESHOLD = 0.20  # If face size changes >20%, reset calibration
MIN_STABLE_FRAMES = 5  # Require 5 consecutive frames before changing dominant emotion
MIN_STABLE_EXPRESSION_FRAMES = 5  # Require 5 consecutive frames before changing dominant expression

def get_session_state(session_id):
    """Get or create session state for a client"""
    if session_id not in session_states:
        # Initialize new session state
        session_states[session_id] = {
            # Tracking for temporal features
            'eye_blink_history': deque(maxlen=15),
            'head_position_history': deque(maxlen=15),
            'mouth_openness_history': deque(maxlen=15),
            'mouth_width_history': deque(maxlen=15),
            'ear_history': deque(maxlen=15),
            'smile_intensity_history': deque(maxlen=15),
            'mouth_corner_history': deque(maxlen=15),
            'sad_expression_history': deque(maxlen=15),
            'emotion_history': deque(maxlen=15),
            'eyebrow_position_history': deque(maxlen=15),
            
            # Stable dominant emotion tracking
            'stable_dominant_emotion': 'neutral',
            'stable_emotion_count': 0,
            'stable_emotion_history': deque(maxlen=7),
            
            # Expression/Action tracking
            'expression_history': deque(maxlen=15),
            'stable_dominant_expression': 'none',
            'stable_expression_count': 0,
            'stable_expression_history': deque(maxlen=7),
            
            # Baseline calibration
            'baseline_mouth_width_ratio': None,
            'baseline_mouth_corner_position': None,
            'baseline_ear': None,
            'baseline_brow_ratio': None,
            'baseline_brow_to_eye': None,
            'baseline_calibration_frames': deque(maxlen=30),
            'baseline_face_width': None,
            'last_blink_time': time.time(),
            'blink_count': 0,
            
            # EMA smoothing state (for faster emotion updates)
            'last_smoothed_emotions': None
        }
    return session_states[session_id]

def cleanup_session_state(session_id):
    """Clean up session state when client disconnects"""
    if session_id in session_states:
        del session_states[session_id]
        print(f"Cleaned up session state for client {session_id}")

def calculate_distance(landmark1, landmark2, image_width, image_height):
    """Calculate Euclidean distance between two MediaPipe landmarks"""
    x1 = landmark1.x * image_width
    y1 = landmark1.y * image_height
    x2 = landmark2.x * image_width
    y2 = landmark2.y * image_height
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def validate_face_quality(landmarks, image_width, image_height):
    """Validate face detection quality - reject poor quality detections"""
    try:
        left_eye_outer = landmarks.landmark[LEFT_EYE_OUTER]
        right_eye_outer = landmarks.landmark[RIGHT_EYE_OUTER]
        nose_tip = landmarks.landmark[NOSE_TIP]
        
        # Calculate face width
        face_width = calculate_distance(left_eye_outer, right_eye_outer, image_width, image_height)
        face_width_norm = face_width / image_width if image_width > 0 else 0
        
        # 1. Face size check (should be reasonable size)
        if face_width_norm < 0.10 or face_width_norm > 0.50:  # Too small or too large
            return False, "Face size out of range"
        
        # 2. Face position check (should be reasonably centered)
        face_center_x = (left_eye_outer.x + right_eye_outer.x) / 2
        face_center_y = (left_eye_outer.y + right_eye_outer.y) / 2
        horizontal_offset = abs(face_center_x - 0.5)
        vertical_offset = abs(face_center_y - 0.5)
        
        if horizontal_offset > 0.4 or vertical_offset > 0.4:  # Too far from center
            return False, "Face position out of range"
        
        # 3. Face angle check (should be facing forward, not too tilted)
        eye_level_diff = abs(left_eye_outer.y - right_eye_outer.y)
        if eye_level_diff > 0.05:  # Eyes not level (face tilted)
            return False, "Face angle too extreme"
        
        # 4. Landmark validity check (check if key landmarks are reasonable)
        # Check if mouth is below nose and eyes are above nose
        left_eye = landmarks.landmark[LEFT_EYE_INNER]
        right_eye = landmarks.landmark[RIGHT_EYE_INNER]
        left_mouth = landmarks.landmark[LEFT_MOUTH_CORNER]
        right_mouth = landmarks.landmark[RIGHT_MOUTH_CORNER]
        
        avg_eye_y = (left_eye.y + right_eye.y) / 2
        avg_mouth_y = (left_mouth.y + right_mouth.y) / 2
        
        if avg_eye_y > nose_tip.y or avg_mouth_y < nose_tip.y:  # Invalid landmark positions
            return False, "Invalid landmark positions"
        
        # 5. Symmetry check (face should be reasonably symmetric)
        left_eye_distance = abs(left_eye.x - nose_tip.x)
        right_eye_distance = abs(right_eye.x - nose_tip.x)
        symmetry_diff = abs(left_eye_distance - right_eye_distance)
        
        if symmetry_diff > 0.08:  # Too asymmetric (profile view or error)
            return False, "Face too asymmetric"
        
        # All checks passed
        return True, "Good quality"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def update_baseline_calibration(landmarks, image_width, image_height, session_state):
    """Update baseline calibration for adaptive thresholds (normalize per person, distance-aware)"""
    
    try:
        # Calculate key features
        left_mouth = landmarks.landmark[LEFT_MOUTH_CORNER]
        right_mouth = landmarks.landmark[RIGHT_MOUTH_CORNER]
        top_lip = landmarks.landmark[TOP_LIP]
        bottom_lip = landmarks.landmark[BOTTOM_LIP]
        left_eye_outer = landmarks.landmark[LEFT_EYE_OUTER]
        right_eye_outer = landmarks.landmark[RIGHT_EYE_OUTER]
        left_eye_top = landmarks.landmark[LEFT_EYE_TOP]
        left_eye_bottom = landmarks.landmark[LEFT_EYE_BOTTOM]
        right_eye_top = landmarks.landmark[RIGHT_EYE_TOP]
        right_eye_bottom = landmarks.landmark[RIGHT_EYE_BOTTOM]
        left_brow_inner = landmarks.landmark[LEFT_BROW_INNER]
        right_brow_inner = landmarks.landmark[RIGHT_BROW_INNER]
        
        face_width = calculate_distance(left_eye_outer, right_eye_outer, image_width, image_height)
        
        # Normalize face width by image size for distance-invariant comparison
        face_width_normalized = face_width / image_width if image_width > 0 else 0
        
        # Check for significant distance change - if face size changes >20%, reset calibration
        if session_state['baseline_face_width'] is not None:
            distance_change_ratio = abs(face_width_normalized - session_state['baseline_face_width']) / session_state['baseline_face_width']
            if distance_change_ratio > DISTANCE_CHANGE_THRESHOLD:
                # Significant distance change detected - reset calibration
                session_state['baseline_calibration_frames'].clear()
                session_state['baseline_mouth_width_ratio'] = None
                session_state['baseline_mouth_corner_position'] = None
                session_state['baseline_ear'] = None
                session_state['baseline_brow_ratio'] = None
                session_state['baseline_brow_to_eye'] = None
                session_state['baseline_face_width'] = face_width_normalized
                # Don't continue calibration this frame - let it rebuild
                return
        
        if face_width > 0:
            # Update baseline face width (normalized)
            if session_state['baseline_face_width'] is None:
                session_state['baseline_face_width'] = face_width_normalized
            else:
                # Smooth update of face width (moving average to handle small fluctuations)
                session_state['baseline_face_width'] = session_state['baseline_face_width'] * 0.9 + face_width_normalized * 0.1
            
            # Calculate baseline features (all normalized by face width for distance invariance)
            mouth_width = calculate_distance(left_mouth, right_mouth, image_width, image_height)
            mouth_width_ratio = mouth_width / face_width
            
            mouth_center_y = (top_lip.y + bottom_lip.y) / 2
            mouth_corner_y = (left_mouth.y + right_mouth.y) / 2
            # Normalize by face width (not image height) for distance invariance
            mouth_corner_position = (mouth_center_y - mouth_corner_y) * image_height / face_width
            
            # Calculate EAR (already normalized, distance-invariant)
            left_eye_vertical = calculate_distance(left_eye_top, left_eye_bottom, image_width, image_height)
            left_eye_horizontal = calculate_distance(left_eye_outer, landmarks.landmark[LEFT_EYE_INNER], image_width, image_height)
            right_eye_vertical = calculate_distance(right_eye_top, right_eye_bottom, image_width, image_height)
            right_eye_horizontal = calculate_distance(right_eye_outer, landmarks.landmark[RIGHT_EYE_INNER], image_width, image_height)
            
            if left_eye_horizontal > 0 and right_eye_horizontal > 0:
                left_ear = left_eye_vertical / left_eye_horizontal
                right_ear = right_eye_vertical / right_eye_horizontal
                avg_ear = (left_ear + right_ear) / 2
            else:
                avg_ear = 0.2  # Default
            
            brow_distance = calculate_distance(left_brow_inner, right_brow_inner, image_width, image_height)
            brow_ratio = brow_distance / face_width if face_width > 0 else 0.35
            
            # Calculate brow position (for eyebrow raised detection)
            brow_center_y = (left_brow_inner.y + right_brow_inner.y) / 2
            eye_top_y = (left_eye_top.y + right_eye_top.y) / 2
            brow_to_eye = (brow_center_y - eye_top_y) * image_height / face_width if face_width > 0 else 0.040
            
            # Store in calibration frames (all ratios are distance-invariant)
            session_state['baseline_calibration_frames'].append({
                'mouth_width_ratio': mouth_width_ratio,
                'mouth_corner_position': mouth_corner_position,
                'ear': avg_ear,
                'brow_ratio': brow_ratio,
                'brow_to_eye': brow_to_eye,  # Track brow position for eyebrow raised detection
                'face_width_normalized': face_width_normalized  # Store normalized face width
            })
            
            # Update baseline averages when we have enough frames (increased from 10 to 20 for better neutral calibration)
            if len(session_state['baseline_calibration_frames']) >= 20:
                # Use median for more robust baseline (less affected by outliers)
                calibration_data = list(session_state['baseline_calibration_frames'])
                session_state['baseline_mouth_width_ratio'] = np.median([f['mouth_width_ratio'] for f in calibration_data])
                session_state['baseline_mouth_corner_position'] = np.median([f['mouth_corner_position'] for f in calibration_data])
                session_state['baseline_ear'] = np.median([f['ear'] for f in calibration_data])
                session_state['baseline_brow_ratio'] = np.median([f['brow_ratio'] for f in calibration_data])
                session_state['baseline_brow_to_eye'] = np.median([f['brow_to_eye'] for f in calibration_data])
        
    except Exception as e:
        pass  # Silently fail calibration update

def detect_eye_blink(landmarks, image_width, image_height, session_state):
    """Detect eye blinking using Eye Aspect Ratio (EAR)"""
    
    # Get eye landmarks
    left_eye_top = landmarks.landmark[LEFT_EYE_TOP]
    left_eye_bottom = landmarks.landmark[LEFT_EYE_BOTTOM]
    right_eye_top = landmarks.landmark[RIGHT_EYE_TOP]
    right_eye_bottom = landmarks.landmark[RIGHT_EYE_BOTTOM]
    left_eye_outer = landmarks.landmark[LEFT_EYE_OUTER]
    left_eye_inner = landmarks.landmark[LEFT_EYE_INNER]
    right_eye_outer = landmarks.landmark[RIGHT_EYE_OUTER]
    right_eye_inner = landmarks.landmark[RIGHT_EYE_INNER]
    
    # Calculate EAR: (sum of vertical distances) / (2 * horizontal distance)
    # For left eye: vertical distances between top-bottom, and horizontal between outer-inner
    left_eye_vertical1 = calculate_distance(left_eye_top, left_eye_bottom, image_width, image_height)
    left_eye_horizontal = calculate_distance(left_eye_outer, left_eye_inner, image_width, image_height)
    
    if left_eye_horizontal == 0:
        left_eye_horizontal = 1.0  # Prevent division by zero
    left_ear = left_eye_vertical1 / left_eye_horizontal
    
    # For right eye
    right_eye_vertical1 = calculate_distance(right_eye_top, right_eye_bottom, image_width, image_height)
    right_eye_horizontal = calculate_distance(right_eye_outer, right_eye_inner, image_width, image_height)
    
    if right_eye_horizontal == 0:
        right_eye_horizontal = 1.0  # Prevent division by zero
    right_ear = right_eye_vertical1 / right_eye_horizontal
    
    avg_ear = (left_ear + right_ear) / 2
    session_state['eye_blink_history'].append(avg_ear)
    
    # Blink detection: EAR drops below threshold when eyes close
    # Normal open eye EAR is typically 0.2-0.4, closed eye is < 0.15
    BLINK_THRESHOLD = 0.15  # Lower threshold for better detection
    
    # Detect blink: current frame is below threshold AND previous frame was above threshold
    is_blinking = False
    if len(session_state['eye_blink_history']) >= 2:
        current_ear = avg_ear
        previous_ear = session_state['eye_blink_history'][-2] if len(session_state['eye_blink_history']) >= 2 else current_ear
        
        # Blink detected if: current EAR is low AND previous EAR was higher (eye closing transition)
        if current_ear < BLINK_THRESHOLD and previous_ear > BLINK_THRESHOLD:
            current_time = time.time()
            # Debounce: only count if enough time has passed since last blink
            if current_time - session_state['last_blink_time'] > 0.2:  # 200ms minimum between blinks
                session_state['blink_count'] += 1
                session_state['last_blink_time'] = current_time
                is_blinking = True
    
    return is_blinking, avg_ear

def detect_yawning(landmarks, image_width, image_height, session_state):
    """Detect yawning: very wide and open mouth, sustained opening (not rapid like speaking)"""
    
    top_lip = landmarks.landmark[TOP_LIP]
    bottom_lip = landmarks.landmark[BOTTOM_LIP]
    left_mouth = landmarks.landmark[LEFT_MOUTH_CORNER]
    right_mouth = landmarks.landmark[RIGHT_MOUTH_CORNER]
    
    mouth_height = calculate_distance(top_lip, bottom_lip, image_width, image_height)
    mouth_width = calculate_distance(left_mouth, right_mouth, image_width, image_height)
    
    if mouth_width > 0:
        mouth_openness = mouth_height / mouth_width
        session_state['mouth_openness_history'].append(mouth_openness)
        
        # Normalize by face width for distance invariance
        left_eye_outer = landmarks.landmark[LEFT_EYE_OUTER]
        right_eye_outer = landmarks.landmark[RIGHT_EYE_OUTER]
        face_width = calculate_distance(left_eye_outer, right_eye_outer, image_width, image_height)
        
        if face_width > 0:
            mouth_width_ratio = mouth_width / face_width
            
            # Yawning characteristics:
            # 1. Very large mouth openness (typically 0.5+ for true yawning, much higher than speaking)
            # 2. Sustained opening over multiple frames (yawning is slow, not rapid like speaking)
            # 3. Wide mouth width (not just vertically open, but also horizontally wide)
            # 4. Low variance in mouth movement (sustained state, not varying like speaking)
            
            # Single frame check: require very high openness (much higher than speaking)
            if mouth_openness > 0.50:  # Increased from 0.4 to 0.5 (yawning is very open)
                # Multi-frame analysis to distinguish from speaking
                if len(session_state['mouth_openness_history']) >= 5:
                    recent_openness = list(session_state['mouth_openness_history'])[-5:]
                    avg_openness = np.mean(recent_openness)
                    max_openness = max(recent_openness)
                    variance = np.var(recent_openness)
                    
                    # Yawning indicators:
                    # - Average openness is very high (>0.45)
                    # - Maximum openness is very high (>0.5)
                    # - Low variance (sustained opening, not rapid changes like speaking)
                    # - Wide mouth width (normal to wide, typically 0.42-0.55)
                    
                    if (avg_openness > 0.45 and  # Very high average openness
                        max_openness > 0.50 and  # At least one very open frame
                        variance < 0.003 and  # Low variance (sustained, not rapid changes)
                        0.42 < mouth_width_ratio < 0.58):  # Wide mouth (not narrow)
                        # Calculate intensity based on how open
                        intensity = min(1.0, (avg_openness / 0.60) * 0.8 + (max_openness / 0.70) * 0.2)
                        return True, float(intensity)
                    
                    # Very strong yawn: extremely open with sustained pattern
                    if (avg_openness > 0.55 and
                        max_openness > 0.60 and
                        variance < 0.002 and  # Very low variance
                        0.40 < mouth_width_ratio < 0.60):
                        intensity = min(1.0, avg_openness / 0.65)
                        return True, float(intensity)
                
                # Fallback for very strong single-frame yawns (but still check it's not just one frame)
                if mouth_openness > 0.60 and mouth_width_ratio > 0.45:
                    # Only return true if we have some history suggesting sustained opening
                    if len(session_state['mouth_openness_history']) >= 3:
                        recent = list(session_state['mouth_openness_history'])[-3:]
                        if all(o > 0.45 for o in recent):  # All recent frames are very open
                            intensity = min(1.0, mouth_openness / 0.70)
                            return True, float(intensity)
    
    return False, 0

def detect_smile_intensity(landmarks, image_width, image_height, session_state):
    """Calculate smile intensity (0-1) based on mouth width and corner position"""
    left_mouth = landmarks.landmark[LEFT_MOUTH_CORNER]
    right_mouth = landmarks.landmark[RIGHT_MOUTH_CORNER]
    top_lip = landmarks.landmark[TOP_LIP]
    bottom_lip = landmarks.landmark[BOTTOM_LIP]
    nose_tip = landmarks.landmark[NOSE_TIP]
    
    mouth_width = calculate_distance(left_mouth, right_mouth, image_width, image_height)
    # Use eye outer corners for more accurate face width
    face_width = calculate_distance(
        landmarks.landmark[LEFT_EYE_OUTER], 
        landmarks.landmark[RIGHT_EYE_OUTER], 
        image_width, image_height
    )
    
    if face_width > 0:
        mouth_width_ratio = mouth_width / face_width
        
        # Calculate mouth corner position relative to lip center
        # Smiling lifts the mouth corners upward
        mouth_center_y = (top_lip.y + bottom_lip.y) / 2
        mouth_corner_y = (left_mouth.y + right_mouth.y) / 2
        # Positive value means corners are above center (smiling)
        corner_raise = (mouth_center_y - mouth_corner_y) * image_height / face_width
        
        # Use adaptive thresholds based on baseline calibration (normalize per person)
        if session_state['baseline_mouth_width_ratio'] is not None:
            # Adaptive thresholds based on person's baseline
            neutral_mouth_width = session_state['baseline_mouth_width_ratio']
            neutral_corner_raise = session_state['baseline_mouth_corner_position'] if session_state['baseline_mouth_corner_position'] is not None else 0.0
            
            # Calculate deviation from baseline
            mouth_width_deviation = mouth_width_ratio - neutral_mouth_width
            corner_raise_deviation = corner_raise - neutral_corner_raise
            
            # Stricter thresholds with dead zone for neutral faces
            # Require BOTH width AND corner raise to be significant for any smile detection
            DEAD_ZONE_WIDTH = 0.03  # No smile if deviation is less than 3% of face width
            DEAD_ZONE_CORNER = 0.005  # No smile if corner raise is less than this
            
            # Check if we're in the neutral dead zone
            if abs(mouth_width_deviation) < DEAD_ZONE_WIDTH and abs(corner_raise_deviation) < DEAD_ZONE_CORNER:
                # Within dead zone - definitely neutral, no smile
                intensity = 0.0
            elif mouth_width_deviation < -0.015:
                # Narrower than baseline - not smiling (sad or neutral)
                intensity = 0.0
            elif mouth_width_deviation > 0.10 and corner_raise_deviation > 0.008:
                # Much wider than baseline AND corners raised - strong smile
                # Require BOTH conditions to be true
                width_component = min(1.0, (mouth_width_deviation - 0.10) / 0.15)  # Scale 0.10-0.25 to 0-1
                corner_component = min(1.0, (corner_raise_deviation - 0.008) / 0.020)  # Scale 0.008-0.028 to 0-1
                intensity = min(1.0, 0.5 + (width_component * 0.3 + corner_component * 0.2))
            elif mouth_width_deviation > 0.06 and corner_raise_deviation > 0.006:
                # Moderate smile - require BOTH width and corner raise
                width_factor = (mouth_width_deviation - 0.06) / 0.04  # Normalize 0.06-0.10 to 0-1
                corner_factor = (corner_raise_deviation - 0.006) / 0.010  # Normalize 0.006-0.016 to 0-1
                # Both must be positive, and we weight them
                if width_factor > 0 and corner_factor > 0:
                    intensity = min(0.6, width_factor * 0.4 + corner_factor * 0.3)
                else:
                    intensity = 0.0
            else:
                # Not enough deviation in both dimensions - neutral
                intensity = 0.0
        else:
            # Fallback to fixed thresholds if baseline not yet calibrated
            # Stricter thresholds - require BOTH wide mouth AND raised corners
            # Neutral mouth width ratio is typically 0.38-0.43
            # Smiling increases it to 0.48-0.55+ AND raises corners
            if mouth_width_ratio < 0.42:
                # Narrow mouth - not smiling (neutral or sad)
                intensity = 0.0
            elif mouth_width_ratio > 0.52 and corner_raise > 0.010:
                # Very wide AND corners raised - strong smile
                width_component = min(1.0, (mouth_width_ratio - 0.52) / 0.10)  # Scale 0.52-0.62 to 0-1
                corner_component = min(1.0, (corner_raise - 0.010) / 0.020)  # Scale 0.010-0.030 to 0-1
                intensity = min(1.0, 0.5 + (width_component * 0.3 + corner_component * 0.2))
            elif mouth_width_ratio > 0.48 and corner_raise > 0.008:
                # Moderate smile - require BOTH conditions
                width_factor = (mouth_width_ratio - 0.48) / 0.04  # Normalize 0.48-0.52 to 0-1
                corner_factor = (corner_raise - 0.008) / 0.010  # Normalize 0.008-0.018 to 0-1
                if width_factor > 0 and corner_factor > 0:
                    intensity = min(0.5, width_factor * 0.3 + corner_factor * 0.2)
                else:
                    intensity = 0.0
            else:
                # Not enough width or corner raise - neutral
                intensity = 0.0
        
        current_intensity = max(0.0, min(1.0, intensity))
        session_state['smile_intensity_history'].append(current_intensity)
        
        # Use temporal smoothing - average over recent frames for stability
        if len(session_state['smile_intensity_history']) >= 3:
            recent_intensities = list(session_state['smile_intensity_history'])[-5:]
            smoothed_intensity = np.mean(recent_intensities)
            return smoothed_intensity
        
        return current_intensity
    return 0.0

def detect_laughing(landmarks, image_width, image_height, session_state, smile_intensity=None):
    """Detect laughing: very wide mouth + smile + mouth openness over multiple frames"""
    
    left_mouth = landmarks.landmark[LEFT_MOUTH_CORNER]
    right_mouth = landmarks.landmark[RIGHT_MOUTH_CORNER]
    top_lip = landmarks.landmark[TOP_LIP]
    bottom_lip = landmarks.landmark[BOTTOM_LIP]
    
    mouth_width = calculate_distance(left_mouth, right_mouth, image_width, image_height)
    mouth_height = calculate_distance(top_lip, bottom_lip, image_width, image_height)
    face_width = calculate_distance(
        landmarks.landmark[LEFT_EYE_OUTER],
        landmarks.landmark[RIGHT_EYE_OUTER],
        image_width, image_height
    )
    
    if face_width > 0:
        mouth_width_ratio = mouth_width / face_width
        mouth_openness = mouth_height / mouth_width if mouth_width > 0 else 0
        
        session_state['mouth_width_history'].append(mouth_width_ratio)
        session_state['mouth_openness_history'].append(mouth_openness)
        
        # Laughing: very wide mouth + significant openness + smile indicators
        if len(session_state['mouth_width_history']) >= 5 and len(session_state['mouth_openness_history']) >= 5:
            avg_mouth_width = np.mean(list(session_state['mouth_width_history'])[-5:])
            avg_openness = np.mean(list(session_state['mouth_openness_history'])[-5:])
            max_openness = max(list(session_state['mouth_openness_history'])[-5:])
            
            # Get smile intensity from history or parameter
            if smile_intensity is not None:
                avg_smile = smile_intensity
            else:
                avg_smile = np.mean(list(session_state['smile_intensity_history'])[-5:]) if len(session_state['smile_intensity_history']) >= 5 else 0
            
            # Laughing indicators:
            # 1. Very wide mouth (0.55+ ratio)
            # 2. Moderate to high mouth openness (but not yawning level)
            # 3. Smile present
            if (avg_mouth_width > 0.55 and 
                0.12 < avg_openness < 0.30 and  # Open but not yawning
                avg_smile > 0.3):  # Smiling present (lowered threshold slightly)
                # Calculate laughing intensity - ensure it's higher than smile intensity
                width_factor = min(1.0, (avg_mouth_width - 0.55) / 0.15)
                openness_factor = min(1.0, max_openness / 0.25)
                base_intensity = (width_factor * 0.5 + openness_factor * 0.3 + avg_smile * 0.2)
                
                # Ensure laughing intensity is always higher than smile intensity
                laughing_intensity = min(1.0, base_intensity)
                if smile_intensity is not None and laughing_intensity <= smile_intensity:
                    # Boost laughing intensity to be at least 0.15 higher than smile
                    laughing_intensity = min(1.0, smile_intensity + 0.15)
                
                return True, laughing_intensity
            
            # Strong laughing: very wide + very open
            if avg_mouth_width > 0.58 and avg_openness > 0.15:
                strong_laugh_intensity = min(1.0, (avg_mouth_width - 0.55) * 3)
                # Ensure it's higher than smile intensity
                if smile_intensity is not None and strong_laugh_intensity <= smile_intensity:
                    strong_laugh_intensity = min(1.0, smile_intensity + 0.20)
                return True, strong_laugh_intensity
    
    return False, 0

def detect_sleepy(landmarks, image_width, image_height, session_state):
    """Detect sleepy: consistently low eye aspect ratio over multiple frames with sustained pattern"""
    
    left_eye_top = landmarks.landmark[LEFT_EYE_TOP]
    left_eye_bottom = landmarks.landmark[LEFT_EYE_BOTTOM]
    right_eye_top = landmarks.landmark[RIGHT_EYE_TOP]
    right_eye_bottom = landmarks.landmark[RIGHT_EYE_BOTTOM]
    left_eye_outer = landmarks.landmark[LEFT_EYE_OUTER]
    left_eye_inner = landmarks.landmark[LEFT_EYE_INNER]
    right_eye_outer = landmarks.landmark[RIGHT_EYE_OUTER]
    right_eye_inner = landmarks.landmark[RIGHT_EYE_INNER]
    
    # Calculate EAR (Eye Aspect Ratio)
    left_eye_vertical = calculate_distance(left_eye_top, left_eye_bottom, image_width, image_height)
    left_eye_horizontal = calculate_distance(left_eye_outer, left_eye_inner, image_width, image_height)
    if left_eye_horizontal == 0:
        left_eye_horizontal = 1.0
    left_ear = left_eye_vertical / left_eye_horizontal
    
    right_eye_vertical = calculate_distance(right_eye_top, right_eye_bottom, image_width, image_height)
    right_eye_horizontal = calculate_distance(right_eye_outer, right_eye_inner, image_width, image_height)
    if right_eye_horizontal == 0:
        right_eye_horizontal = 1.0
    right_ear = right_eye_vertical / right_eye_horizontal
    
    avg_ear = (left_ear + right_ear) / 2
    session_state['ear_history'].append(avg_ear)
    
    # Sleepy detection using multi-frame analysis
    # Need at least 6 frames to analyze pattern
    if len(session_state['ear_history']) >= 6:
        recent_ear = list(session_state['ear_history'])[-10:]  # Use last 10 frames if available
        avg_recent_ear = np.mean(recent_ear)
        min_recent_ear = min(recent_ear)
        max_recent_ear = max(recent_ear)
        
        # Count frames with low EAR (indicating closed/partially closed eyes)
        # Normal open eyes: EAR ~0.20-0.35
        # Partially closed (drowsy): EAR ~0.10-0.18
        # Closed: EAR <0.10
        drowsy_threshold = 0.18  # Threshold for drowsy eyes
        closed_threshold = 0.12  # Threshold for nearly closed eyes
        
        below_drowsy_count = sum(1 for ear in recent_ear if ear < drowsy_threshold)
        below_closed_count = sum(1 for ear in recent_ear if ear < closed_threshold)
        
        # Calculate variance - sleepy eyes have less variation (sustained closed state)
        ear_variance = np.var(recent_ear)
        
        # Strong sleepy indicators:
        # 1. Average EAR consistently below drowsy threshold
        # 2. Most frames have low EAR
        # 3. Low variance (sustained pattern, not blinking)
        
        if len(recent_ear) >= 8:
            # Very sleepy: average EAR very low, most frames closed
            if (avg_recent_ear < 0.12 and 
                below_closed_count >= 6 and
                ear_variance < 0.002):  # Low variance = sustained state
                # Calculate intensity: lower EAR = more sleepy
                intensity = 1.0 - (avg_recent_ear / 0.12)  # Normalize: 0.12 = 0%, 0.0 = 100%
                return True, min(1.0, max(0.0, intensity))
            
            # Sleepy: average EAR low, many frames below threshold
            if (avg_recent_ear < 0.16 and 
                below_drowsy_count >= 7 and
                below_closed_count >= 4 and
                ear_variance < 0.003):
                intensity = (0.18 - avg_recent_ear) / 0.06  # Scale 0.12-0.18 to 1-0
                return True, min(0.9, max(0.5, intensity))
        
        # Moderate sleepy: less strict criteria
        if len(recent_ear) >= 6:
            if (avg_recent_ear < 0.18 and 
                below_drowsy_count >= 5 and
                ear_variance < 0.004):
                intensity = (0.20 - avg_recent_ear) / 0.08  # Scale 0.12-0.20 to 1-0.25
                return True, min(0.75, max(0.3, intensity))
            
            # Light drowsy: consistently lower than normal
            if (avg_recent_ear < 0.20 and 
                below_drowsy_count >= 4):
                intensity = (0.22 - avg_recent_ear) / 0.10  # Scale 0.12-0.22 to 1-0
                return True, min(0.6, max(0.2, intensity))
    
    return False, 0

def detect_head_shake(landmarks, image_width, image_height, session_state):
    """Detect head shake by tracking nose position"""
    nose_tip = landmarks.landmark[NOSE_TIP]
    head_center_x = nose_tip.x
    
    session_state['head_position_history'].append(head_center_x)
    
    if len(session_state['head_position_history']) >= 5:
        # Calculate variance in head position
        positions = list(session_state['head_position_history'])
        variance = np.var(positions)
        # High variance indicates head shaking
        if variance > 0.001:
            return True, variance
    return False, 0

def detect_attention(landmarks, image_width, image_height, session_state):
    """Detect attention/focus based on eye openness and head orientation"""
    eye_blink, ear = detect_eye_blink(landmarks, image_width, image_height, session_state)
    
    # Eyes open and steady = attentive
    if ear > 0.3:  # Eyes open
        # Check if looking forward (nose alignment)
        nose_tip = landmarks.landmark[NOSE_TIP]
        left_eye = landmarks.landmark[LEFT_EYE_INNER]
        right_eye = landmarks.landmark[RIGHT_EYE_INNER]
        
        eye_center_x = (left_eye.x + right_eye.x) / 2
        nose_offset = abs(nose_tip.x - eye_center_x)
        
        # Low offset = looking forward = attentive
        if nose_offset < 0.05:
            return True, 1.0 - (nose_offset * 10)
    return False, 0.5

def detect_eyebrow_raised(landmarks, image_width, image_height, session_state):
    """Detect eyebrow raised using multi-frame analysis and adaptive baseline calibration"""
    
    left_brow_inner = landmarks.landmark[LEFT_BROW_INNER]
    right_brow_inner = landmarks.landmark[RIGHT_BROW_INNER]
    left_eye_top = landmarks.landmark[LEFT_EYE_TOP]
    right_eye_top = landmarks.landmark[RIGHT_EYE_TOP]
    
    # Calculate face width for normalization
    left_eye_outer = landmarks.landmark[LEFT_EYE_OUTER]
    right_eye_outer = landmarks.landmark[RIGHT_EYE_OUTER]
    face_width = calculate_distance(left_eye_outer, right_eye_outer, image_width, image_height)
    
    if face_width > 0:
        # Calculate brow-to-eye distance (normalized by face width)
        # Positive values mean brows are above eyes (raised), negative means lowered
        brow_center_y = (left_brow_inner.y + right_brow_inner.y) / 2
        eye_top_y = (left_eye_top.y + right_eye_top.y) / 2
        brow_to_eye_distance = (brow_center_y - eye_top_y) * image_height / face_width
        
        # Store in history for multi-frame analysis
        session_state['eyebrow_position_history'].append(brow_to_eye_distance)
        
        # Calculate brow spread (furrowed vs raised)
        brow_distance = calculate_distance(left_brow_inner, right_brow_inner, image_width, image_height)
        brow_ratio = brow_distance / face_width
        
        # Use adaptive baseline if available, otherwise use default
        if session_state['baseline_brow_to_eye'] is not None:
            baseline_brow_to_eye_value = session_state['baseline_brow_to_eye']
        else:
            # Default baseline (typical neutral position)
            baseline_brow_to_eye_value = 0.040
        
        # Calculate deviation from baseline (raised = positive deviation)
        brow_raise_deviation = brow_to_eye_distance - baseline_brow_to_eye_value
        
        # Check individual brow positions for symmetry
        left_brow_to_eye = (left_brow_inner.y - left_eye_top.y) * image_height / face_width
        right_brow_to_eye = (right_brow_inner.y - right_eye_top.y) * image_height / face_width
        brow_asymmetry = abs(left_brow_to_eye - right_brow_to_eye)
        
        # Multi-frame analysis for stability (more lenient)
        if len(session_state['eyebrow_position_history']) >= 3:
            recent_positions = list(session_state['eyebrow_position_history'])[-3:]  # Use fewer frames for faster response
            avg_position = np.mean(recent_positions)
            max_position = max(recent_positions)
            min_position = min(recent_positions)
            variance = np.var(recent_positions)
            
            # Calculate average and max deviation from baseline
            avg_deviation = avg_position - baseline_brow_to_eye_value
            max_deviation = max_position - baseline_brow_to_eye_value
            min_deviation = min_position - baseline_brow_to_eye_value
            
            # More lenient criteria for eyebrow raised detection:
            # 1. Brows above baseline (even slightly)
            # 2. Not furrowed (brow_ratio > 0.30, more lenient)
            # 3. Reasonably symmetric (asymmetry < 0.012, more lenient)
            # 4. Average position is clearly raised (avg_deviation > 0.003, lowered threshold)
            
            # Light raised eyebrows (more sensitive)
            if (avg_deviation > 0.003 and  # Lowered from 0.005 - brows slightly above baseline
                max_deviation > 0.005 and  # Lowered from 0.007 - at least one frame raised
                brow_ratio > 0.30 and  # Lowered from 0.32 - not furrowed (more lenient)
                brow_asymmetry < 0.012 and  # Raised from 0.008 - allow more asymmetry
                variance < 0.0003):  # Raised from 0.0001 - allow more variance
                # Calculate intensity based on how raised
                intensity = min(1.0, (avg_deviation / 0.018) * 0.6 + (max_deviation / 0.025) * 0.4)
                # Ensure minimum intensity for visibility
                intensity = max(0.2, intensity)
                return True, float(intensity)
            
            # Moderately raised eyebrows
            if (avg_deviation > 0.006 and
                max_deviation > 0.009 and
                brow_ratio > 0.32 and
                brow_asymmetry < 0.010 and
                variance < 0.0002):
                intensity = min(1.0, (avg_deviation / 0.020) * 0.7 + (max_deviation / 0.030) * 0.3)
                return True, float(intensity)
            
            # Very raised eyebrows
            if (avg_deviation > 0.010 and
                max_deviation > 0.015 and
                brow_ratio > 0.34 and
                brow_asymmetry < 0.008):
                intensity = min(1.0, avg_deviation / 0.025)
                return True, float(intensity)
        
        # Single frame check for very strong signals (more lenient)
        if (brow_raise_deviation > 0.012 and  # Lowered from 0.015
            brow_ratio > 0.30 and  # Lowered from 0.33
            brow_asymmetry < 0.015):  # Raised from 0.010
            # Only return true if we have some history
            if len(session_state['eyebrow_position_history']) >= 2:
                recent = list(session_state['eyebrow_position_history'])[-2:]
                if all(pos > baseline_brow_to_eye_value + 0.003 for pos in recent):  # Lowered from 0.005
                    intensity = min(1.0, brow_raise_deviation / 0.025)
                    intensity = max(0.25, intensity)  # Ensure minimum visibility
                    return True, float(intensity)
    
    return False, 0

def detect_crying(landmarks, image_width, image_height, session_state):
    """Detect crying (both regular and silent): sad expression + downturned mouth corners + eyes partially closed over multiple frames
    
    Returns:
        (is_crying, cry_score, is_silent_crying) - tuple of (bool, float, bool)
        is_silent_crying indicates if it's silent crying (closed mouth) vs regular crying
    """
    
    top_lip = landmarks.landmark[TOP_LIP]
    bottom_lip = landmarks.landmark[BOTTOM_LIP]
    left_mouth = landmarks.landmark[LEFT_MOUTH_CORNER]
    right_mouth = landmarks.landmark[RIGHT_MOUTH_CORNER]
    nose_tip = landmarks.landmark[NOSE_TIP]
    left_brow_inner = landmarks.landmark[LEFT_BROW_INNER]
    right_brow_inner = landmarks.landmark[RIGHT_BROW_INNER]
    
    # Get eye landmarks for EAR calculation
    left_eye_top = landmarks.landmark[LEFT_EYE_TOP]
    left_eye_bottom = landmarks.landmark[LEFT_EYE_BOTTOM]
    right_eye_top = landmarks.landmark[RIGHT_EYE_TOP]
    right_eye_bottom = landmarks.landmark[RIGHT_EYE_BOTTOM]
    left_eye_outer = landmarks.landmark[LEFT_EYE_OUTER]
    left_eye_inner = landmarks.landmark[LEFT_EYE_INNER]
    right_eye_outer = landmarks.landmark[RIGHT_EYE_OUTER]
    right_eye_inner = landmarks.landmark[RIGHT_EYE_INNER]
    
    # Calculate mouth features
    mouth_height = calculate_distance(top_lip, bottom_lip, image_width, image_height)
    mouth_width = calculate_distance(left_mouth, right_mouth, image_width, image_height)
    face_width = calculate_distance(
        landmarks.landmark[LEFT_EYE_OUTER],
        landmarks.landmark[RIGHT_EYE_OUTER],
        image_width, image_height
    )
    
    if face_width > 0 and mouth_width > 0:
        # Calculate mouth corner position relative to nose (downturned = crying indicator)
        mouth_center_y = (top_lip.y + bottom_lip.y) / 2
        mouth_corner_y = (left_mouth.y + right_mouth.y) / 2
        # Negative value means corners are below center (downturned = sad/crying)
        mouth_corner_drop = (mouth_corner_y - mouth_center_y) * image_height / face_width
        
        mouth_width_ratio = mouth_width / face_width
        mouth_openness = mouth_height / mouth_width
        
        # Track mouth openness for silent crying detection
        session_state['mouth_openness_history'].append(mouth_openness)
        
        # Calculate EAR for eye state
        left_eye_vertical = calculate_distance(left_eye_top, left_eye_bottom, image_width, image_height)
        left_eye_horizontal = calculate_distance(left_eye_outer, left_eye_inner, image_width, image_height)
        if left_eye_horizontal == 0:
            left_eye_horizontal = 1.0
        left_ear = left_eye_vertical / left_eye_horizontal
        
        right_eye_vertical = calculate_distance(right_eye_top, right_eye_bottom, image_width, image_height)
        right_eye_horizontal = calculate_distance(right_eye_outer, right_eye_inner, image_width, image_height)
        if right_eye_horizontal == 0:
            right_eye_horizontal = 1.0
        right_ear = right_eye_vertical / right_eye_horizontal
        avg_ear = (left_ear + right_ear) / 2
        
        # Calculate eyebrow position (furrowed = sad)
        brow_center_y = (left_brow_inner.y + right_brow_inner.y) / 2
        eye_top_y = (left_eye_top.y + right_eye_top.y) / 2
        brow_to_eye = (brow_center_y - eye_top_y) * image_height / face_width
        
        # Track features over frames
        session_state['mouth_corner_history'].append(mouth_corner_drop)
        session_state['ear_history'].append(avg_ear)  # Track EAR for crying analysis
        
        # Calculate current frame crying score
        sad_score = 0.0
        
        # Crying indicators (more sensitive thresholds):
        # 1. Downturned mouth corners
        if mouth_corner_drop > 0.008:  # Lowered threshold from 0.015
            sad_score += 0.5
        elif mouth_corner_drop > 0.005:  # Even lower for weak signals
            sad_score += 0.3
        
        # 2. Narrow mouth (sad expression)
        if mouth_width_ratio < 0.42:
            sad_score += 0.3
        elif mouth_width_ratio < 0.44:  # Slightly narrow
            sad_score += 0.15
        
        # 3. Eyes partially closed or closed (crying often involves closing eyes)
        if 0.05 < avg_ear < 0.20:  # Partially closed or closing
            sad_score += 0.3
        elif avg_ear < 0.05:  # Very closed (sobbing)
            sad_score += 0.4
        elif 0.20 < avg_ear < 0.25:  # Slightly closed
            sad_score += 0.1
        
        # 4. Moderate mouth openness (crying can involve open or closed mouth)
        if 0.08 < mouth_openness < 0.30:  # Wider range
            sad_score += 0.2
        elif mouth_openness > 0.30:  # Very open (wailing)
            sad_score += 0.15
        
        # 5. Check eyebrow position (sad brows)
        if brow_to_eye < 0.038:  # Lowered brows
            sad_score += 0.2
        
        session_state['sad_expression_history'].append(sad_score)
        
        # Check if it's silent crying (closed mouth) or regular crying (open mouth)
        is_silent_crying = False
        # Silent crying: closed or very slightly open mouth (openness < 0.12)
        if mouth_openness < 0.12:
            is_silent_crying = True
        
        # Single-frame strong crying detection (immediate response)
        if (mouth_corner_drop > 0.020 and  # Very downturned
            mouth_width_ratio < 0.40 and  # Very narrow mouth
            sad_score > 0.7):  # High sad score
            intensity = min(1.0, sad_score)
            return True, intensity, is_silent_crying
        
        # Multi-frame analysis (reduced from 6 to 3 frames for faster detection)
        if len(session_state['mouth_corner_history']) >= 3 and len(session_state['sad_expression_history']) >= 3:
            recent_drops = list(session_state['mouth_corner_history'])[-5:]  # Use last 5 frames if available
            recent_sad = list(session_state['sad_expression_history'])[-5:]
            
            avg_drop = np.mean(recent_drops)
            avg_sad = np.mean(recent_sad)
            max_sad = max(recent_sad)
            
            # Count frames with downturned corners (lowered threshold)
            downturned_count = sum(1 for drop in recent_drops if drop > 0.008)  # Lowered from 0.012
            
            # Get recent EAR values
            if len(session_state['ear_history']) >= 3:
                recent_ear = list(session_state['ear_history'])[-5:]
                partially_closed_count = sum(1 for ear in recent_ear if ear < 0.22)  # Wider range
                very_closed_count = sum(1 for ear in recent_ear if ear < 0.12)
            else:
                partially_closed_count = 0
                very_closed_count = 0
            
            # Calculate average mouth openness from recent frames
            if len(session_state['mouth_openness_history']) >= 3:
                recent_openness = list(session_state['mouth_openness_history'])[-5:]
                avg_mouth_openness = np.mean([mo for mo in recent_openness if mo < 1.0])  # Filter outliers
            else:
                avg_mouth_openness = mouth_openness
            
            # Determine if silent crying (closed mouth) or regular crying
            is_silent_crying = avg_mouth_openness < 0.12  # Closed or very slightly open
            
            # Strong crying detection: lowered thresholds
            if (avg_drop > 0.010 and  # Lowered from 0.015
                downturned_count >= 3 and  # Lowered from 4
                avg_sad > 0.4):  # Lowered from 0.5
                
                # Calculate intensity based on severity
                intensity = min(1.0, (avg_sad * 0.7 + (avg_drop / 0.03) * 0.3))
                if partially_closed_count >= 2:
                    intensity = min(1.0, intensity + 0.15)  # Boost if eyes partially closed
                if very_closed_count >= 1:
                    intensity = min(1.0, intensity + 0.1)  # Extra boost for very closed eyes
                
                # Silent crying bonus: if mouth is closed, add slight boost
                if is_silent_crying:
                    intensity = min(1.0, intensity + 0.05)
                
                return True, min(1.0, intensity), is_silent_crying
            
            # Moderate crying: even more lenient criteria
            if (avg_drop > 0.006 and  # Lowered from 0.010
                downturned_count >= 2 and  # Lowered from 3
                avg_sad > 0.35):  # Lowered from 0.4
                intensity = min(0.85, avg_sad * 0.85)
                if partially_closed_count >= 1:
                    intensity = min(0.9, intensity + 0.1)
                
                # Update silent crying status
                if not is_silent_crying:
                    is_silent_crying = avg_mouth_openness < 0.12
                if is_silent_crying:
                    intensity = min(0.9, intensity + 0.05)
                
                return True, intensity, is_silent_crying
            
            # Light crying: very lenient for subtle crying
            if (avg_drop > 0.005 and
                downturned_count >= 2 and
                avg_sad > 0.3 and
                mouth_width_ratio < 0.43):
                intensity = min(0.7, avg_sad * 0.7)
                
                # Update silent crying status
                if not is_silent_crying:
                    is_silent_crying = avg_mouth_openness < 0.12 if len(session_state['mouth_openness_history']) >= 3 else (mouth_openness < 0.12)
                if is_silent_crying:
                    intensity = min(0.75, intensity + 0.05)
                
                return True, intensity, is_silent_crying
    
    return False, 0, False

def detect_speaking(landmarks, image_width, image_height, session_state):
    """Detect speaking by tracking mouth movement patterns - made stricter to reduce false positives"""
    
    top_lip = landmarks.landmark[TOP_LIP]
    bottom_lip = landmarks.landmark[BOTTOM_LIP]
    left_mouth = landmarks.landmark[LEFT_MOUTH_CORNER]
    right_mouth = landmarks.landmark[RIGHT_MOUTH_CORNER]
    
    # Calculate mouth openness
    mouth_height = calculate_distance(top_lip, bottom_lip, image_width, image_height)
    mouth_width = calculate_distance(left_mouth, right_mouth, image_width, image_height)
    
    if mouth_width > 0:
        mouth_openness = mouth_height / mouth_width
        session_state['mouth_openness_history'].append(mouth_openness)
        
        face_width = calculate_distance(
            landmarks.landmark[LEFT_EYE_OUTER],
            landmarks.landmark[RIGHT_EYE_OUTER],
            image_width, image_height
        )
        
        if face_width > 0:
            mouth_width_ratio = mouth_width / face_width
            
            # Speaking is characterized by frequent, irregular mouth movements
            # Require more frames and stronger patterns than before
            
            if len(session_state['mouth_openness_history']) >= 8:
                # Use recent frames for analysis (more reliable than all history)
                recent_openness = list(session_state['mouth_openness_history'])[-8:]
                variance = np.var(recent_openness)
                avg_openness = np.mean(recent_openness)
                
                # Calculate movement pattern (speaking has more irregular, frequent changes)
                diffs = [recent_openness[i+1] - recent_openness[i] for i in range(len(recent_openness)-1)]
                # Count sign changes (indicates opening/closing cycles)
                sign_changes = sum(1 for i in range(len(diffs)-1) if (diffs[i] > 0) != (diffs[i+1] > 0))
                
                # Calculate range (max - min) to ensure there's actual movement
                openness_range = max(recent_openness) - min(recent_openness)
                
                # Speaking indicators (made stricter):
                # 1. Moderate mouth openness (0.10-0.22) - narrower range
                # 2. Higher variance than eating/drinking (more irregular) - require > 0.001
                # 3. Multiple sign changes (frequent opening/closing) - require >= 3
                # 4. Significant range (actual movement) - require > 0.02
                # 5. Normal mouth width (not too narrow/wide)
                
                # Stricter thresholds:
                if (0.10 < avg_openness < 0.22 and  # Narrower openness range
                    variance > 0.001 and  # Higher variance required (2x previous)
                    0.40 < mouth_width_ratio < 0.55 and  # Normal mouth width
                    sign_changes >= 3 and  # At least 3 opening/closing cycles
                    openness_range > 0.02):  # Significant actual movement
                    
                    # Calculate intensity based on multiple factors (cap it lower)
                    # Speaking intensity should be based on variance and movement pattern
                    variance_component = min(0.5, variance * 400)  # Cap variance component
                    movement_component = min(0.4, (sign_changes / 6.0) * 0.4)  # Normalize sign changes
                    range_component = min(0.1, (openness_range / 0.10) * 0.1)  # Normalize range
                    
                    intensity = min(0.85, variance_component + movement_component + range_component)
                    return True, float(intensity)
            
            # More strict fallback: require at least 6 frames with strong patterns
            if len(session_state['mouth_openness_history']) >= 6:
                recent_openness = list(session_state['mouth_openness_history'])[-6:]
                variance = np.var(recent_openness)
                avg_openness = np.mean(recent_openness)
                openness_range = max(recent_openness) - min(recent_openness)
                
                # Stricter fallback criteria
                if (0.12 < avg_openness < 0.20 and
                    variance > 0.0012 and
                    0.40 < mouth_width_ratio < 0.55 and
                    openness_range > 0.025):
                    # Lower intensity for fallback
                    intensity = min(0.65, variance * 350)
                    return True, float(intensity)
    
    return False, 0

def detect_kissing(landmarks, image_width, image_height, session_state):
    """Detect kissing: lips pursed forward (distance-normalized)"""
    top_lip = landmarks.landmark[TOP_LIP]
    bottom_lip = landmarks.landmark[BOTTOM_LIP]
    left_mouth = landmarks.landmark[LEFT_MOUTH_CORNER]
    right_mouth = landmarks.landmark[RIGHT_MOUTH_CORNER]
    nose_tip = landmarks.landmark[NOSE_TIP]
    
    mouth_width = calculate_distance(left_mouth, right_mouth, image_width, image_height)
    mouth_height = calculate_distance(top_lip, bottom_lip, image_width, image_height)
    
    # Normalize by face width for distance invariance
    left_eye_outer = landmarks.landmark[LEFT_EYE_OUTER]
    right_eye_outer = landmarks.landmark[RIGHT_EYE_OUTER]
    face_width = calculate_distance(left_eye_outer, right_eye_outer, image_width, image_height)
    
    # Kissing: narrow mouth, lips forward (z-coordinate would help but using x,y approximation)
    if mouth_width > 0 and face_width > 0:
        mouth_ratio = mouth_height / mouth_width
        # Normalize mouth width by face width (distance-invariant)
        mouth_width_ratio = mouth_width / face_width
        # Pursed lips: taller than wide, but not too open, and narrow relative to face
        if 0.3 < mouth_ratio < 0.6 and mouth_width_ratio < 0.35:  # Narrow pursed mouth (normalized)
            return True, mouth_ratio
    return False, 0

def detect_eating(landmarks, image_width, image_height, session_state):
    """Detect eating: repetitive mouth movements (chewing pattern)"""
    
    top_lip = landmarks.landmark[TOP_LIP]
    bottom_lip = landmarks.landmark[BOTTOM_LIP]
    left_mouth = landmarks.landmark[LEFT_MOUTH_CORNER]
    right_mouth = landmarks.landmark[RIGHT_MOUTH_CORNER]
    
    # Calculate mouth features
    mouth_height = calculate_distance(top_lip, bottom_lip, image_width, image_height)
    mouth_width = calculate_distance(left_mouth, right_mouth, image_width, image_height)
    
    if mouth_width > 0:
        mouth_openness = mouth_height / mouth_width
        session_state['mouth_openness_history'].append(mouth_openness)
        
        face_width = calculate_distance(
            landmarks.landmark[LEFT_EYE_OUTER],
            landmarks.landmark[RIGHT_EYE_OUTER],
            image_width, image_height
        )
        
        if face_width > 0:
            mouth_width_ratio = mouth_width / face_width
            
            # Eating indicators:
            # 1. Repetitive mouth opening/closing pattern (chewing)
            # 2. Moderate mouth openness (0.10-0.25)
            # 3. Regular mouth width (not too narrow/wide)
            # 4. Lower variance than speaking (more rhythmic)
            
            if len(session_state['mouth_openness_history']) >= 8:
                recent_openness = list(session_state['mouth_openness_history'])[-8:]
                avg_openness = np.mean(recent_openness)
                variance = np.var(recent_openness)
                
                # Check for rhythmic pattern (eating has more regular pattern than speaking)
                # Calculate periodicity - eating has more consistent cycles
                if len(recent_openness) >= 6:
                    # Look for pattern: open -> close -> open (chewing cycle)
                    diffs = [recent_openness[i+1] - recent_openness[i] for i in range(len(recent_openness)-1)]
                    # Count sign changes (indicates opening/closing cycles)
                    sign_changes = sum(1 for i in range(len(diffs)-1) if (diffs[i] > 0) != (diffs[i+1] > 0))
                    
                    # Eating: moderate openness, some variance (movement), rhythmic pattern
                    if (0.10 < avg_openness < 0.25 and
                        0.0003 < variance < 0.002 and  # Less variance than speaking (more rhythmic)
                        0.40 < mouth_width_ratio < 0.55 and
                        sign_changes >= 2):  # At least 2 cycles
                        # Calculate intensity based on movement and pattern
                        intensity = min(1.0, (variance * 500) + (sign_changes / 4.0) * 0.3)
                        return True, intensity
            
            # Fallback: if mouth is moderately open with consistent movement
            if len(session_state['mouth_openness_history']) >= 5:
                recent = list(session_state['mouth_openness_history'])[-5:]
                avg = np.mean(recent)
                if 0.12 < avg < 0.22:
                    return True, 0.6
    
    return False, 0

def detect_drinking(landmarks, image_width, image_height, session_state):
    """Detect drinking: mouth movements with potential head tilt"""
    
    top_lip = landmarks.landmark[TOP_LIP]
    bottom_lip = landmarks.landmark[BOTTOM_LIP]
    left_mouth = landmarks.landmark[LEFT_MOUTH_CORNER]
    right_mouth = landmarks.landmark[RIGHT_MOUTH_CORNER]
    nose_tip = landmarks.landmark[NOSE_TIP]
    
    # Calculate mouth features
    mouth_height = calculate_distance(top_lip, bottom_lip, image_width, image_height)
    mouth_width = calculate_distance(left_mouth, right_mouth, image_width, image_height)
    
    if mouth_width > 0:
        mouth_openness = mouth_height / mouth_width
        session_state['mouth_openness_history'].append(mouth_openness)
        
        face_width = calculate_distance(
            landmarks.landmark[LEFT_EYE_OUTER],
            landmarks.landmark[RIGHT_EYE_OUTER],
            image_width, image_height
        )
        
        if face_width > 0:
            mouth_width_ratio = mouth_width / face_width
            
            # Calculate head position (for tilt detection - drinking often involves head back)
            left_eye = landmarks.landmark[LEFT_EYE_INNER]
            right_eye = landmarks.landmark[RIGHT_EYE_INNER]
            eye_center_y = (left_eye.y + right_eye.y) / 2
            session_state['head_position_history'].append(eye_center_y)
            
            # Drinking indicators:
            # 1. Mouth opening pattern (similar to eating but may be more sustained)
            # 2. Moderate to high mouth openness (0.12-0.30)
            # 3. Possible head tilt (head back)
            # 4. Sustained opening (less variation than eating)
            
            if len(session_state['mouth_openness_history']) >= 6:
                recent_openness = list(session_state['mouth_openness_history'])[-6:]
                avg_openness = np.mean(recent_openness)
                variance = np.var(recent_openness)
                
                # Check for sustained opening (drinking often has longer open phase)
                sustained_open = sum(1 for o in recent_openness if o > 0.15)
                
                # Check head position (drinking may involve head tilt)
                head_tilt = 0.0
                if len(session_state['head_position_history']) >= 3:
                    recent_head = list(session_state['head_position_history'])[-3:]
                    # If head is moving up (y decreasing), might be tilting back
                    if len(recent_head) >= 2:
                        head_movement = recent_head[-1] - recent_head[0]
                        if head_movement < -0.01:  # Head moving up (tilting back)
                            head_tilt = abs(head_movement) * 10
                
                # Drinking: moderate-high openness, sustained opening, possible head tilt
                if (0.12 < avg_openness < 0.30 and
                    0.0002 < variance < 0.0015 and  # Less variation (more sustained)
                    0.40 < mouth_width_ratio < 0.55 and
                    sustained_open >= 3):  # At least 3 frames with open mouth
                    # Calculate intensity
                    intensity = min(1.0, (avg_openness / 0.30) * 0.6 + (sustained_open / 6.0) * 0.3 + head_tilt * 0.1)
                    return True, intensity
            
            # Fallback: sustained moderate opening
            if len(session_state['mouth_openness_history']) >= 4:
                recent = list(session_state['mouth_openness_history'])[-4:]
                avg = np.mean(recent)
                if 0.15 < avg < 0.28:
                    return True, 0.65
    
    return False, 0

def detect_emotions_from_landmarks(landmarks, image_width, image_height, session_state, smile_intensity=None, is_crying=False, cry_intensity=0, is_laughing=False):
    """Accurate emotion detection using MediaPipe landmarks with multi-frame analysis"""
    
    if not landmarks:
        return {'dominant': 'neutral', 'all': {'neutral': 0.5, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'surprise': 0.0, 'fear': 0.0, 'disgust': 0.0}, 'confidence': 0.5}
    
    try:
        left_mouth = landmarks.landmark[LEFT_MOUTH_CORNER]
        right_mouth = landmarks.landmark[RIGHT_MOUTH_CORNER]
        top_lip = landmarks.landmark[TOP_LIP]
        bottom_lip = landmarks.landmark[BOTTOM_LIP]
        left_brow_inner = landmarks.landmark[LEFT_BROW_INNER]
        right_brow_inner = landmarks.landmark[RIGHT_BROW_INNER]
        nose_tip = landmarks.landmark[NOSE_TIP]
        left_eye_top = landmarks.landmark[LEFT_EYE_TOP]
        right_eye_top = landmarks.landmark[RIGHT_EYE_TOP]
        
        # Calculate face dimensions using eye outer corners (more reliable)
        face_width = calculate_distance(
            landmarks.landmark[LEFT_EYE_OUTER],
            landmarks.landmark[RIGHT_EYE_OUTER],
            image_width, image_height
        )
        
        if face_width == 0:
            return {'dominant': 'neutral', 'all': {'neutral': 0.5, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'surprise': 0.0, 'fear': 0.0, 'disgust': 0.0}, 'confidence': 0.5}
        
        # Calculate facial features
        mouth_width = calculate_distance(left_mouth, right_mouth, image_width, image_height)
        mouth_height = calculate_distance(top_lip, bottom_lip, image_width, image_height)
        mouth_width_ratio = mouth_width / face_width
        mouth_height_ratio = mouth_height / face_width if mouth_width > 0 else 0
        mouth_openness = mouth_height / mouth_width if mouth_width > 0 else 0
        
        # Mouth corner position (key for happy/sad)
        mouth_center_y = (top_lip.y + bottom_lip.y) / 2
        mouth_corner_y = (left_mouth.y + right_mouth.y) / 2
        # Positive = corners above center (smiling), Negative = corners below center (sad)
        mouth_corner_raise = (mouth_center_y - mouth_corner_y) * image_height / face_width
        
        # Eyebrow features
        brow_distance = calculate_distance(left_brow_inner, right_brow_inner, image_width, image_height)
        brow_ratio = brow_distance / face_width
        
        # Eyebrow height (furrowed = angry, raised = surprise/fear)
        brow_center_y = (left_brow_inner.y + right_brow_inner.y) / 2
        eye_top_y = (left_eye_top.y + right_eye_top.y) / 2
        brow_to_eye_distance = (brow_center_y - eye_top_y) * image_height / face_width
        
        # Eye aspect ratio for eye state
        left_eye_bottom = landmarks.landmark[LEFT_EYE_BOTTOM]
        right_eye_bottom = landmarks.landmark[RIGHT_EYE_BOTTOM]
        left_eye_outer = landmarks.landmark[LEFT_EYE_OUTER]
        left_eye_inner = landmarks.landmark[LEFT_EYE_INNER]
        right_eye_outer = landmarks.landmark[RIGHT_EYE_OUTER]
        right_eye_inner = landmarks.landmark[RIGHT_EYE_INNER]
        
        left_eye_vertical = calculate_distance(left_eye_top, left_eye_bottom, image_width, image_height)
        left_eye_horizontal = calculate_distance(left_eye_outer, left_eye_inner, image_width, image_height)
        if left_eye_horizontal == 0:
            left_eye_horizontal = 1.0
        left_ear = left_eye_vertical / left_eye_horizontal
        
        right_eye_vertical = calculate_distance(right_eye_top, right_eye_bottom, image_width, image_height)
        right_eye_horizontal = calculate_distance(right_eye_outer, right_eye_inner, image_width, image_height)
        if right_eye_horizontal == 0:
            right_eye_horizontal = 1.0
        right_ear = right_eye_vertical / right_eye_horizontal
        avg_ear = (left_ear + right_ear) / 2
        
        # Initialize emotion scores
        emotions = {
            'neutral': 0.2,
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'surprise': 0.0,
            'fear': 0.0,
            'disgust': 0.0
        }
        
        # HAPPY detection - use smile intensity if available, otherwise calculate
        if smile_intensity is not None:
            # Only map to happy if smile intensity is significant (above threshold)
            # This prevents neutral faces from being classified as happy
            if smile_intensity > 0.25:  # Require at least 25% smile intensity
                emotions['happy'] = smile_intensity * 0.9  # Direct mapping from smile intensity
            else:
                emotions['happy'] = 0.0  # Too weak to be considered happy
        else:
            # Calculate happy from mouth features
            if mouth_width_ratio > 0.45:  # Wide mouth
                happy_score = (mouth_width_ratio - 0.45) / 0.20  # Scale 0.45-0.65 to 0-1
                if mouth_corner_raise > 0.010:  # Corners raised
                    happy_score += min(0.3, mouth_corner_raise * 15)
                emotions['happy'] = min(0.95, happy_score)
        
        if is_laughing:
            emotions['happy'] = max(emotions['happy'], 0.85)  # Laughing = very happy
        
        # SAD detection
        if is_crying:
            emotions['sad'] = 0.7 + (cry_intensity * 0.3)  # Crying = strong sad
        else:
            # Calculate sad from facial features
            sad_score = 0.0
            if mouth_width_ratio < 0.42:  # Narrow mouth
                sad_score += (0.42 - mouth_width_ratio) / 0.15 * 0.4
            if mouth_corner_raise < -0.010:  # Downturned corners
                sad_score += min(0.4, abs(mouth_corner_raise) * 20)
            if 0.10 < avg_ear < 0.18:  # Partially closed eyes
                sad_score += 0.2
            emotions['sad'] = min(0.95, sad_score)
        
        # ANGRY detection - requires strong, combined signals
        angry_score = 0.0
        
        # Strong angry indicators (all must be present for high confidence):
        # 1. Significantly furrowed brows (brow_ratio < 0.30, not just < 0.34)
        # 2. Brows significantly lowered (brow_to_eye_distance < 0.030)
        # 3. Tight/pressed mouth (mouth_width_ratio < 0.38)
        # 4. Downturned mouth corners (mouth_corner_raise < -0.010)
        # 5. NOT smiling (mouth_corner_raise should be negative)
        # 6. Eyes not wide open (avg_ear < 0.25)
        
        # Check for strong angry signals
        strong_furrowed = brow_ratio < 0.30  # Significantly furrowed
        moderate_furrowed = brow_ratio < 0.32  # Moderately furrowed
        brows_lowered = brow_to_eye_distance < 0.030  # Brows significantly lowered
        brows_slightly_lowered = brow_to_eye_distance < 0.033  # Brows slightly lowered
        tight_mouth = mouth_width_ratio < 0.38  # Tight mouth
        downturned_corners = mouth_corner_raise < -0.010  # Clearly downturned
        not_smiling = mouth_corner_raise < 0.0  # Not smiling
        eyes_normal = avg_ear < 0.25  # Eyes not wide open (not surprised)
        
        # Require multiple strong indicators for angry
        strong_indicators = 0
        if strong_furrowed:
            strong_indicators += 1
            angry_score += 0.3
        elif moderate_furrowed:
            strong_indicators += 0.5
            angry_score += 0.15
        
        if brows_lowered:
            strong_indicators += 1
            angry_score += 0.25
        elif brows_slightly_lowered:
            strong_indicators += 0.5
            angry_score += 0.1
        
        if tight_mouth:
            strong_indicators += 1
            angry_score += 0.2
        elif mouth_width_ratio < 0.40:
            strong_indicators += 0.5
            angry_score += 0.1
        
        if downturned_corners:
            strong_indicators += 1
            angry_score += 0.2
        elif mouth_corner_raise < -0.005:
            strong_indicators += 0.5
            angry_score += 0.05
        
        # Negative checks - if these are present, reduce angry score
        if mouth_corner_raise > 0.005:  # Smiling - can't be angry
            angry_score *= 0.3
        if avg_ear > 0.28:  # Wide eyes - more likely surprised than angry
            angry_score *= 0.5
        if mouth_openness > 0.12:  # Open mouth - less likely angry
            angry_score *= 0.6
        
        # Require at least 2.5 strong indicators for any angry detection
        if strong_indicators < 2.5:
            angry_score *= 0.4  # Reduce score if not enough indicators
        
        # Cap the score and ensure it's reasonable
        emotions['angry'] = min(0.85, max(0.0, angry_score))
        
        # SURPRISE detection
        surprise_score = 0.0
        if brow_ratio > 0.38:  # Raised brows
            surprise_score += (brow_ratio - 0.38) / 0.10 * 0.4
        if brow_to_eye_distance > 0.045:  # Brows raised high
            surprise_score += (brow_to_eye_distance - 0.045) / 0.025 * 0.3
        if mouth_openness > 0.15:  # Open mouth
            surprise_score += min(0.3, (mouth_openness - 0.15) / 0.20 * 0.3)
        if avg_ear > 0.25:  # Wide eyes
            surprise_score += min(0.2, (avg_ear - 0.25) / 0.15 * 0.2)
        emotions['surprise'] = min(0.95, surprise_score)
        
        # FEAR detection (similar to surprise but with tension)
        fear_score = 0.0
        if brow_ratio > 0.36:  # Raised brows (like surprise)
            fear_score += (brow_ratio - 0.36) / 0.12 * 0.3
        if mouth_width_ratio > 0.46:  # Wide mouth (but not as wide as surprise)
            fear_score += (mouth_width_ratio - 0.46) / 0.15 * 0.2
        if avg_ear > 0.23:  # Wide eyes
            fear_score += min(0.3, (avg_ear - 0.23) / 0.17 * 0.3)
        if mouth_openness > 0.12 and mouth_openness < 0.25:  # Moderately open
            fear_score += 0.2
        emotions['fear'] = min(0.90, fear_score)
        
        # DISGUST detection - made more strict to reduce false positives
        disgust_score = 0.0
        # Require multiple strong indicators for disgust (it's often confused with other emotions)
        disgust_indicators = 0
        
        # 1. Slightly furrowed brows (more specific range)
        if 0.32 < brow_ratio < 0.35:  # Narrower range, not too close to angry
            disgust_score += 0.15
            disgust_indicators += 0.5
        
        # 2. Tight/narrow mouth (but not too tight - that's angry)
        if 0.38 < mouth_width_ratio < 0.42:  # Narrow range
            disgust_score += 0.2
            disgust_indicators += 1
        
        # 3. Brows slightly lowered (specific range)
        if 0.037 < brow_to_eye_distance < 0.040:  # Narrow range
            disgust_score += 0.15
            disgust_indicators += 0.5
        
        # 4. Compressed mouth (key feature of disgust)
        if mouth_height_ratio < 0.025:  # More strict - very compressed
            disgust_score += 0.25
            disgust_indicators += 1
        
        # 5. Nose wrinkle approximation (using brow position + compressed mouth)
        if (0.32 < brow_ratio < 0.35 and 
            mouth_width_ratio < 0.42 and 
            mouth_height_ratio < 0.028):
            disgust_score += 0.15
            disgust_indicators += 0.5
        
        # Require at least 2 strong indicators for any disgust score
        if disgust_indicators < 2.0:
            disgust_score *= 0.3  # Heavily reduce if not enough indicators
        
        # Additional negative checks - reduce disgust if other emotions are strong
        if mouth_corner_raise > 0.005:  # Smiling = not disgust
            disgust_score *= 0.2
        if avg_ear > 0.28:  # Wide eyes = more likely surprise
            disgust_score *= 0.3
        if mouth_openness > 0.10:  # Open mouth = less likely disgust
            disgust_score *= 0.4
        
        emotions['disgust'] = min(0.75, max(0.0, disgust_score))  # Cap at 0.75 instead of 0.85
        
        # CONFLICT RESOLUTION: Prevent confusion between similar expressions
        # These emotions share similar landmarks, so we need to distinguish them clearly
        
        # 1. HAPPY vs SURPRISE (both can have raised features)
        # Happy: corners up (smile), Surprise: open mouth, wide eyes
        if emotions['happy'] > 0.4 and emotions['surprise'] > 0.4:
            if mouth_corner_raise > 0.015 and mouth_openness < 0.15:
                # Strong smile with closed mouth = happy, reduce surprise
                emotions['surprise'] *= 0.4
            elif mouth_openness > 0.18 and avg_ear > 0.28:
                # Very open mouth with wide eyes = surprise, reduce happy
                emotions['happy'] *= 0.5
            elif emotions['happy'] > emotions['surprise'] * 1.2:
                # Happy clearly stronger
                emotions['surprise'] *= 0.5
            else:
                # Surprise clearly stronger
                emotions['happy'] *= 0.5
        
        # 2. SAD vs ANGRY (both have downturned mouth, furrowed brows)
        # Angry: very furrowed brows, tight mouth. Sad: softer features, eyes partially closed
        if emotions['sad'] > 0.3 and emotions['angry'] > 0.3:
            if brow_ratio < 0.30 and brow_to_eye_distance < 0.030:
                # Very furrowed brows = angry (stronger signal)
                emotions['sad'] *= 0.5
            elif avg_ear < 0.18 and mouth_corner_raise < -0.010 and brow_ratio > 0.32:
                # Eyes partially closed + soft brows = sad
                emotions['angry'] *= 0.5
            elif emotions['angry'] > emotions['sad'] * 1.3:
                # Angry clearly stronger (more indicators)
                emotions['sad'] *= 0.4
            else:
                # Sad clearly stronger
                emotions['angry'] *= 0.4
        
        # 3. FEAR vs SURPRISE (both have wide eyes, raised brows)
        # Surprise: very open mouth, brows very high. Fear: tense mouth, moderate opening
        if emotions['fear'] > 0.3 and emotions['surprise'] > 0.3:
            if mouth_openness > 0.20 and brow_to_eye_distance > 0.048:
                # Very open mouth + very raised brows = surprise
                emotions['fear'] *= 0.4
            elif mouth_openness < 0.20 and mouth_openness > 0.12:
                # Moderate mouth opening with tension = fear
                emotions['surprise'] *= 0.5
            elif emotions['surprise'] > emotions['fear'] * 1.2:
                # Surprise clearly stronger
                emotions['fear'] *= 0.5
            else:
                # Fear clearly stronger
                emotions['surprise'] *= 0.5
        
        # 4. DISGUST vs ANGRY (both have furrowed brows, tight mouth)
        # Disgust: slightly furrowed, compressed mouth. Angry: strongly furrowed, tight/pressed mouth
        if emotions['disgust'] > 0.3 and emotions['angry'] > 0.3:
            if brow_ratio < 0.30 and brow_to_eye_distance < 0.030:
                # Very furrowed = angry (stronger signal)
                emotions['disgust'] *= 0.5
            elif 0.32 < brow_ratio < 0.36 and mouth_height_ratio < 0.030:
                # Slightly furrowed + compressed mouth = disgust
                emotions['angry'] *= 0.6
            elif emotions['angry'] > emotions['disgust'] * 1.2:
                # Angry clearly stronger
                emotions['disgust'] *= 0.5
            else:
                # Disgust clearly stronger
                emotions['angry'] *= 0.5
        
        # 5. SAD vs CRYING (crying is sad but more intense with eyes closed)
        # Crying already sets sad score high, but if crying detected, prioritize it
        if is_crying:
            # Crying detected = strong sad, reduce other negative emotions slightly
            if emotions['angry'] > 0.3:
                emotions['angry'] *= 0.6  # Can't be angry while crying
            if emotions['disgust'] > 0.3:
                emotions['disgust'] *= 0.7
        
        # 6. HAPPY vs SAD/ANGRY (mutually exclusive - strong smile reduces negative emotions)
        if emotions['happy'] > 0.5:
            # Strong happy signal = reduce negative emotions
            if emotions['sad'] > 0.2:
                emotions['sad'] *= 0.3
            if emotions['angry'] > 0.2:
                emotions['angry'] *= 0.2
            if emotions['disgust'] > 0.2:
                emotions['disgust'] *= 0.4
        
        # 7. SURPRISE vs SAD/ANGRY (surprise has wide eyes, negative emotions have closed/narrow eyes)
        if emotions['surprise'] > 0.5:
            if avg_ear > 0.28:  # Wide eyes = surprise
                if emotions['sad'] > 0.2:
                    emotions['sad'] *= 0.4  # Sad has closed eyes
                if emotions['angry'] > 0.2:
                    emotions['angry'] *= 0.3  # Angry has narrowed eyes
        
        # Normalize scores to ensure one is dominant
        max_score = max(emotions.values())
        if max_score < 0.3:
            # All scores low, default to neutral (boost neutral for true neutral faces)
            emotions['neutral'] = 0.6  # Increased from 0.5 to ensure neutral wins when no strong emotions
        else:
            # Boost the highest score slightly, but only if it's clearly dominant
            for emotion in emotions:
                if emotions[emotion] == max_score:
                    # Only boost if second highest is significantly lower (clear winner)
                    sorted_scores = sorted(emotions.values(), reverse=True)
                    if len(sorted_scores) > 1 and sorted_scores[0] > sorted_scores[1] * 1.3:
                        emotions[emotion] = min(0.95, emotions[emotion] * 1.1)
                    break  # Exit loop after boosting the max emotion

        # Optimized: Use exponential moving average (EMA) for faster, smoother updates
        session_state['emotion_history'].append(emotions.copy())
        if len(session_state['emotion_history']) >= 3:
            # Use EMA with alpha=0.35 for balanced response (was averaging over 7 frames)
            # EMA: new_value = alpha * current + (1-alpha) * previous
            alpha = 0.35  # Balanced: responsive but stable
            last_smoothed = session_state.get('last_smoothed_emotions')
            
            if last_smoothed is None or len(session_state['emotion_history']) <= 5:
                # First few frames: use simple average for initialization
                smoothing_frames = list(session_state['emotion_history'])[-min(5, len(session_state['emotion_history'])):]
                smoothed_emotions = {
                    'neutral': 0.0,
                    'happy': 0.0,
                    'sad': 0.0,
                    'angry': 0.0,
                    'surprise': 0.0,
                    'fear': 0.0,
                    'disgust': 0.0
                }
                for frame_emotions in smoothing_frames:
                    for emotion in smoothed_emotions:
                        smoothed_emotions[emotion] += frame_emotions.get(emotion, 0.0)
                for emotion in smoothed_emotions:
                    smoothed_emotions[emotion] /= len(smoothing_frames)
                emotions = smoothed_emotions
                session_state['last_smoothed_emotions'] = emotions.copy()
            else:
                # EMA smoothing: faster response with proper history
                smoothed_emotions = {}
                for emotion in emotions:
                    prev_value = last_smoothed.get(emotion, emotions[emotion])
                    smoothed_emotions[emotion] = alpha * emotions[emotion] + (1 - alpha) * prev_value
                emotions = smoothed_emotions
                session_state['last_smoothed_emotions'] = emotions.copy()
        
        # Find current dominant emotion from smoothed scores
        current_dominant = max(emotions, key=emotions.get)
        current_confidence = emotions[current_dominant]
        
        # Special handling: If happy score is very low (weak smile) and other emotions are also weak,
        # boost neutral to ensure neutral faces are detected correctly
        if emotions['happy'] < 0.2 and current_confidence < 0.35:
            # Weak emotions overall - likely neutral face
            if emotions['neutral'] < 0.4:
                emotions['neutral'] = 0.5  # Boost neutral
            current_dominant = 'neutral'
            current_confidence = emotions['neutral']
        
        # Ensure confidence is reasonable and stable
        if current_confidence < 0.25:
            # If confidence is too low, check if we have a stable emotion with better confidence
            if session_state.get('stable_dominant_emotion') and session_state['stable_dominant_emotion'] != 'neutral':
                stable_confidence = emotions.get(session_state['stable_dominant_emotion'], 0.0)
                if stable_confidence > 0.25:
                    current_dominant = session_state['stable_dominant_emotion']
                    current_confidence = stable_confidence
                else:
                    current_dominant = 'neutral'
                    current_confidence = 0.5
            else:
                current_dominant = 'neutral'
                current_confidence = 0.5
        
        # STABLE EMOTION TRACKING: Prevent rapid changes
        session_state['stable_emotion_history'].append(current_dominant)
        
        # Count how many of the last frames had the same dominant emotion
        if len(session_state['stable_emotion_history']) >= MIN_STABLE_FRAMES:
            recent_dominants = list(session_state['stable_emotion_history'])[-MIN_STABLE_FRAMES:]
            # Count occurrences of current dominant in recent frames
            current_count = recent_dominants.count(current_dominant)
            
            # Only change dominant if:
            # 1. Current emotion appears in at least 60% of recent frames (3 out of 5)
            # 2. AND it's different from current stable emotion
            # 3. AND it has significantly higher confidence (at least 0.15 higher)
            
            if (current_dominant != session_state['stable_dominant_emotion'] and 
                current_count >= (MIN_STABLE_FRAMES * 0.6) and
                current_confidence > 0.35):  # Require minimum confidence
                
                # Check if new emotion is significantly better than stable one
                stable_confidence = emotions.get(session_state['stable_dominant_emotion'], 0.0)
                confidence_gap = current_confidence - stable_confidence
                
                # Require at least 0.15 confidence gap to switch
                if confidence_gap >= 0.15:
                    session_state['stable_dominant_emotion'] = current_dominant
                    session_state['stable_emotion_count'] = current_count
                else:
                    # Keep stable emotion, but update count if same emotion continues
                    if current_dominant == session_state['stable_dominant_emotion']:
                        session_state['stable_emotion_count'] = current_count
                    # Otherwise, don't change (not enough gap)
            elif current_dominant == session_state['stable_dominant_emotion']:
                # Same emotion continues, update count
                session_state['stable_emotion_count'] = current_count
            # If conditions not met, keep previous stable emotion
        
        # Use stable dominant emotion for output
        dominant = session_state['stable_dominant_emotion']
        confidence = emotions[dominant]
        
        # If stable emotion has very low confidence but current has high confidence, allow change
        if confidence < 0.20 and current_confidence > 0.50:
            dominant = current_dominant
            confidence = current_confidence
            session_state['stable_dominant_emotion'] = dominant
        
        return {
            'dominant': dominant,
            'all': emotions,
            'confidence': confidence
        }
    except Exception as e:
        print(f"Error in landmark-based emotion detection: {e}")
        import traceback
        traceback.print_exc()
        return {'dominant': 'neutral', 'all': {'neutral': 0.5, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'surprise': 0.0, 'fear': 0.0, 'disgust': 0.0}, 'confidence': 0.5}

def detect_comprehensive_expressions(frame_rgb, landmarks, image_width, image_height, session_state):
    """Detect all expressions and emotions"""
    results = {
        'emotions': {},
        'expressions': {},
        'features': {}
    }
    
    # 1. Use FER for basic emotions (happy, sad, angry, surprise, fear, disgust, neutral)
    if fer_detector:
        try:
            fer_emotions = fer_detector.detect_emotions(frame_rgb)
            if fer_emotions and len(fer_emotions) > 0:
                # Get top emotion from FER
                top_emotion_data = fer_emotions[0].get('emotions', {})
                if top_emotion_data:
                    # Get dominant emotion
                    dominant_emotion = max(top_emotion_data, key=top_emotion_data.get)
                    results['emotions'] = {
                        'dominant': dominant_emotion,
                        'all': top_emotion_data,
                        'confidence': top_emotion_data[dominant_emotion]
                    }
        except Exception as e:
            print(f"FER detection error: {e}")
            # Fallback to MediaPipe-based emotion detection
            results['emotions'] = detect_emotions_from_landmarks(landmarks, image_width, image_height, session_state)
    else:
        # Use MediaPipe-based emotion detection as fallback
        results['emotions'] = detect_emotions_from_landmarks(landmarks, image_width, image_height, session_state)
    
    # 2. Use MediaPipe for additional features
    if landmarks:
        # Smile intensity (needed for emotion detection)
        smile_intensity = detect_smile_intensity(landmarks, image_width, image_height, session_state)
        
        # Laughing (needed for emotion detection) - pass smile_intensity to ensure laughing > smile
        is_laughing, laughing_intensity = detect_laughing(landmarks, image_width, image_height, session_state, smile_intensity=smile_intensity)
        
        # If laughing is detected, reduce smile intensity to be less than laughing intensity
        if is_laughing and laughing_intensity > smile_intensity:
            smile_intensity = max(0.0, laughing_intensity - 0.15)  # Smile intensity is 0.15 less than laughing
        
        results['features']['smile_intensity'] = float(smile_intensity)
        results['expressions']['laughing'] = bool(is_laughing)
        results['features']['laughing_intensity'] = float(laughing_intensity)
        
        # Crying (needed for emotion detection) - now detects both regular and silent crying
        is_crying, cry_score, is_silent_crying = detect_crying(landmarks, image_width, image_height, session_state)
        results['expressions']['crying'] = bool(is_crying)  # Ensure Python bool
        results['features']['crying_intensity'] = float(cry_score)  # Ensure Python float
        results['expressions']['silent_crying'] = bool(is_silent_crying)  # Ensure Python bool
        results['features']['silent_crying_intensity'] = float(cry_score if is_silent_crying else 0.0)
        
        # Update emotion detection with expression information
        if not fer_detector:
            results['emotions'] = detect_emotions_from_landmarks(
                landmarks, image_width, image_height, session_state,
                smile_intensity=smile_intensity,
                is_crying=is_crying,
                cry_intensity=cry_score,
                is_laughing=is_laughing
            )
        
        # Yawning
        is_yawning, yawn_score = detect_yawning(landmarks, image_width, image_height, session_state)
        results['expressions']['yawning'] = bool(is_yawning)
        results['features']['yawning_intensity'] = float(yawn_score)
        
        # Eye blinking
        is_blinking, ear = detect_eye_blink(landmarks, image_width, image_height, session_state)
        results['expressions']['eye_blinking'] = bool(is_blinking)
        results['features']['eye_aspect_ratio'] = float(ear)
        results['features']['blink_count'] = int(session_state['blink_count'])
        
        # Sleepy (multi-frame)
        is_sleepy, sleepy_intensity = detect_sleepy(landmarks, image_width, image_height, session_state)
        results['expressions']['sleepy'] = bool(is_sleepy)
        results['features']['sleepy_intensity'] = float(sleepy_intensity)
        
        # Head shake
        is_shaking, shake_variance = detect_head_shake(landmarks, image_width, image_height, session_state)
        results['expressions']['head_shake'] = bool(is_shaking)
        results['features']['head_shake_intensity'] = float(shake_variance)
        
        # Attention
        is_attentive, attention_score = detect_attention(landmarks, image_width, image_height, session_state)
        results['expressions']['attention'] = bool(is_attentive)
        results['features']['attention_score'] = float(attention_score)
        
        # Speaking
        is_speaking, speaking_intensity = detect_speaking(landmarks, image_width, image_height, session_state)
        results['expressions']['speaking'] = bool(is_speaking)
        results['features']['speaking_intensity'] = float(speaking_intensity)
        
        # Kissing
        is_kissing, kiss_score = detect_kissing(landmarks, image_width, image_height, session_state)
        results['expressions']['kissing'] = bool(is_kissing)
        results['features']['kissing_intensity'] = float(kiss_score)
        
        # Eating
        is_eating, eating_intensity = detect_eating(landmarks, image_width, image_height, session_state)
        results['expressions']['eating'] = bool(is_eating)
        results['features']['eating_intensity'] = float(eating_intensity)
        
        # Drinking
        is_drinking, drinking_intensity = detect_drinking(landmarks, image_width, image_height, session_state)
        results['expressions']['drinking'] = bool(is_drinking)
        results['features']['drinking_intensity'] = float(drinking_intensity)
        
        # Eyebrow raised
        is_eyebrow_raised, eyebrow_raised_intensity = detect_eyebrow_raised(landmarks, image_width, image_height, session_state)
        results['expressions']['eyebrow_raised'] = bool(is_eyebrow_raised)
        results['features']['eyebrow_raised_intensity'] = float(eyebrow_raised_intensity)
    
    # Calculate dominant expression/action
    dominant_expression_data = calculate_dominant_expression(
        results.get('expressions', {}),
        results.get('features', {}),
        session_state
    )
    results['dominant_expression'] = dominant_expression_data
    
    return results

def calculate_dominant_expression(expressions, features, session_state):
    """Calculate the dominant expression/action based on detected expressions and intensities"""
    
    # Expression priority mapping (higher priority expressions override lower ones)
    # Format: (expression_key, display_name, weight)
    expression_scores = {}
    
    # Priority-based scoring: some expressions are more important/noticeable
    # Higher priority expressions get higher base scores
    
    # Very high priority expressions (strong indicators)
    if expressions.get('laughing', False):
        intensity = features.get('laughing_intensity', 0.5)
        expression_scores['laughing'] = {
            'name': 'Laughing',
            'score': intensity * 1.2,  # Boost laughing
            'intensity': intensity
        }
    
    if expressions.get('crying', False):
        intensity = features.get('crying_intensity', 0.5)
        expression_scores['crying'] = {
            'name': 'Crying',
            'score': intensity * 1.2,  # Boost crying
            'intensity': intensity
        }
    
    if expressions.get('silent_crying', False):
        intensity = features.get('silent_crying_intensity', 0.5)
        expression_scores['silent_crying'] = {
            'name': 'Silent Crying',
            'score': intensity * 1.25,  # Boost silent crying (even more than regular crying)
            'intensity': intensity
        }
    
    # Yawning - check conflicts with speaking first
    if expressions.get('yawning', False):
        intensity = features.get('yawning_intensity', 0.5)
        
        # If speaking is detected, yawning is less likely (they're mutually exclusive)
        if expressions.get('speaking', False):
            # Speaking involves rapid mouth movements, yawning is sustained
            # Reduce yawning score significantly if speaking present
            intensity *= 0.3
            # Only show yawning if intensity is still very high after reduction
            if intensity > 0.6:  # Only show if very strong yawn (not just open mouth from speaking)
                expression_scores['yawning'] = {
                    'name': 'Yawning',
                    'score': intensity * 1.0,
                    'intensity': intensity
                }
        else:
            # No conflict, use normal yawning score
            expression_scores['yawning'] = {
                'name': 'Yawning',
                'score': intensity * 1.0,
                'intensity': intensity
            }
    
    if expressions.get('sleepy', False):
        intensity = features.get('sleepy_intensity', 0.5)
        expression_scores['sleepy'] = {
            'name': 'Sleepy',
            'score': intensity * 1.0,
            'intensity': intensity
        }
    
    if expressions.get('kissing', False):
        intensity = features.get('kissing_intensity', 0.5)
        expression_scores['kissing'] = {
            'name': 'Kissing',
            'score': intensity * 1.1,
            'intensity': intensity
        }
    
    # Medium priority expressions
    # Speaking detection - check conflicts with eating/drinking first
    if expressions.get('speaking', False):
        # Reduce speaking score if eating or drinking is also detected (they take priority)
        intensity = features.get('speaking_intensity', 0.5)
        
        # If eating or drinking detected, speaking is less likely
        if expressions.get('eating', False) or expressions.get('drinking', False):
            # Reduce speaking score significantly if eating/drinking present
            intensity *= 0.4
            # Only show speaking if intensity is still significant after reduction
            if intensity > 0.25:
                expression_scores['speaking'] = {
                    'name': 'Speaking',
                    'score': intensity * 0.9,
                    'intensity': intensity
                }
        else:
            # No conflict, use normal speaking score
            expression_scores['speaking'] = {
                'name': 'Speaking',
                'score': intensity * 0.9,
                'intensity': intensity
            }
    
    if expressions.get('head_shake', False):
        intensity = features.get('head_shake_intensity', 0.5)
        expression_scores['head_shake'] = {
            'name': 'Head Shaking',
            'score': intensity * 0.85,
            'intensity': intensity
        }
    
    # Lower priority but still significant
    smile_intensity = features.get('smile_intensity', 0.0)
    if smile_intensity > 0.3:  # Only consider significant smiles
        expression_scores['smiling'] = {
            'name': 'Smiling',
            'score': smile_intensity * 0.8,
            'intensity': smile_intensity
        }
    
    # Note: Attention is excluded from dominant action calculation
    
    # Eye blinking is very common, lower priority
    if expressions.get('eye_blinking', False):
        blink_count = features.get('blink_count', 0)
        # Normalize blink count (assuming ~15-20 blinks per minute is normal)
        expression_scores['eye_blinking'] = {
            'name': 'Blinking',
            'score': 0.3,  # Fixed low score since it's very common
            'intensity': min(1.0, blink_count / 20.0)
        }
    
    # Find dominant expression
    if not expression_scores:
        current_dominant = 'none'
        current_score = 0.0
        current_intensity = 0.0
    else:
        # Sort by score
        sorted_expressions = sorted(expression_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        current_dominant_key, current_dominant_data = sorted_expressions[0]
        current_dominant = current_dominant_data['name']
        current_score = current_dominant_data['score']
        current_intensity = current_dominant_data.get('intensity', 0.0)
    
    # Store in history for smoothing
    session_state['expression_history'].append({
        'expression': current_dominant,
        'score': current_score,
        'intensity': current_intensity
    })
    
    # STABLE EXPRESSION TRACKING: Prevent rapid changes (similar to emotion)
    session_state['stable_expression_history'].append(current_dominant)
    
    # Optimized: Use fewer frames (5 instead of 7) for faster response
    if len(session_state['expression_history']) >= 3:
        smoothing_frames = list(session_state['expression_history'])[-5:] if len(session_state['expression_history']) >= 5 else list(session_state['expression_history'])
        # Count occurrences of each expression
        expression_counts = {}
        expression_total_scores = {}
        expression_total_intensities = {}
        
        for frame_data in smoothing_frames:
            expr = frame_data['expression']
            expression_counts[expr] = expression_counts.get(expr, 0) + 1
            expression_total_scores[expr] = expression_total_scores.get(expr, 0.0) + frame_data['score']
            expression_total_intensities[expr] = expression_total_intensities.get(expr, 0.0) + frame_data['intensity']
        
        # Find expression that appears most frequently with highest average score
        if expression_counts:
            # Weight by both frequency and average score
            weighted_scores = {}
            smoothing_len = len(smoothing_frames)
            for expr, count in expression_counts.items():
                avg_score = expression_total_scores[expr] / count
                avg_intensity = expression_total_intensities[expr] / count
                # Combine frequency (60%) and score (40%)
                weighted_scores[expr] = {
                    'weight': (count / smoothing_len) * 0.6 + (avg_score / 1.2) * 0.4,
                    'count': count,
                    'score': avg_score,
                    'intensity': avg_intensity
                }
            
            # Get dominant from smoothed data
            smoothed_dominant = max(weighted_scores.items(), key=lambda x: x[1]['weight'])
            current_dominant = smoothed_dominant[0]
            current_score = smoothed_dominant[1]['score']
            current_intensity = smoothed_dominant[1]['intensity']
            current_count = smoothed_dominant[1]['count']
        else:
            current_count = 0
    
    # Apply stability filter
    if len(session_state['stable_expression_history']) >= MIN_STABLE_EXPRESSION_FRAMES:
        recent_expressions = list(session_state['stable_expression_history'])[-MIN_STABLE_EXPRESSION_FRAMES:]
        current_expression_count = recent_expressions.count(current_dominant)
        
        # Only change dominant if:
        # 1. Current expression appears in at least 60% of recent frames
        # 2. AND it's different from current stable expression
        # 3. AND it has reasonable score (>0.3)
        if (current_dominant != session_state['stable_dominant_expression'] and
            current_expression_count >= (MIN_STABLE_EXPRESSION_FRAMES * 0.6) and
            current_score > 0.3):
            
            # Check if new expression is significantly better than stable one
            stable_score = 0.0
            if session_state['stable_dominant_expression'] != 'none':
                # Find stable expression's current score
                for frame_data in session_state['expression_history']:
                    if frame_data['expression'] == session_state['stable_dominant_expression']:
                        stable_score = frame_data['score']
                        break
            
            score_gap = current_score - stable_score
            
            # Require at least 0.2 score gap to switch
            if score_gap >= 0.2:
                session_state['stable_dominant_expression'] = current_dominant
                session_state['stable_expression_count'] = current_expression_count
            else:
                # Keep stable expression
                if current_dominant == session_state['stable_dominant_expression']:
                    session_state['stable_expression_count'] = current_expression_count
        elif current_dominant == session_state['stable_dominant_expression']:
            # Same expression continues
            session_state['stable_expression_count'] = current_expression_count
    
    # Initialize defaults
    dominant = 'none'
    intensity = 0.0
    score = 0.0
    
    # Use current dominant expression for output (more responsive)
    # Only use stable if we have enough history, otherwise use current
    if len(session_state['expression_history']) >= MIN_STABLE_EXPRESSION_FRAMES:
        # Apply stability filter but allow faster updates
        recent_expressions = list(session_state['stable_expression_history'])[-MIN_STABLE_EXPRESSION_FRAMES:]
        current_expression_count = recent_expressions.count(current_dominant)
        
        # More lenient: if current expression appears in at least 40% of recent frames, use it
        # OR if current score is significantly higher (>0.4)
        if (current_expression_count >= (MIN_STABLE_EXPRESSION_FRAMES * 0.4) or current_score > 0.4):
            if current_score > 0.2:  # Minimum threshold
                dominant = current_dominant
                intensity = current_intensity
                score = current_score
                # Update stable if it's consistently appearing
                if current_expression_count >= (MIN_STABLE_EXPRESSION_FRAMES * 0.6):
                    session_state['stable_dominant_expression'] = current_dominant
        else:
            # Use stable if it still has reasonable score
            dominant = session_state['stable_dominant_expression']
            intensity = current_intensity if dominant == current_dominant else 0.0
            score = current_score if dominant == current_dominant else 0.0
            
            # If stable is 'none' or very low, use current anyway
            if dominant == 'none' or score < 0.15:
                if current_score > 0.2:
                    dominant = current_dominant
                    intensity = current_intensity
                    score = current_score
    else:
        # Not enough history yet - use current detection directly
        if current_score > 0.2:  # Minimum threshold
            dominant = current_dominant
            intensity = current_intensity
            score = current_score
        else:
            dominant = 'none'
            intensity = 0.0
            score = 0.0
    
    return {
        'expression': dominant,
        'score': score,
        'intensity': intensity,
        'all_expressions': expression_scores  # For debugging/display
    }

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_bgr
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None

@app.route('/')
def index():
    """Serve the test page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return {'status': 'healthy', 'service': 'emotion-detection'}, 200

@app.errorhandler(500)
def handle_500(error):
    """Handle 500 errors gracefully"""
    print(f"Internal server error: {error}")
    return "Internal Server Error", 500

@socketio.on_error_default
def default_error_handler(e):
    """Handle Socket.IO errors gracefully"""
    error_str = str(e).lower()
    if "too many packets" in error_str or "payload" in error_str:
        # This is a client-side issue (sending too much data too fast)
        # Log but don't crash - client should throttle itself
        print(f"Warning: Client {request.sid} sent payload too large - client should reduce frame rate/size")
    else:
        print(f"Socket.IO error: {e}")
        import traceback
        traceback.print_exc()

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    try:
        session_id = request.sid
        print(f'Client connected: {session_id}')
        print(f'Connection details - Transport: {request.environ.get("HTTP_UPGRADE", "unknown")}')
        print(f'Origin: {request.environ.get("HTTP_ORIGIN", "unknown")}')
        # Session state will be created automatically on first use
        emit('response', {'status': 'connected', 'message': 'Connected to comprehensive emotion detection server'})
        print(f'Sent response to client: {session_id}')
    except Exception as e:
        print(f'Error in handle_connect: {e}')
        import traceback
        traceback.print_exc()
        # Don't emit on error, just log it

@socketio.on('disconnect')
def handle_disconnect(*args, **kwargs):
    """Handle client disconnection - accepts any arguments from Flask-SocketIO"""
    try:
        session_id = request.sid
        print(f'Client disconnected: {session_id}')
        cleanup_session_state(session_id)
    except Exception as e:
        print(f'Error in handle_disconnect: {e}')
        import traceback
        traceback.print_exc()

@socketio.on('webcam_frame')
def handle_frame(data):
    """Handle incoming webcam frame from browser"""
    try:
        # Get session state for this client FIRST (before any processing)
        # This ensures session state is always initialized and isolated per session
        session_id = request.sid
        
        # Validate session_id exists
        if not session_id:
            emit('face_emotion', {
                'emotion': 'error',
                'error': 'Invalid session ID',
                'success': False
            })
            return
        
        # Get or create isolated session state for this specific client
        session_state = get_session_state(session_id)
        
        # Extract base64 image
        image_data = data.get('frame') or data
        if isinstance(image_data, dict):
            image_data = image_data.get('frame', '')
        
        # Validate image data size (prevent oversized payloads)
        if not image_data or len(image_data) < 100:
            emit('face_emotion', {
                'emotion': 'error',
                'error': 'Invalid image data received',
                'success': False
            })
            return
        
        # Convert base64 to OpenCV image
        frame = base64_to_image(image_data)
        if frame is None:
            emit('face_emotion', {
                'emotion': 'error',
                'error': 'Failed to decode image',
                'success': False
            })
            return
        
        # Convert BGR to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        
        # Process with MediaPipe for landmarks
        landmarks_result = None
        try:
            # Get thread-local face_mesh instance to prevent cross-contamination between sessions
            face_mesh = get_face_mesh()
            # Suppress MediaPipe stderr output (timestamp mismatch warnings)
            # These are non-fatal and occur when frames arrive out of order
            stderr_suppressor = StringIO()
            with redirect_stderr(stderr_suppressor):
                mp_results = face_mesh.process(frame_rgb)
            if mp_results and mp_results.multi_face_landmarks:
                landmarks_result = mp_results.multi_face_landmarks[0]
                
                # Validate face quality - reject poor detections
                is_valid, validation_msg = validate_face_quality(landmarks_result, width, height)
                if not is_valid:
                    # Skip this frame - poor quality detection
                    emit('face_emotion', {
                        'success': False,
                        'emotion': 'neutral',
                        'error': f'Poor face detection quality: {validation_msg}',
                        'message': 'Please position your face properly in front of the camera'
                    })
                    return
                
                # Update baseline calibration for adaptive thresholds (per-person normalization)
                update_baseline_calibration(landmarks_result, width, height, session_state)
                
        except (ValueError, RuntimeError) as mp_error:
            # Handle MediaPipe timestamp/processing errors gracefully
            error_str = str(mp_error).lower()
            if "timestamp" in error_str or "calculator" in error_str:
                # These are non-fatal - just skip this frame
                pass
            else:
                # Re-raise other errors
                raise
        
        # Detect all expressions (only if landmarks are valid)
        if landmarks_result:
            detection_results = detect_comprehensive_expressions(
                frame_rgb, landmarks_result, width, height, session_state
            )
        else:
            detection_results = {
                'emotions': {'dominant': 'neutral', 'all': {'neutral': 0.5}, 'confidence': 0.5},
                'expressions': {},
                'features': {}
            }
        
        # Convert NumPy types to native Python types for JSON serialization
        def convert_to_native(obj):
            """Recursively convert NumPy types to native Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        # Format response - include session_id for verification
        response = {
            'success': True,
            'session_id': session_id,  # Include session ID for client verification
            'emotions': convert_to_native(detection_results.get('emotions', {})),
            'expressions': convert_to_native(detection_results.get('expressions', {})),
            'features': convert_to_native(detection_results.get('features', {})),
            'dominant_emotion': detection_results.get('emotions', {}).get('dominant', 'neutral'),
            'dominant_expression': detection_results.get('dominant_expression', {}).get('expression', 'none'),
            'dominant_expression_data': convert_to_native(detection_results.get('dominant_expression', {})),
            'detected_expressions': [k for k, v in detection_results.get('expressions', {}).items() if v]
        }
        
        # Convert detected_expressions list items to native types
        response['detected_expressions'] = [str(k) for k in response['detected_expressions']]
        
        # Emit response to client
        try:
            emit('face_emotion', response)
            # Optimized: removed random debug logging for speed
        except Exception as emit_error:
            print(f'Error emitting response: {emit_error}')
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error processing frame for session {request.sid}: {e}")
        import traceback
        traceback.print_exc()
        try:
            emit('face_emotion', {
                'emotion': 'error',
                'error': str(e),
                'success': False
            })
        except Exception as emit_err:
            print(f"Failed to emit error response: {emit_err}")

if __name__ == '__main__':
    print("Starting Comprehensive Emotion & Expression Detection Server...")
    print("Using MediaPipe for full emotion & expression detection")
    print("Access the web interface at: http://localhost:5000")
    print(f"Using async mode: {async_mode}")
    print(f"Socket.IO configured with transports: websocket, polling")
    if async_mode == 'threading':
        print("Note: For better WebSocket support, install gevent: pip install gevent gevent-websocket")
    # Run with allow_unsafe_werkzeug to avoid write() before start_response errors
    # Use_reloader=False to avoid issues with debugging
    try:
        print("Starting Flask-SocketIO server...")
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        try:
            socketio.stop()
        except:
            pass
else:
    # When running under gunicorn (production)
    print("Flask app loaded under gunicorn with eventlet workers")
    print(f"Socket.IO async mode: {async_mode}")
    print("WebSocket support should be available")
