import pickle
import os
import sys
import warnings
import time
from collections import deque, Counter

# Suppress TensorFlow and MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warnings
warnings.filterwarnings('ignore')

import cv2
import mediapipe as mp
import numpy as np

# Check if model file exists
model_path = './model.p'
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found!")
    print("\nPlease train the model first by running:")
    print("  1. python create_dataset.py  (if data.pickle doesn't exist)")
    print("  2. python train_classifier.py")
    sys.exit(1)

model_dict = pickle.load(open(model_path, 'rb'))
model = model_dict['model']

# Try to initialize camera - try different indices
cap = None
camera_indices = [0, 1, 2]  # Try common camera indices

for idx in camera_indices:
    test_cap = cv2.VideoCapture(idx)
    if test_cap.isOpened():
        ret, frame = test_cap.read()
        if ret and frame is not None:
            cap = test_cap
            print(f"Camera initialized successfully on index {idx}")
            break
        else:
            test_cap.release()
    else:
        test_cap.release()

if cap is None or not cap.isOpened():
    print("Error: Could not initialize camera!")
    print("Please make sure your webcam is connected and not being used by another application.")
    sys.exit(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: 'space', 27: 'nothing', 28: 'del'
}

print("Starting sign language detection...")
print("Press 'q' to quit")
print("Press 'c' to clear collected text\n")

# Text collection variables
collected_text = ""
last_prediction = None
last_capture_time = 0
DEBOUNCE_DELAY = 1.5  # Minimum time (in seconds) between captures

# Stability check - require same prediction for multiple frames
prediction_buffer = deque(maxlen=20)  # Store last 20 predictions
STABILITY_THRESHOLD = 15  # Require same prediction in at least 15 out of 20 frames
MIN_HOLD_TIME = 1.0  # Minimum time (seconds) prediction must be stable before capturing

# Track when stable prediction first appeared
stable_prediction_start_time = None
current_stable_prediction = None

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    
    # Check if frame was successfully read
    if not ret or frame is None:
        print("Warning: Could not read frame from camera")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    # Reset prediction buffer when no hand is detected
    if not results.multi_hand_landmarks:
        last_prediction = None
        prediction_buffer.clear()
        stable_prediction_start_time = None
        current_stable_prediction = None
    
    if results.multi_hand_landmarks:
        # Draw landmarks for all hands (for visualization)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Only process the first hand for prediction (matching dataset creation)
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Reset for each frame
        x_ = []
        y_ = []
        data_aux = []
        
        # Collect all x and y coordinates first
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        # Normalize coordinates relative to the hand's bounding box
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        # Only make prediction if we have the expected number of features (42)
        if len(data_aux) == 42:
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            prediction = model.predict([np.asarray(data_aux)])
            
            # Model returns string labels directly (e.g., 'A', 'B', 'space', etc.)
            predicted_character = prediction[0]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            
            # Add prediction to buffer (ignore 'nothing' for stability check)
            if predicted_character != 'nothing':
                prediction_buffer.append(predicted_character)
            else:
                # If 'nothing' is detected, reset stability tracking
                if len(prediction_buffer) > 0:
                    prediction_buffer.clear()
                stable_prediction_start_time = None
                current_stable_prediction = None
            
            # Stability check: require same prediction in majority of recent frames
            current_time = time.time()
            
            if len(prediction_buffer) >= STABILITY_THRESHOLD:
                # Count occurrences of each prediction
                prediction_counts = Counter(prediction_buffer)
                most_common = prediction_counts.most_common(1)[0]
                stable_prediction = most_common[0]
                stable_count = most_common[1]
                
                # Check if prediction is stable enough
                if stable_count >= STABILITY_THRESHOLD:
                    # Check if this is a new stable prediction or continuation of previous
                    if stable_prediction != current_stable_prediction:
                        # New stable prediction detected - start tracking time
                        current_stable_prediction = stable_prediction
                        stable_prediction_start_time = current_time
                    else:
                        # Same stable prediction continues - check if held long enough
                        if stable_prediction_start_time is not None:
                            hold_duration = current_time - stable_prediction_start_time
                            
                            # Only capture if:
                            # 1. Prediction has been held for minimum time
                            # 2. It's different from last captured
                            # 3. Enough time has passed since last capture
                            time_since_last_capture = current_time - last_capture_time
                            
                            if (hold_duration >= MIN_HOLD_TIME and
                                stable_prediction != last_prediction and 
                                time_since_last_capture >= DEBOUNCE_DELAY):
                                
                                if stable_prediction == 'space':
                                    collected_text += ' '
                                elif stable_prediction == 'del':
                                    if len(collected_text) > 0:
                                        collected_text = collected_text[:-1]
                                else:
                                    collected_text += stable_prediction
                                
                                last_prediction = stable_prediction
                                last_capture_time = current_time
                                prediction_buffer.clear()  # Clear buffer after capturing
                                stable_prediction_start_time = None
                                current_stable_prediction = None
                else:
                    # Prediction not stable enough - reset tracking
                    stable_prediction_start_time = None
                    current_stable_prediction = None
            else:
                # Not enough predictions yet - reset tracking
                stable_prediction_start_time = None
                current_stable_prediction = None
    
    # Draw text collection box
    box_height = 120
    box_y = H - box_height - 20
    box_x = 20
    box_width = W - 40
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), 2)
    
    # Draw title
    cv2.putText(frame, "Collected Text:", (box_x + 10, box_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display collected text (wrap if needed)
    if collected_text:
        # Wrap text to fit in box
        words = collected_text.split(' ')
        lines = []
        current_line = ""
        max_chars = 50
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if len(test_line) > max_chars:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            lines.append(current_line)
        
        # Show last 3 lines
        start_line = max(0, len(lines) - 3)
        for i, line in enumerate(lines[start_line:]):
            y_pos = box_y + 55 + (i * 25)
            cv2.putText(frame, line, (box_x + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No text collected yet...", (box_x + 10, box_y + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    cv2.imshow('frame', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Clear collected text
        collected_text = ""
        last_prediction = None
        last_capture_time = 0
        prediction_buffer.clear()
        stable_prediction_start_time = None
        current_stable_prediction = None

cap.release()
cv2.destroyAllWindows()