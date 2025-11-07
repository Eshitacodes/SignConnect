import pickle
import os
import sys
import warnings

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
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z',
    26: 'space',
    27: 'nothing',
    28: 'del'
}

print("Starting sign language detection...")
print("Press 'q' to quit\n")

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

    cv2.imshow('frame', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
