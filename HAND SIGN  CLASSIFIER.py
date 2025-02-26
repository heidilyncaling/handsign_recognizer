import pickle
import cv2
import mediapipe as mp
import numpy as np

try:
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Model file 'model.pickle' not found. Train the model first!")
    exit()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

cap = cv2.VideoCapture(1)  
if not cap.isOpened():
    print("❌ Error: Could not open webcam. Try changing the camera index!")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

labels_dict = {i: chr(65 + i) for i in range(26)}  

while cap.isOpened():
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture frame!")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            min_x, min_y = min(x_), min(y_)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)

            expected_features = 42 
            if len(data_aux) != expected_features:
                print(f"❌ Error: Expected {expected_features} features, but got {len(data_aux)}")
                continue

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]

            x1, y1 = int(min_x * W) - 20, int(min_y * H) - 20
            x2, y2 = int(max(x_) * W) + 20, int(max(y_) * H) + 20
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Hand Sign Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Program terminated successfully.")
