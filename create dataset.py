import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dataset Directory
DATA_DIR = './data'

# Ensure the dataset directory exists
if not os.path.exists(DATA_DIR):
    print(f"Error: DATA_DIR '{DATA_DIR}' does not exist. Run the data collection script first.")
    exit()

# Initialize data storage
data = []
labels = []

# Get class labels (A-Z, 0-9)
class_labels = sorted(os.listdir(DATA_DIR))  # Ensures proper ordering

# Process each label (A-Z, 0-9)
for label in class_labels:
    label_path = os.path.join(DATA_DIR, label)
    
    if not os.path.isdir(label_path):
        continue  # Skip if it's not a directory
    
    for img_path in os.listdir(label_path):
        img_file = os.path.join(label_path, img_path)
        
        img = cv2.imread(img_file)
        if img is None:
            print(f"Warning: Unable to read image {img_file}. Skipping.")
            continue  # Skip if the image cannot be loaded

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []
                y_ = []
                data_aux = []

                # Extract hand landmarks
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                # Normalize landmarks
                min_x, min_y = min(x_), min(y_)
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min_x)
                    data_aux.append(landmark.y - min_y)

                data.append(data_aux)
                labels.append(label)  # Store the label (A-Z, 0-9)

# Save the processed dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("✅ Dataset creation completed successfully. Saved as 'data.pickle'.")
