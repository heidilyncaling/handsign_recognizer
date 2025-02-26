import os
import cv2

class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 100

cap = cv2.VideoCapture(1)

for label in class_labels:
    folder_path = os.path.join(DATA_DIR, label)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print(f'Collecting data for class {label}')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't capture image. Check your camera.")
            break

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't capture image.")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        cv2.imwrite(os.path.join(folder_path, f'{label}_{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
