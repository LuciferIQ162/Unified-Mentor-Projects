import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the model

model = load_model('/home/shobhit/AI-ML/Unified mentor/ASL_detection_ML/asl_model.h5')

# Class names
class_names = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing']

#Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

#Initialize the default camera
cap = cv2.VideoCapture(1)

while True:
    #Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
       break
    print(f"Frame shape: {frame.shape}")

    #Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    #Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Process the frame and find hands
    results = hands.process(rgb_frame)

    #Resize the frame to match the model's input shape
    img = cv2.resize(frame, (200, 200))
    print(f"Resized image shape: {img.shape}")

    #Preprocess the image
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    print(f"Expanded image shape: {img.shape}")

    #Make predictions
    prediction = model.predict(img)
    print(f"Prediction: {prediction}")
    predicted_class = np.argmax(prediction)
    predicted_letter = class_names[predicted_class]

    #Display the prediction on the frame
    cv2.putText(frame, predicted_letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #Draw hand landmarks and display position
    if results.multi_hand_landmarks:
       for hand_landmarks in results.multi_hand_landmarks:
           # Draw landmarks on the frame
           mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

           # Get the wrist coordinates (landmark 0)
           wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
           cx, cy = int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])

           # Display the hand position
           cv2.putText(frame, f'Pos: ({cx}, {cy})', (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    #Show the frame
    cv2.imshow('ASL Detection', frame)

    #Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
    print("Processed a frame")

#Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()