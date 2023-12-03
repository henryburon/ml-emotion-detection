import cv2
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

# Load your trained logistic regression model
loaded_model = joblib.load('logistic_regression/trained_model/happy_neutral_surprised.pk1')

# Mapping indices to emotions
emotion_indices = {
    0: 'angry',
    1: 'happy',
    2: 'neutral',
    3: 'sad',
    4: 'disgusted',
    5: 'fearful',
    6: 'surprised'
}

# Function to detect emotion in a frame and overlay it on the screen
def detect_emotion(frame):
    # Convert the frame to grayscale and resize to 48x48
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (48, 48))

    # Extract HOG features
    features = hog(
        gray_resized,
        orientations=12,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )

    # Reshape the features to match the model input shape
    features = features.reshape(1, -1)

    # Predict emotion using the loaded model
    prediction = loaded_model.predict(features)[0]

    # Map the predicted index to emotion label
    predicted_emotion = emotion_indices[prediction]

    # Overlay the predicted emotion on the frame
    cv2.putText(frame, predicted_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)  # Use '0' for the default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the frame with detected emotion
    cv2.imshow('Emotion Detection', detect_emotion(frame))

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()
