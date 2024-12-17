import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Initialize MediaPipe Hands and akaze detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

akaze = cv2.AKAZE_create()

# Load the trained model
model = load_model('model/model_akaze.keras')

# Function to process a frame from the webcam
def process_frame(frame):
    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to get hand landmarks
    result = hands.process(rgb_frame)

    # max number of descriptors
    max_descriptors = 100

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get the landmark points
            points = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in hand_landmarks.landmark]

            # Flatten the landmarks to a 1D array (21 landmarks * 2 coordinates)
            flattened_landmarks = np.array([coord for point in points for coord in point])

            # Calculate the bounding box coordinates
            x_min = min(point[0] for point in points)
            y_min = min(point[1] for point in points)
            x_max = max(point[0] for point in points)
            y_max = max(point[1] for point in points)

            # Expand the bounding box by a margin
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(frame.shape[1], x_max + margin)
            y_max = min(frame.shape[0], y_max + margin)

            # Crop the frame to only show the bounding box
            cropped_frame = frame[y_min:y_max, x_min:x_max]

            # Convert the cropped frame to grayscale for akaze keypoint detection
            gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply histogram equalization
            gray = cv2.equalizeHist(gray)

            # Apply Canny edge detection
            edges = cv2.Canny(gray, 100, 200)

            # Detect keypoints using akaze
            keypoints = akaze.detect(edges, None)
            keypoints, descriptors = akaze.compute(edges, keypoints)

            # Truncate the descriptors to max_descriptors if they're less, then pad with zeros
            if descriptors is not None:
                if len(descriptors) > max_descriptors:
                    descriptors = descriptors[:max_descriptors]
                else:
                    descriptors = np.pad(descriptors, ((0, max_descriptors - len(descriptors)), (0, 0)), 'constant')

            # If descriptors are None, set them to a zero array
            if descriptors is None:
                descriptors = np.zeros((max_descriptors, 32))  # akaze descriptor length is typically 32 bytes (adjust as needed)

            # Flatten the descriptors to a 1D array (N * 32 where N is the number of keypoints)
            flattened_descriptors = descriptors.flatten()

            # Combine the flattened hand landmarks and akaze descriptors into one vector
            combined_features = np.concatenate((flattened_landmarks, flattened_descriptors))

            return combined_features, (x_min, y_min, x_max, y_max)

    return None, None

# Create a label map from a-z
label_map = {i: chr(65 + i) for i in range(26)}

# Example usage
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        combined_features, bbox = process_frame(frame)
        if combined_features is not None:
            # Reshape combined_features to match the input shape of the model
            combined_features = combined_features.reshape(1, -1)

            # Predict the label using the model
            prediction = model.predict(combined_features)
            predicted_label = np.argmax(prediction, axis=1)[0]

            # Draw the bounding box and print the label
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, str(label_map[predicted_label]), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()