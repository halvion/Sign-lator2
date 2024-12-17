import cv2
import mediapipe as mp
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input

# Initialize MediaPipe Hands and ORB detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

orb = cv2.ORB_create()

# Directory containing the images
train_dir = './dataset/train/images/'
test_dir = './dataset/test/images/'
val_dir = './dataset/valid/images/'

labels = os.listdir(train_dir)


# Function to process each image
def process_image(image_path):
    # Load the image
    frame = cv2.imread(image_path)

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

            # Convert the cropped frame to grayscale for ORB keypoint detection
            gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply histogram equalization
            gray = cv2.equalizeHist(gray)

            # Apply Canny edge detection
            edges = cv2.Canny(gray, 100, 200)

            # Detect keypoints using ORB
            keypoints = orb.detect(edges, None)
            keypoints, descriptors = orb.compute(edges, keypoints)

            # Truncate the descriptors to max_descriptors if they're less, then pad with zeros
            if descriptors is not None:
                if len(descriptors) > max_descriptors:
                    descriptors = descriptors[:max_descriptors]
                else:
                    descriptors = np.pad(descriptors, ((0, max_descriptors - len(descriptors)), (0, 0)), 'constant')

            # If descriptors are None, set them to a zero array
            if descriptors is None:
                descriptors = np.zeros((max_descriptors, 32))  # ORB descriptor length is typically 32 bytes (adjust as needed)

            # Flatten the descriptors to a 1D array (N * 32 where N is the number of keypoints)
            flattened_descriptors = descriptors.flatten()

            # Combine the flattened hand landmarks and ORB descriptors into one vector
            combined_features = np.concatenate((flattened_landmarks, flattened_descriptors))

            return combined_features

    return None

# List of combined features and labels
features_list = []
labels_list = []

# Iterate over images in the directory and process them
for dir in [train_dir, test_dir, val_dir]:
    for idx, label in enumerate(os.listdir(dir)):
        for filename in os.listdir(os.path.join(dir, label)):
            filename = dir + label + "/" + filename
            print(f"{filename}: {label}")
            # Process each image
            combined_features = process_image(filename)

            if combined_features is not None:
                # Combine label idx and features into one line
                feature_line = np.concatenate(([idx], combined_features))
                features_list.append(feature_line)

# Convert features_list to a 2D numpy array
features_array = np.array(features_list)

# Save the features_array into a csv file
np.savetxt('preprocess/features_orb.csv', features_array, delimiter=',')