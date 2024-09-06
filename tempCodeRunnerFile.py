import cv2
import numpy as np
import json

# Load all face encodings from face_features.json
def load_face_encodings():
    try:
        with open("assets/face_features_old.json", "r") as f:
            face_data = json.load(f)
        return face_data
    except OSError:
        print("Failed to load face recognition data")
        return None

# Function to capture image from webcam and return a frame
def capture_frame():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()  # Capture a frame from webcam
    if ret:
        return frame
    else:
        print("Failed to capture image")
        return None

# Extract basic features from the captured frame
def extract_basic_features(frame):
    # Resize the image to make feature extraction faster
    resized_frame = cv2.resize(frame, (64, 64))

    # Convert to grayscale for simplified feature extraction
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Simulate feature extraction: we'll use the flattened pixel values as "features"
    flattened_features = gray_frame.flatten()

    # Normalize the features for comparison
    normalized_features = flattened_features / 255.0

    return normalized_features

# Compute Euclidean distance between two feature vectors (captured and stored)
def compute_similarity(features1, features2):
    # Ensure both features are arrays (vectors)
    features1 = np.array(features1)
    features2 = np.array(features2)

    # Calculate the Euclidean distance between the two vectors
    distance = np.linalg.norm(features1 - features2)

    # Convert distance to similarity score (lower distance = higher similarity)
    similarity_score = 1 / (1 + distance)  # Normalized similarity between 0 and 1

    return similarity_score

# Compare captured image features with all stored encodings
def compare_face(captured_img, face_data):
    captured_features = extract_basic_features(captured_img)

    # Iterate through all stored face encodings
    for name, stored_features in face_data.items():
        similarity_score = compute_similarity(captured_features, stored_features)

        # Set a threshold for face matching
        if similarity_score > 0.8:
            return name

    return None

# Main function to run the face recognition
def main():
    # Load face encodings from the JSON file
    face_data = load_face_encodings()
    if face_data is None:
        print("No face data found. Exiting...")
        return

    # Capture frame from webcam
    while True:
        frame = capture_frame()
        if frame is not None:
            # Compare the captured frame with the stored face encodings
            recognized_name = compare_face(frame, face_data)

            # Display the results
            if recognized_name:
                print(f"Recognized: {recognized_name}")
                cv2.putText(frame, f"Recognized: {recognized_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                print("No match found")
                cv2.putText(frame, "No match found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show the frame in a window
            cv2.imshow('Webcam Feed', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the webcam and close windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
