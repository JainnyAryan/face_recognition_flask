from flask import Flask
from flask_socketio import SocketIO, emit
import face_recognition
import numpy as np
import json
from io import BytesIO
import base64
from PIL import Image

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


def save_and_show_received_image(image: Image):
    image.save('received_image_rotated.jpg')
    print('Rotated image saved as received_image_rotated.jpg')
    image.show()


with open('assets/face_features.json', 'r') as f:
    features_data_from_file = json.load(f)
    features = {k: np.array(v) for k, v in features_data_from_file.items()}
    known_face_encodings = list(features.values())
    known_face_names = list(features.keys())


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        image_data = base64.b64decode(data['frame'])
        image = Image.open(BytesIO(image_data))
        image = image.rotate(-90, expand=True)
        # save_and_show_received_image(image)
        image_np = np.array(image)
        face_locations = face_recognition.face_locations(image_np, number_of_times_to_upsample=3)
        face_encodings = face_recognition.face_encodings(image_np, face_locations, num_jitters=3)

        if not face_encodings:
            print("No face encodings found.")
            emit('recognition_result', json.dumps({"name": "Unknown", "confidence": 0.0, "box": [0, 0, 0, 0]}))
            return

        print(f"Found {len(face_encodings)} face(s).")
        face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
        best_match_index = np.argmin(face_distances)
        name = known_face_names[best_match_index]
        confidence = 1 - face_distances[best_match_index]

        top, right, bottom, left = face_locations[0]

        # Adjust the bounding box coordinates for the rotation
        image_width, image_height = image.size

        # Rotate 90 degrees clockwise
        new_left = image_height - bottom
        new_top = left
        new_right = image_height - top
        new_bottom = right
        box = [new_left, new_top, new_right, new_bottom]

        print(f"Detected face: {name} with confidence {confidence}", box)
        emit('recognition_result', json.dumps({"name": name, "confidence": confidence, "box": box}))
    except Exception as e:
        print(f"Error processing video frame: {e}")
        emit('recognition_result', json.dumps({"name": "Unknown", "confidence": 0.0, "box": [0, 0, 0, 0]}))

if __name__ == '__main__':
    socketio.run(app, port=6000, debug=True)
