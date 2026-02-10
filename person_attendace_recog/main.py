import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime

# Path to the folder containing employee images
EMPLOYEE_IMAGES_PATH = 'images/'

# Attendance CSV
ATTENDANCE_FILE = 'attendance.csv'

# Load known faces
print("[INFO] Loading employee images...")
known_face_encodings = []
known_face_names = []

for filename in os.listdir(EMPLOYEE_IMAGES_PATH):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        name = os.path.splitext(filename)[0]
        filepath = os.path.join(EMPLOYEE_IMAGES_PATH, filename)
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
            print(f"[INFO] Loaded face for {name}")
        else:
            print(f"[WARNING] No face found in {filename}")

# Initialize attendance record
def mark_attendance(name):
    with open(ATTENDANCE_FILE, 'a+') as f:
        f.seek(0)
        lines = f.readlines()
        names = [line.split(',')[0] for line in lines]
        if name not in names:
            now = datetime.now()
            time_str = now.strftime('%H:%M:%S')
            date_str = now.strftime('%Y-%m-%d')
            f.write(f'{name},{time_str},{date_str}\n')
            print(f"[INFO] Marked attendance for {name}")

# Start webcam
print("[INFO] Webcam started. Press 'q' to quit.")
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and get encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        name = "Unknown"
        if matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                mark_attendance(name)

        # Scale face locations back to original frame size
        top, right, bottom, left = [v * 4 for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Attendance System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
