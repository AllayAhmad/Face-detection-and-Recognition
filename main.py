import sqlite3
import cv2 as cv
import dlib
import json
import numpy as np
import time

def insert_data(person_id, name, gender, facial_features):
    conn = sqlite3.connect('sqlite.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS person_info (
            person_id INTEGER PRIMARY KEY,
            name TEXT,
            gender TEXT,
            facial_features TEXT
        )
    ''')
    
    # Check if person_id already exists
    cursor.execute("SELECT person_id FROM person_info WHERE person_id=?", (person_id,))
    if cursor.fetchone():
        conn.close()
        return False, "Person ID already exists. Please enter a unique ID."
    
    cursor.execute('''
        INSERT INTO person_info (person_id, name, gender, facial_features)
        VALUES (?, ?, ?, ?)
    ''', (person_id, name, gender, facial_features))
    conn.commit()
    conn.close()
    return True, "Person registered successfully!"

def capture_and_extract_features():
    cap = cv.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    facial_features = {}
    start_time = time.time()
    detection_duration = 10  # seconds
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for i, face in enumerate(faces):
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            landmarks = predictor(gray, face)
            features = {}
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                features[f"point_{n}"] = {"x": x, "y": y}
            facial_features[f"face_{i}"] = features
        
        elapsed_time = time.time() - start_time
        if elapsed_time > detection_duration:
            break
        
        cv.imshow('Frame', frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()
    
    print("Face detection completed.")
    
    facial_features_json = json.dumps(facial_features)
    return facial_features_json

def compare_faces(stored_features, live_face_features):
    stored_features = json.loads(stored_features)
    live_face_features = json.loads(live_face_features)
    
    def calculate_distance(points1, points2):
        distances = []
        for key in points1:
            if key in points2:
                p1 = points1[key]
                p2 = points2[key]
                distance = np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
                distances.append(distance)
        return np.mean(distances) if distances else float('inf')

    if not live_face_features:
        return False
    
    for key in stored_features:
        if key in live_face_features:
            stored_points = stored_features[key]
            live_points = live_face_features[key]
            distance = calculate_distance(stored_points, live_points)
            if distance < 50:  # Adjust this threshold as necessary
                return True
    return False

def mark_attendance(person_id):
    conn = sqlite3.connect('sqlite.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            person_id INTEGER, 
            time DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(person_id, time)
        )
    ''')
    cursor.execute('''
        INSERT INTO attendance (person_id)
        VALUES (?)
    ''', (person_id,))
    print(f"Attendance marked for person_id: {person_id}")
    conn.commit()
    conn.close()
    return True

def main():
    print("Welcome")
    print("Choose your choice: ")
    print("1. Register a new person")
    print("2. Mark attendance")
    choice = input("Enter your choice: ")
    if choice == '1':
        while True:
            person_id = input("Enter your person ID: ")
            name = input("Enter your name: ")
            gender = input("Enter your gender: ")
            facial_features = capture_and_extract_features()
            success, message = insert_data(person_id, name, gender, facial_features)
            print(message)
            if success:
                break
    elif choice == '2':
        person_id = input("Enter your person ID: ")
        conn = sqlite3.connect('sqlite.db')
        cursor = conn.cursor()
        cursor.execute("SELECT facial_features FROM person_info WHERE person_id=?", (person_id,))
        stored_features = cursor.fetchone()
        conn.close()
        if stored_features:
            live_face_features = capture_and_extract_features()
            if compare_faces(stored_features[0], live_face_features):
                mark_attendance(person_id)
            else:
                print("Face not recognized")
        else:
            print("Person ID not found")

if __name__ == "__main__":
    main()
