import cv2
import numpy as np
import os
import csv
import time
import pickle
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

video_capture = cv2.VideoCapture(0)

face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

min_length = min(len(FACES), len(LABELS))
FACES = FACES[:min_length]
LABELS = LABELS[:min_length]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

print(f"FACES shape: {FACES.shape}")
print(f"LABELS length: {len(LABELS)}")

imgBack = cv2.imread("img.png")

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video_capture.read()
    
    if not ret:
        print("Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_detect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        
        output = knn.predict(resized_img)
        
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        
        file_path = "Attendance/Attendance_" + date + ".csv"
        exist = os.path.isfile(file_path)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, output[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        attendance = [str(output[0]), str(timestamp)]
        
        imgBack[162:162 + 480, 55:55 + 640] = frame

    cv2.imshow("frame", imgBack)

    k = cv2.waitKey(1)

    if k == ord('o'):
        time.sleep(5)

        with open(file_path, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)
            writer.writerow(attendance)   

    if k == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
