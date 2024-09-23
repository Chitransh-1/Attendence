import cv2
import numpy as np
import os
import pickle
 
video_capture = cv2.VideoCapture(0)

face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_data = []

i = 0

name = input("Enter Your Name: ")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_detect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50))

        if len(face_data) < 100 and i % 10 == 0:
            face_data.append(resized_img)
            cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("frame", frame)
    
    i += 1 
    if len(face_data) == 50:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

face_data = np.array(face_data)
face_data = face_data.reshape(50, -1)

if not os.path.exists('data/'):
    os.makedirs('data/')

if 'names.pkl' not in os.listdir('data/'):
    names = [name] * 50
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * 50
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'face_data.pkl' not in os.listdir('data/'):
    with open('data/face_data.pkl', 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open('data/face_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, face_data, axis=0)
    with open('data/face_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
