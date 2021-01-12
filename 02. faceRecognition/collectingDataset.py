'''
    Press 'q' to close the camera and stop the process.
'''

from cv2 import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = './Dataset/'

fileName = input('Enter Name: ')

while True:
    ret, frame = cap.read()

    if ret == False:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        #Extract frame with offset 10px
        face_section = frame[y - 10:y + h + 10][x - 10:x + w + 10]

        try:
            face_section = cv2.resize(face_section, (100, 100))
        except:
            pass

        skip += 0
        if skip % 5 == 0:
            face_data.append(face_section)
            print(len(face_data))

    cv2.imshow('Frame', frame)
    # cv2.imshow('Face', face_section)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

np.save(dataset_path + fileName + '.npy', face_data)
print('Data saved at ' + dataset_path + fileName + '.npy')

cap.release()
cv2.destroyAllWindows()