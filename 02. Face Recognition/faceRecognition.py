'''
    Press 'q' to close the camera and stop the process.
'''

from cv2 import cv2
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

###################################   KNN   ######################################


def dist(x1, x2):
    return np.sqrt(sum((x1 - x2)**2))


def knn(X, Y, query, k=10):
    vals = []
    m = X.shape[0]
    for i in range(m):
        d = dist(query, X[i])
        vals.append((d, Y[i]))

    vals = sorted(vals)
    vals = vals[:k]
    vals = np.array(vals)
    vals = np.unique(vals[:, 1], return_counts=True)

    idx = vals[1].argmax()
    ans = vals[0][idx]

    return int(ans)


###################################################################################

skip = 0
face_data = []
label = []
classid = 0
names = {}

dataset_path = './Dataset/'

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[classid] = fx[:-4]
        dataItem = np.load(dataset_path + fx)
        dataItem = dataItem[ :20, :]
        face_data.append(dataItem)

        target = classid * np.ones((dataItem.shape[0], ))
        classid += 1
        label.append(target)

face_data = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(label, axis=0)

print(face_data.shape)
print(face_labels.shape)

### TESTING

while True:

    ret, frame = cap.read()
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        #Extract frame with offset 10px
        face_section = frame[y - 10:y + h + 10][x - 10:x + w + 10]

        try:
            face_section = cv2.resize(face_section, (100, 100))

            out = knn(face_data, face_labels, face_section.flatten())

            cv2.putText(frame, names[out], (x, y - 10), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        except:
            pass

    cv2.imshow('Frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
