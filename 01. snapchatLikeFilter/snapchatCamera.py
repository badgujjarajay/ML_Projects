from cv2 import cv2
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

skip = 0
face_data = []
label = []
classid = 0
names = {}

cap = cv2.VideoCapture(0)
eyesCascade = cv2.CascadeClassifier('C:\\Users\\AJAY\\Desktop\\ML\\snapchatLikeFilter\\frontalEyes35x16.xml')
noseCascade = cv2.CascadeClassifier('C:\\Users\\AJAY\\Desktop\\ML\\snapchatLikeFilter\\Nose18x15.xml')

glasses = cv2.imread('C:\\Users\\AJAY\\Desktop\\ML\\snapchatLikeFilter\\filters\glasses.png', -1)
mustache = cv2.imread('C:\\Users\\AJAY\\Desktop\\ML\\snapchatLikeFilter\\filters\mustache.png', -1)


while True:

    ret, frame = cap.read()
    if ret == False:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    eyes = eyesCascade.detectMultiScale(frame)
    for x, y, w, h in eyes:
        glasses = cv2.resize(glasses, (w, h))
        gw, gh, gc = glasses.shape
        for i in range(gw):
            for j in range(gh):
                if glasses[i, j][3] != 0:
                    frame[y + i, x + j] = glasses[i, j]

    nose = noseCascade.detectMultiScale(frame, 1.3, 7)
    for x, y, w, h in nose:
        mustache = cv2.resize(mustache, (w + 20, int(h / 1.5)))
        mw, mh, mc = mustache.shape
        for i in range(mw):
            for j in range(mh):
                if mustache[i, j][3] != 0:
                    frame[y + h // 2 + 5 + i, x - 8 + j] = mustache[i, j]

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
