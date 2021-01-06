'''
    Press 'q' to close the camera and stop the process.
'''

from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

eyesCascade = cv2.CascadeClassifier('C:\\Users\\AJAY\\Desktop\\ML\\snapchatLikeFilter\\frontalEyes35x16.xml')
noseCascade = cv2.CascadeClassifier('C:\\Users\\AJAY\\Desktop\\ML\\snapchatLikeFilter\\Nose18x15.xml')
img = cv2.imread('C:\\Users\\AJAY\\Desktop\\ML\\snapchatLikeFilter\\photos\\Before.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

glasses = cv2.imread('C:\\Users\\AJAY\\Desktop\\ML\\snapchatLikeFilter\\filters\glasses.png', -1)
mustache = cv2.imread('C:\\Users\\AJAY\\Desktop\\ML\\snapchatLikeFilter\\filters\mustache.png', -1)

eyes = eyesCascade.detectMultiScale(img)
for x, y, w, h in eyes:
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    glasses = cv2.resize(glasses, (w, h))
    gw, gh, gc = glasses.shape
    for i in range(gw):
        for j in range(gh):
            if glasses[i, j][3] != 0:
                img[y + i, x + j] = glasses[i, j]

nose = noseCascade.detectMultiScale(img, 1.3, 7)
for x, y, w, h in nose:
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    mustache = cv2.resize(mustache, (w + 20, int(h / 1.5)))
    mw, mh, mc = mustache.shape
    for i in range(mw):
        for j in range(mh):
            if mustache[i, j][3] != 0:
                img[y + h // 2 + 5 + i, x - 8 + j] = mustache[i, j]
img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
cv2.imshow('Pic', img)
cv2.waitKey(0)