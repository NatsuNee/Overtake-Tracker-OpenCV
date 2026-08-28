import cv2
import numpy as np

cap = cv2.VideoCapture('assets/Samplecvfootage.mp4')

while True:
    ret, frame = cap.read()

    cv2.imshow('Racing Video Tracking', frame)
    if cv2.waitKey(8) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
