import cv2
import numpy as np

cap = cv2.VideoCapture('assets/Samplecvfootage.mp4')

ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read first frame.")
    exit()

print("Select a region of interest (ROI) by dragging a rectangle.")
boundingbox = cv2.selectROI("Select ROI", first_frame, True)
cv2.destroyAllWindows()

print(f"Selected ROI: {boundingbox}")

tracker = cv2.TrackerCSRT_create()
tracker.init(first_frame, boundingbox)

while True: #Real Time
    ret, frame = cap.read()

    if not ret:
        break

    success, boundingbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in boundingbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Selected ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking Lost", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Racing Video Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
