import cv2
import numpy as np

video = cv2.VideoCapture("Lane_detectionVideo_beginEndCut.mp4")

while True:
    ret, frame = video.read()

    grayVideo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(grayVideo, 128, 255)

    if not ret:
        video = cv2.VideoCapture("Lane_detectionVideo_beginEndCut.mp4")
        grayVideo = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        continue

    cv2.imshow("original", frame)
    cv2.imshow("canny", canny)

    key = cv2.waitKey(25)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()