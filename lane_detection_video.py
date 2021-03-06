import cv2
import numpy as np
import matplotlib.pyplot as plt

video = cv2.VideoCapture("Lane_detectionVideo_beginEndCut.mp4")

while True:
    ret, frame = video.read()

    grayVideo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(grayVideo, 128, 255)

    #put mask on canny
    mask = np.zeros_like(canny)
    pointsOfInterest = np.array([[340,615],[580,450],[680,450],[850,600]])
    cv2.fillPoly(mask, [pointsOfInterest], 255)
    maskedCanny = cv2.bitwise_and(canny,mask)
    blurredMaskedCanny = cv2.GaussianBlur(maskedCanny, (5,5),0)

    # hough on masked canny
    houghLine = cv2.HoughLinesP(blurredMaskedCanny, 1, np.pi/180, 1, np.array([]), 128, 200)
    try:
        for x1, y1, x2, y2 in houghLine[0]:
            maxOfX = max(x1, x2)
            minOfX = min(x1, x2)
            maxOfY = max(y1, y2)
            minOfY = min(y1, y2)
            slope = abs((maxOfY - minOfY) / (maxOfX - minOfX))
            if slope > 0.5:
                cv2.line(frame, (x1, y1),(x2, y2),(0,255,0),2)
    except:
        pass

    #if not ret:
    #    video = cv2.VideoCapture("Lane_detectionVideo_beginEndCut.mp4")
    #    grayVideo = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    #    continue

    #plt.imshow(frame)
    #plt.show()
    cv2.imshow("blurry canny", blurredMaskedCanny)
    cv2.imshow("hough", frame)

    key = cv2.waitKey(25)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()