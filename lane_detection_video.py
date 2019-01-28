import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

video = cv2.VideoCapture("Lane_detectionVideo_dark_city.mp4")

while True:
    ret, frame = video.read()

    if ret:
        grayVideo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(grayVideo, 10, 255)
    else: break

    #put mask on canny
    mask = np.zeros_like(canny)
    pointsOfInterest = np.array([[300,frame.shape[0] - 160],[505,475],[720,475],[870,frame.shape[0] - 160],[620,520],[550,610],[400,610]])
    cv2.fillPoly(mask, [pointsOfInterest], 255)
    maskedCanny = cv2.bitwise_and(canny,mask)
    #blurredMaskedCanny = cv2.GaussianBlur(maskedCanny, (5,5),0)

    # hough on masked canny
    houghLine = cv2.HoughLinesP(maskedCanny, 1, np.pi/180, 9, np.array([]), 8, 200)
    try:
        for x1, y1, x2, y2 in houghLine[0]:
            maxOfX = max(x1, x2)
            minOfX = min(x1, x2)
            maxOfY = max(y1, y2)
            minOfY = min(y1, y2)
            slope = abs((maxOfY - minOfY) / (maxOfX - minOfX))
            if slope > 0.25 and slope < math.inf:
                cv2.line(frame, (x1, y1),(x2, y2),(0,255,0),2)
    except:
        pass

    #plt.imshow(frame)
    #plt.show()
    cv2.imshow("canny", maskedCanny)
    cv2.imshow("hough", frame)

    key = cv2.waitKey(25)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()