import cv2
import numpy as np

image = cv2.imread('Autobahnspur2.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize = 3)
sobely = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize = 3)

#mask = np.zeros_like(canny)

# mask image in 2 halfs to extract the street only
#mask_points_left = np.array([[0, 190], [205, 93], [266, 103], [209, 294], [0, 294]])
#mask_points_right = np.array([[285, 104], [339, 97], [513, 187], [513, 294], [325, 294]])
#cv2.fillPoly(mask, [mask_points_left], 255)
#half_masked_image = cv2.bitwise_and(canny, mask)
#cv2.fillPoly(mask, [mask_points_right], 255)
#masked_image = cv2.bitwise_and(canny, mask)

#minLineLength = 3
circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(gray_image,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(gray_image,(i[0],i[1]),2,(0,0,255),3)
'''
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
'''
cv2.imshow('sobelx', sobelx)
cv2.imshow('sobely', sobely)
#cv2.imshow('canny edge detector', canny)
cv2.imshow('original with hough lines', gray_image)

# wait for any key to be pressed and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
