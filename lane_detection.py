import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Autobahnspur2.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize = 3)
sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize = 3)
sobelall = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize = 3)

canny = cv2.Canny(gray_image, 128, 255)
mask = np.zeros_like(canny)

# mask image in 2 halfs to extract the street only
mask_points_left = np.array([[0, 190], [205, 93], [266, 103], [209, 294], [0, 294]])
mask_points_right = np.array([[285, 104], [339, 97], [513, 187], [513, 294], [325, 294]])
cv2.fillPoly(mask, [mask_points_left], 255)
half_masked_image = cv2.bitwise_and(canny, mask)
cv2.fillPoly(mask, [mask_points_right], 255)
masked_image = cv2.bitwise_and(canny, mask)

minLineLength = 3
lines = cv2.HoughLinesP(masked_image, 1, np.pi/180, 20, minLineLength, maxLineGap = 2)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

cv2.imshow('sobelx', sobelx)
cv2.imshow('sobely', sobely)
cv2.imshow('sobelall', sobelall)
cv2.imshow('canny edge detector', canny)
cv2.imshow('original with hough lines', image)

# wait for any key to be pressed and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
