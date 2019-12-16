import cv2
import numpy as np
import math

def trimDimensions(source):
    trimmedVideo = source[math.floor(source.shape[0] * (1/4)):source.shape[0], math.floor(source.shape[1] * (1/5)):math.ceil(source.shape[1] * (4/5))]
    return trimmedVideo

'''
Calculates the birds eye view transformation matrix and transforms 
the given input image to the desired transformedImage
'''
def birdsEyeTransform(image):
    source = np.float32([
        [0,50 + image.shape[0] / 2], # (0, halbe bildhoehe)
        [image.shape[1],50 + image.shape[0] / 2], # (bildbreite, halbe bildhoehe)
        [0,image.shape[0]], # (0,bildhoehe)
        [image.shape[1],image.shape[0]]
        ])
    destination = np.float32([[0,0],[image.shape[1],0],[0,image.shape[0]],[image.shape[1],image.shape[0]]])
    transformMatrix = cv2.getPerspectiveTransform(source, destination)
    transformedImage = cv2.warpPerspective(image, transformMatrix, (image.shape[1],image.shape[0]))
    
    cv2.imshow("Bird view",transformedImage)
    return transformedImage

def do():
    video = cv2.VideoCapture("FF_besser_Trim.mp4")

    while True:
        ret, frame = video.read()

        if not ret:
            video = cv2.VideoCapture("FF_besser_Trim.mp4")
            continue

        frame = trimDimensions(frame)
        frame = birdsEyeTransform(frame)
        grayVideo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(grayVideo, 128, 255)

        #put mask on canny
        '''
        mask = np.zeros_like(canny)
        pointsOfInterest = np.array([[300,frame.shape[0] - 160],[505,475],[720,475],[870,frame.shape[0] - 160],[620,520],[550,610],[400,610]])
        cv2.fillPoly(mask, [pointsOfInterest], 255)
        maskedCanny = cv2.bitwise_and(canny,mask)
        '''
        #blurredMaskedCanny = cv2.GaussianBlur(maskedCanny, (5,5),0)

        # hough on masked canny
        houghLine = cv2.HoughLinesP(canny, 1, np.pi/180, 9, np.array([]), 8, 200)
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

        cv2.imshow("canny", canny)
        cv2.imshow("hough", frame)

        key = cv2.waitKey(25)
        if key == 27:
            break
    video.release()

if __name__ == '__main__':
    do()

cv2.destroyAllWindows()