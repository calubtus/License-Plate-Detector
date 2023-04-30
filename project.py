import cv2 as cv
import numpy as np
import random
import time
import imutils

def verifySizes(rect):
    (x, y), (width, height), angle = rect
    error=0.4
    #Spain car plate size: 52x11 aspect 4,7272
    aspect=4.7272
    #Set a min and max area. All other patchs are discarded
    min= 15*aspect*15 # minimum area
    max= 125*aspect*125 # maximum area
    #Get only patchs that match to a respect ratio.
    rmin= aspect-aspect*error
    rmax= aspect+aspect*error

    area = height * width

    try:
        r = width / height
    except ZeroDivisionError:
        return False
    if r < 1:
        r = height / width


    if (( area < min or area > max ) or ( r < rmin or r > rmax )):
        return False
    else:
        return True

# image_path = '2715DTZ.jpg'
image_path = '3028BYS.JPG'
input = cv.imread(image_path)
rows, columns, channels = input.shape

# Convert image into grayscale image as color can't help us
img_gray = cv.cvtColor(input, cv.COLOR_BGR2GRAY)

# Apply a Gaussian blur of 5x5 to remove vertical edges that are produced from noise
img_blur = cv.blur(img_gray, (5, 5))

# Find vertical edges using Sobel filter and first horizontal derivative
img_sobel = cv.Sobel(img_blur, cv.CV_8U, 1, 0, ksize=3, scale=1, delta=0)


# Apply threshold filter to obtain binary image (threshold value from Otsu's method)
img_binary = cv.threshold(img_sobel, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)[1]

# Apply close morphological opertaion to remove blank spaces between each vertical edge
# and connect all regions that have high number of edges
element = cv.getStructuringElement(cv.MORPH_RECT,(17,3))
img_morph =  cv.morphologyEx(img_binary, cv.MORPH_CLOSE, element)

# Find external contours to list possible regions where license plate can be located
contours = cv.findContours(img_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
contours = imutils.grab_contours(contours)
# contours = list(contours)

# Create new image on what will be the final processed image
result = input.copy()

# For each contour detected, extract the bounding rectangle of minimal area
rects = []
for i,contour in enumerate(contours):
    minRec = cv.minAreaRect(contour)
    if verifySizes(minRec) == True:
        # Draw contours on the result image
        cv.drawContours(result, [contour], -1, (255, 0, 0), 1)
        rects.append(minRec)

mask = np.zeros(input.shape[:2], dtype='uint8')

# define range of blue color in HSV
low_blue = np.array([50,140,100])
high_blue = np.array([130,255,255])

for i, rect in enumerate(rects):
    mask = np.zeros(input.shape[:2], dtype='uint8')
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(mask, [box], 0, 255, cv.FILLED)
    result = cv.bitwise_and(input, input, mask = mask)
    result_hsv = cv.cvtColor(result, cv.COLOR_BGR2HSV)
    # # Check how much blue is in the image
    blue_mask = cv.inRange(result_hsv, low_blue, high_blue)
    result = cv.bitwise_and(input, input, mask = blue_mask)
    print(sum(sum(blue_mask)))
    if sum(sum(blue_mask)) > 2000:
        input = cv.drawContours(input,[box],-1,(0, 0, 255),2)
    cv.imshow("mask", result)
    cv.waitKey(0)

    # box = cv.boxPoints(rect)
    # box = np.int0(box)
    # test = cv.drawContours(result,[box],-1,(0, 0, 255),2)
cv.imshow("rect", input)
cv.waitKey(0)
    # (x, y), (width, height), angle = rect
    # center = (int(x),int(y))

 # Ratio of image size vs how much blue is expected