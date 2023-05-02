import cv2 as cv
import numpy as np
import imutils

# Validation psudocode taken from (Levgeb et al. 169) 
def isPlateSize(rect):
    _, (width, height), _ = rect
    error_margin = 0.4
    # License plate size is 52x11 hence aspect ratio is 4.72
    aspect_ratio = 52/11
    # Set a minimum and maximum area expected to be valid
    minimal_area = 15 * aspect_ratio * 15
    maximum_area = 125 * aspect_ratio * 125
    # Set a minimum and maximum aspect ratios to be valid
    minimal_ratio = aspect_ratio - (aspect_ratio * error_margin)
    maximum_ratio = aspect_ratio + (aspect_ratio * error_margin)

    # Retrieve rectangle properties
    area = height * width
    try:
        ratio = width / height
    except ZeroDivisionError:
        return False
    if ratio < 1:
        ratio = height / width

    # Check if passed rectagle passes licese plate criteria
    if (( area < minimal_area or area > maximum_area ) or ( ratio < minimal_ratio or ratio > maximum_ratio)):
        return False
    else:
        return True

def displayImage(image):
    cv.imshow(" ", image)
    cv.waitKey(0)

# Modify to include a different image from database
image_path = '2715DTZ.jpg'
# image_path = '3028BYS.JPG'
input = cv.imread(image_path)

# Convert image into grayscale image as color can't help us
img_gray = cv.cvtColor(input, cv.COLOR_BGR2GRAY)

# Apply a Gaussian blur using a 5x5 kernel to remove vertical edges that are produced from noise
img_blur = cv.blur(img_gray, (5, 5))
# Display Gaussian filter
displayImage(img_blur)

# Find vertical edges using Sobel filter and first horizontal derivative
img_sobel = cv.Sobel(img_blur, cv.CV_8U, 1, 0, ksize=3, scale=1, delta=0)
# Display Sobel filter
displayImage(img_sobel)

# Apply threshold filter to obtain binary image (threshold value from Otsu's method)
img_binary = cv.threshold(img_sobel, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)[1]
# Display binary transformation
displayImage(img_binary)

# Apply close morphological opertaion to remove blank spaces between each vertical edge
# and connect tight regions
element = cv.getStructuringElement(cv.MORPH_RECT,(17,3))
img_morph =  cv.morphologyEx(img_binary, cv.MORPH_CLOSE, element)
# Display morphological opertaion
displayImage(img_morph)

# Find external contours to list possible regions where license plate can be located
contours = cv.findContours(img_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
contours = imutils.grab_contours(contours)

# For each possible contour region detected extract the minimal bounding rectangle area
rects = []
# Debug image thats shows image processing steps
img_rects = input.copy()
for i,contour in enumerate(contours):
    minRec = cv.minAreaRect(contour)
    if isPlateSize(minRec) == True:
        # Draw contours in region where there could be a license plate
        cv.drawContours(img_rects, [contour], -1, (255, 0, 0), 1)
        rects.append(minRec)

# Display contours
displayImage(img_rects)
for rect in rects:

    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(img_rects,[box],-1,(0, 0, 255),2)
# Display minimum rectangle areas
displayImage(img_rects)

# Create new image to what will be the final processed image
output = input.copy()

# Define the accepted ranges for blue color
low_blue = np.array([50,140,100])
high_blue = np.array([130,255,255])

for i, rect in enumerate(rects):
    mask = np.zeros(input.shape[:2], dtype='uint8')
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(mask, [box], 0, 255, cv.FILLED)
    mask = cv.bitwise_and(input, input, mask = mask)
    hsv_mask = cv.cvtColor(mask, cv.COLOR_BGR2HSV)
    # Check how much blue is in the image
    blue_mask = cv.inRange(hsv_mask, low_blue, high_blue)
    total_blue = sum(sum(blue_mask))
    if total_blue > 2000 and total_blue < 4000:
        cv.drawContours(output,[box],-1,(0, 0, 255),2)
        img_mask = cv.bitwise_and(input, input, mask = blue_mask)
        # Display masked blue area in license plate
        displayImage(img_mask)

displayImage(output)
