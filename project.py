import cv2 as cv

path = r'C:\Users\Caleb\Documents\ECE_532\Project\2715DTZ.jpg'
input = cv.imread(path)

# Convert image into grayscale image as color can't help us
img_gray = cv.cvtColor(input, cv.COLOR_BGR2GRAY)

# Apply a Gaussian blur of 5x5 to remove vertical edges that are produced from noise
img_blur = cv.blur(img_gray, (5, 5))

# Find vertical edges using Sobel filter and first horizontal derivative
img_sobel = cv.Sobel(img_blur, cv.CV_8U, 1, 0, ksize=3, scale=1, delta=0)


# Apply threshold filter to obtain binary image (threshold value from Otsu's method)
img_binary = cv.threshold(img_sobel, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)[1]

cv.imshow('Binary', img_binary) 
cv.waitKey(0)
# Apply close morphological opertaion to remove blank spaces between each vertical edge
# and connect all regions that have high number of edges
element = cv.getStructuringElement(cv.MORPH_RECT,(17,3))

img_morph =  cv.morphologyEx(img_binary, cv.MORPH_CLOSE, element)

cv.imshow('Morph', img_morph) 
cv.waitKey(0)
# After applying these functions, we have regions in the image that could contain a plate
# to determine. To find which are actually plates we use connected-compenent analysis
# or find countours approach. Once countours are found we are only intereseted in external
# contours (hierarchical relationship and polygonal approximation)

# For each contour detected, extract the bounding rectangle of minimal area

# We make basic validations about the regions detected based on its area and aspect ratio

# We can make more improvements using the license plate's white background property. 
# we can use a flood fill algorithm to retrieve the rotated rectangle for precise cropping