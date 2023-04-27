import cv2 as cv
import numpy as np
import random
import time

def verifySizes(rect):
    (x, y), (width, height), angle = rect
    error=0.4
    #Spain car plate size: 52x11 aspect 4,7272
    aspect=4.7272
    #Set a min and max area. All other patchs are discarded
    min= 15*aspect*15; # minimum area
    max= 125*aspect*125; # maximum area
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

# path = '2715DTZ.jpg'
path = '3028BYS.JPG'
input = cv.imread(path)
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

# After applying these functions, we have regions in the image that could contain a plate
# to determine. To find which are actually plates we use connected-compenent analysis
# or find countours approach. Once countours are found we are only intereseted in external
# contours (hierarchical relationship and polygonal approximation)
contours, hierarchy = cv.findContours(img_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) [-2:]
contours = list(contours)

# Start to iterate to each contour founded
# Remove patch that are not inside limits of aspect ratio and area.
rects = []
for i,contour in enumerate(contours):
    # Create bounding rect of object
    minRec = cv.minAreaRect(contour)
    if verifySizes(minRec) != True:
        contours.pop(i)
    else:
        rects.append(minRec)

# Draw blue contours on a white image
result = input.copy()
cv.drawContours(result, contours, -1, (255, 0, 0), 1)
# cv.imshow("Blue Contour Result", result)
# cv.waitKey(0)

for rect in rects:
    # box = cv.boxPoints(rect)
    # box = np.int0(box)
    # test = cv.drawContours(result,[box],-1,(0, 0, 255),2)
    # cv.imshow("rect", result)
    # cv.waitKey(0)

    (x, y), (width, height), angle = rect
    center = (int(x),int(y))
    # For better rect cropping for each posible box
    # Make floodfill algorithm because the plate has white background
    # And then we can retrieve more clearly the contour box
    result = cv.circle(result, center, 3, (0,255,0), -1); 
    # get the min size between width and height
    if width < height:
        minSize = width
    else: 
        minSize = height
    # print(minSize)
    minSize = minSize - minSize * 0.5
    # initialize rand and get 5 points around center for floodfill algorithm
    random.seed(time.time())
    # Initialize floodfill parameters and variables
    mask = np.zeros((rows+2, columns+2), dtype = "uint8")
    # mask.fill(255)
    loDiff = 30
    upDiff = 30
    connectivity = 4
    newMaskVal = 255
    NumSeeds = 10
    flags = connectivity + (newMaskVal << 8 ) + cv.FLOODFILL_FIXED_RANGE + cv.FLOODFILL_MASK_ONLY
    for _ in range(NumSeeds):
        seed_x = center[0] + random.randint(0,32767) % int(minSize-(minSize/2))
        seed_y = center[1] + random.randint(0,32767) % int(minSize-(minSize/2))
        result = cv.circle(result, (seed_x, seed_y), 1, (0,255,255), -1)
        retval, input, mask, rect = cv.floodFill(input, mask, (seed_x, seed_y), (255,0,0), (loDiff, loDiff, loDiff), (upDiff, upDiff, upDiff), flags)

    # cv.imshow("Mask", mask)
    # cv.waitKey(0)

    #Check new floodfill mask match for a correct patch.
    #Get all points detected for get Minimal rotated Rect
    pointsOfInterest = []
    for x, row in enumerate(mask):
        for y, pointValue in enumerate(row):
            if pointValue==255:
                pointsOfInterest.append([x,y])

    pointsOfInterest = np.array(pointsOfInterest)
    minRect = cv.minAreaRect(pointsOfInterest)
    (x, y), (width, height), angle = minRect
    # print(minRec)
    if verifySizes(minRect):
        # rotated rectangle drawing 
        rect_points = cv.boxPoints(minRect)
        # print(rect_points)
        for pnt in range(4):
            result = cv.line(result, rect_points[pnt].astype(int), rect_points[(pnt+1)%4].astype(int), (0,0,255), 1, 8)

        #Get rotation matrix
        r = width / height
        if r < 1:
            angle = 90 + angle
        rotmat= cv.getRotationMatrix2D((x, y), angle,1)

        # Create and rotate image
        img_rotated = cv.warpAffine(input, rotmat, (int(width), int(height)), cv.INTER_CUBIC);

        # Crop image
        if r < 1:
            tmp = width
            width = height
            height = tmp

        img_crop = cv.getRectSubPix(img_rotated, (int(width), int(height)), (x, y))
        
        resultResized = np.zeros((33, 144, 3), dtype = "uint8")
        print(resultResized.shape[1])
        resultResized = cv.resize(img_crop, (resultResized.shape[0], resultResized.shape[1]), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
        # Equalize croped image
        grayResult = cv.cvtColor(resultResized, cv.COLOR_BGR2GRAY)
        grayResult = cv.blur(grayResult, (3,3))
        grayResult = cv.equalizeHist(grayResult)
        # if(saveRegions){ 
        #     stringstream ss(stringstream::in | stringstream::out);
        #     ss << "tmp/" << filename << "_" << i << ".jpg";
        #     imwrite(ss.str(), grayResult);
        # }
        # output.push_back(Plate(grayResult,minRect.boundingRect()))
        cv.imshow("grayResult", grayResult)
        cv.waitKey(0)

cv.imshow("Contours", result)
cv.waitKey(0)
# For each contour detected, extract the bounding rectangle of minimal area

# We make basic validations about the regions detected based on its area and aspect ratio

# We can make more improvements using the license plate's white background property. 
# we can use a flood fill algorithm to retrieve the rotated rectangle for precise cropping

