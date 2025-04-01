import cv2
import numpy as np
import dlib
import math

pos=["Right","Left"]

def faceLandmakDetector(predictor,gray,faces):
    # landmarks predictor with 68 landmarks
    landmarks = predictor(gray, faces)
    pointList = []
    for n in range(0, 68):
        point = (landmarks.part(n).x, landmarks.part(n).y)
        # getting x and y coordinates of each mark and adding into list
        pointList.append(point)
    return pointList

def EyeTracking(gray, eyePoints):
    dim = gray.shape
    mask = np.zeros(dim, dtype=np.uint8)
    # converting eyePoints into Numpy arrays
    PollyPoints = np.array(eyePoints, dtype=np.int32)
    cv2.fillPoly(mask, [PollyPoints], 255) # Fill the eyes portion with white

    # gray image where color is white in the mask
    eyeImage = cv2.bitwise_and(gray, gray, mask=mask)

    # getting the max and min points of eye
    maxX = (max(eyePoints, key=lambda item: item[0]))[0]
    minX = (min(eyePoints, key=lambda item: item[0]))[0]
    maxY = (max(eyePoints, key=lambda item: item[1]))[1]
    minY = (min(eyePoints, key=lambda item: item[1]))[1]

    # other then eye area will black, making it white
    eyeImage[mask == 0] = 255

    # cropping the eye form eyeImage.
    cropedEye = eyeImage[minY:maxY, minX:maxX]

    # getting width and height of cropedEye
    height, width = cropedEye.shape

    #dividing the eye into 4 parts
    divPart = int(width/3)

    #  applying the threshold to the eye 
    ret, thresholdEye = cv2.threshold(cropedEye, 100, 255, cv2.THRESH_BINARY)
    # getting threshold of left, right and center parts based on 4 divisions
    rightPart = thresholdEye[0:height, 0:int(divPart*1)]
    centerPart = thresholdEye[0:height, int(divPart*0.5):int(divPart*2.5)]
    leftPart = thresholdEye[0:height, int(divPart*2):width]

    # counting Black pixel in each part using numpy
    rightBlackPx = np.sum(rightPart == 0)
    centerBlackPx = np.sum(centerPart == 0)
    leftBlackPx = np.sum(leftPart == 0)
    pos= Position([rightBlackPx, centerBlackPx, leftBlackPx])

    return mask, pos


def Position(ValuesList):
    maxIndex = ValuesList.index(max(ValuesList))
    posEye = ''
    # Eye position based on max value in the list
    if maxIndex == 0:
        posEye = "Right"
    elif maxIndex == 1:
        posEye = "Center"
    elif maxIndex == 2:
        posEye = "Left"
    else:
        posEye = "Eye Closed"
    return posEye

def candidate_eye(predictor,frame,gray,face,dims):
    PointList = faceLandmakDetector(predictor,gray,face)
    RightEyePoint = PointList[36:42]
    LeftEyePoint = PointList[42:48]

    maskRight, rightPos= EyeTracking(gray, RightEyePoint)
    maskLeft, leftPos= EyeTracking(gray, LeftEyePoint)

    # Candidates wandering eyes left or right will be warned
    # Closed eyes and eyes in the center are ignored since few people think while momentarily closing their eyes
    if rightPos in pos or leftPos in pos:
        return True