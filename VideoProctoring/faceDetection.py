import cv2
import numpy as np
import dlib
import cvzone

def silhouette():
    #Load the silhouette image for positioning the candidate
    silhouetteImg = cv2.imread("Resources/silhouette.png",cv2.IMREAD_UNCHANGED)
    silhouetteImg=cv2.resize(silhouetteImg,(0,0),None,0.3,0.3)
    height,width,_=silhouetteImg.shape
    return silhouetteImg,height,width

def candidate_position(predictor,frame,gray,face,dims):
    h,w=dims[0],dims[1]
    height,width=silhouette()
    # Based on silhouette position, mark the boundary for candidate positioning
    x,y=face.left(),face.top()
    x1,y1=face.right(),face.bottom()
    faceSize=x1-x
    landmarks=predictor(gray,face)

    # if candidate position beyong boundary, send warning
    if x<((height//2)+70-faceSize) or x1>((height//2)+70+faceSize):
        return True
