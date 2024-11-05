import cv2
import numpy as np
import dlib
import imutils
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from deepface import DeepFace
from faceDetection import *
from eyeTracking import *
from stressDetection import *
 
if __name__ == "__main__":
    # Initialise video capturing parameters
    cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,2)
    detector=dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor("C:/GOWDATA/AmritaCollege/SIH2023/Updates/Proctoring/Resources/shape_predictor_68_face_landmarks.dat")
    silhouetteImg,height,width=silhouette()
    
    # Read video input stream
    while cap.isOpened():
        try:
            _,frame=cap.read()
            h,w,_=frame.shape
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=detector(gray)
            if not len(faces):
                # If the candidate is not visible on the screen
                cv2.putText(frame, "No Face, Please be in the frame ", (5, 20), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,0,0), 1)
            if len(faces)>1:
                # If more than one person is visible on the screen
                cv2.putText(frame, "WARNING!! multiple faces detected ", (5, 20), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,0,0), 1)
            else:
                dims=[height,width,h,w]
                try:
                    # Stress detection
                    frame=candidate_emotion(frame)
                    # Eye Tracking
                    frame=candidate_eye(predictor,frame,gray,faces[0],dims)
                    frame=cvzone.overlayPNG(frame,silhouetteImg,[50,50])
                    # Position Tracking
                    frame=candidate_position(predictor,frame,gray,faces[0],dims)
                except:
                    continue
            cv2.imshow("Frame",frame)
            #press esc to exit the webcam
            if cv2.waitKey(1)==27:
                break
        except KeyboardInterrupt:
            break
    cap.release()
    cv2.destroyAllWindows()