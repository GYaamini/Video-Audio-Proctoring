import cv2
from deepface import DeepFace

# face_cascade=cv2.CascadeClassifier("/Proctoring/Resources/haarcascade_frontalface_default.xml")

def candidate_emotion(frame):
    # List of common mood/emotions displayed during an exam
    mood=["sad","fear","angry","disgust"]
    emotions=DeepFace.analyze(img_path = frame , actions=['emotion'], enforce_detection=False )
    score=[emotions[0]["emotion"]["sad"],emotions[0]["emotion"]["fear"],emotions[0]["emotion"]["angry"],emotions[0]["emotion"]["disgust"]]
    # Look for emotions with score >90.00 to be recognised as stress indicator
    stress=[True for ele in score if ele>90.00]
    if str(emotions[0]["dominant_emotion"]) in mood and len(stress)>0:
        # return frame with stress indication 
        cv2.putText(frame, "STRESSED", (5,200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,0,0), 1)
        return frame
    return