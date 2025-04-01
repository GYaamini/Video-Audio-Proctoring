# Computer Vision-Powered Audio/Video Monitoring for Remote Exams
The Audio and Video Proctoring modules were created as a part of proctored exam tool. 
Basic face and body positioning along with eye tracking, stress detection and audio detection are implemented using computer vision techniques.

---
## 🎯 Motivation
Existing proctoring tools fail to address:

- False positives (e.g., stress ≠ cheating).

- Genuine signs of anxiety.

- External help (e.g., keyword whispers, background voices).

This project tackles these gaps with context-aware monitoring.

---
## 📷 Sample Output 
Stressed | Looking Away | Unfocused
:-------------------------:|:-------------------------:|:-------------------------:
![Sample (1)](https://github.com/user-attachments/assets/3e37f530-e2db-46ef-b829-9a1d859a3302) | ![Sample (2)](https://github.com/user-attachments/assets/cae6a018-de60-4613-8493-deafab4ebd91) | ![Sample (3)](https://github.com/user-attachments/assets/e5ea760d-cef6-4709-a2a5-27fc22f24a01)

---
## ✨ Key Features
Module | Technology Used |	Detection Capability
:-------------------------:|:-------------------------:|:-------------------------:
Face Tracking |	OpenCV, Dlib |	Head position, Face count
Eye Tracking |	OpenCV |	Off-screen glances, Open/Closed eye
Stress Detection |	Deepface | Excessive stress, fear, or anger indications
Audio Analysis |	PyAudio |	Notable Keywords like Google, Code, etc

---
## 🛠️ Technical Implementation
**1. Face & Body Positioning Face Detector with `shape_predictor_68_face_landmarks`**

    Violation Triggers:
    - Multiple faces detected
    - Candidate Face positioned beyond silhouette boundary

**2. Eye Tracking
Gaze Vector Calculation: Pupil localization with `dlib.shape_predictor`**

    Cheating Signals:
    - Wandering eyes left or right (closed eyes is ignored as to be considerate of people who think with their eyes closed)

**3. Stress Detection**

    - Deepface stress score on Sad, Anger, Fear, and Disgust exceeding 90

**4. Audio Proctoring Background Speech Detection:**

    Warning Triggers:
    - Any helping words: Google, Find, Code, Answer, Question, Internet, Search, How, What, Quick, Hurry, Now

---