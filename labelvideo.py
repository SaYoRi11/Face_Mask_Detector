#Script to detect faces in video stream and classify them as with_mask, without_mask or mask_weared_incorrect
#Using mtcnn to detect faces because it can detect faces better than haarcascade

import cv2
from mtcnn import MTCNN
from fastbook import *
from fastai.vision.all import *

detector = MTCNN()
learn = load_pickle('learn.pkl')

cap = cv2.VideoCapture(0)

if cap.isOpened() == False:
    print('Error opening the video stream')

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        faces = detector.detect_faces(frame)

        for face in faces:
            x,y,w,h = face['box']

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
            
            faceImg = frame[y:y+h, x:x+w]
            pred = learn.predict(faceImg)[0]
            
            org = (x-10, y-10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.4
            color = (102, 0, 0)
            thickness = 1

            labelledFrame = cv2.putText(frame, pred, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
            cv2.imshow('Frame', labelledFrame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()