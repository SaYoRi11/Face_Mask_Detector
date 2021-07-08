#Script to detect faces in input image and classify them as with_mask, without_mask or mask_weared_incorrect
#Using mtcnn to detect faces because it can detect faces better than haarcascade

import cv2
from mtcnn import MTCNN
from fastbook import *
from fastai.vision.all import *

imgUrl = 'enterurl'
img = cv2.cvtColor(cv2.imread(imgUrl), cv2.COLOR_BGR2RGB)
detector = MTCNN()
faces = detector.detect_faces(img)
learn = load_pickle('learn.pkl')

for face in faces:
  x,y,w,h = face['box']

  cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
  
  faceImg = img[y:y+h, x:x+w]
  pred = learn.predict(faceImg)[0]
  
  org = (x-10, y-10)
  font = cv2.FONT_HERSHEY_SIMPLEX
  fontScale = 0.4
  color = (102, 0, 0)
  thickness = 1

  labelledImg = cv2.putText(img, pred, org, font, fontScale, color, thickness, cv2.LINE_AA, False)

  cv2.imshow('Image', labelledImg)