import cv2
import cvzone
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

#capture video from webcam
cap = cv2.VideoCapture(1)

#face detector
detector = FaceMeshDetector(maxFaces = 1)

#Live Plot
plotFace = LivePlot(640, 360, [300,500], invert = True)
plotTopDown = LivePlot(640, 360, [100,200], invert = True)

while True:
    
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw = True)
    
    if faces:
        face = faces[0]
        cv2.circle(img, face[152], 5, (255, 0, 255), cv2.FILLED) #down
        cv2.circle(img, face[10], 5, (255, 0, 255), cv2.FILLED) #top
        cv2.circle(img, face[4], 5, (255, 0, 255), cv2.FILLED) #mid
        cv2.circle(img, face[130], 5, (255, 0, 255), cv2.FILLED) #left eye
        cv2.circle(img, face[263], 5, (255, 0, 255), cv2.FILLED) #right eye
        cv2.circle(img, face[291], 5, (255, 0, 255), cv2.FILLED) #right mouth
        cv2.circle(img, face[61], 5, (255, 0, 255), cv2.FILLED) #left mouth
        
        cv2.circle(img, face[155], 5, (255, 0, 255), cv2.FILLED) #left eye right corner
        cv2.circle(img, face[362], 5, (255, 0, 255), cv2.FILLED) #right eye left corner

        for i in range(6,12):
            cv2.circle(img, face[12], 5, (255, 0, 255), cv2.FILLED)
        
        dist = detector.findDistance(face[152], face[10])[0]

        eyePlot = plotFace.update(dist)
        img = cv2.resize(img, (640, 360))
        img = cvzone.stackImages([img, eyePlot], 2, 1)
        
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
