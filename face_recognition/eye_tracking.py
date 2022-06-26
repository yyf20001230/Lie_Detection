import cv2
import cvzone
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
from fer import FER
from util import getDominantEmotion, getEyeImage, blob_process, eyeRatio, calibration

#capture video from webcam
cap = cv2.VideoCapture(1)

#face detector
detector = FaceMeshDetector(maxFaces = 1)

#emotional detector
emo_detector = FER(mtcnn = True)

plotY = LivePlot(640, 360, [20, 50], invert = True)
plotFace = LivePlot(640, 360, [20,50], invert = True)

#additional eye contraint = 144, 145, top eyebrow = 27, 28, 29
idListLeft = [22, 23, 24, 27, 28, 29, 155, 157, 158, 159, 160, 130, 243]  

blinkCounter = 0
calibList = []
eyeRatio_avg = 35
eyeArea_avg = 15000
counter = 0
threshold = 55
captured_emotions = []
dominant_emotion = ""

while True:
     
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw = False)

    if faces:

        face = faces[0]

        #for id in idListLeft:
        #    cv2.circle(img, face[id], 5, (255, 0, 255), cv2.FILLED)

        #detecting face orientation
        #cv2.circle(img, face[152], 5, (255, 0, 255), cv2.FILLED)   #top
        #cv2.circle(img, face[10], 5, (255, 0, 255), cv2.FILLED)    #down
        #cv2.circle(img, face[4], 5, (255, 0, 255), cv2.FILLED)     #mid


        #eye tracking
        minY = float('inf')
        maxY = 0
        minX = float('inf')
        maxX = 0
        for id in idListLeft:
            if face[id][1] < minY:
                minY = face[id][1]
            if face[id][1] > maxY:
                maxY = face[id][1]
            if face[id][0] < minX:
                minX = face[id][0]
            if face[id][0] > maxX:
                maxX = face[id][0]

        eyeImg = getEyeImage(img, minX, minY, maxX, maxY)
        eyeArea = 0
        if eyeImg.shape[0] > 0 and eyeImg.shape[1] > 0:
            eyeImg = cv2.resize(eyeImg, (640, 360))
            _, contour = blob_process(eyeImg, threshold)
            if len(contour) != 0:
                eyeArea = cv2.contourArea(contour[0])
                cvzone.putTextRect(eyeImg, f'EyeArea: {eyeArea}', (50,100))
            for cnt in contour:
                cv2.drawContours(eyeImg, [cnt], -1, (0, 0, 255), 3)

        dist = detector.findDistance(face[152], face[10])[0]
        ratio = eyeRatio(face, detector)

        if ratio < eyeRatio_avg * 0.95 and eyeArea < eyeArea_avg * 0.9:
            counter += 1
        else:
            if counter != 0:
                blinkCounter += 1
            elif counter == 0 and eyeImg.shape[0] > 0 and eyeImg.shape[1] > 0:
                
                #perform calibration on eyeRatio and eyeArea when eye is not blinking
                threshold_frame = calibration(cv2.resize(getEyeImage(img, minX, minY, maxX, maxY), (640, 360)))
                calibList.append([ratio, threshold_frame, eyeArea])
                if len(calibList) > 15:
                    calibList.pop(0)
                    eyeRatio_avg = np.sum(calibList, axis = 0)[0] / len(calibList)
                    threshold = np.sum(calibList, axis = 0)[1] / len(calibList)
                    eyeArea_avg = np.sum(calibList, axis = 0)[2] / len(calibList)
                    print(threshold)

            counter = 0

        
        #if cap.get(cv2.CAP_PROP_POS_FRAMES) % 60 == 0:
        #    dominant_emotion, captured_emotions, captured_emotions_list = getDominantEmotion(img, emo_detector)    

        cvzone.putTextRect(img, f'Blink count: {blinkCounter}', (50,100))
        if dominant_emotion != "" and captured_emotions != []:
            cvzone.putTextRect(img, f'Dominate expression: {dominant_emotion}', (50,150))
            for emotion in captured_emotions_list:
                cvzone.putTextRect(img, f'{emotion}: {captured_emotions[emotion]}', (50, 200 + captured_emotions_list.index(emotion) * 50))

        imgPlot = plotY.update(ratio)
        facePlot = plotFace.update(eyeRatio_avg)
        img = cv2.resize(img, (640, 360))

        if eyeImg.shape[0] > 0 and eyeImg.shape[1] > 0:
            imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
            imgStack2 = cvzone.stackImages([eyeImg, facePlot], 2, 1)
            imgStack = cvzone.stackImages([imgStack, imgStack2], 1, 1)
        else:
            imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
        
    else:
        imgStack = cvzone.stackImages([img, img], 2, 1) 

    imgStack = cv2.resize(imgStack, (2560, 1440))
    cv2.imshow("Img", imgStack)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    

    

    

