import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
from fer import FER

cap = cv2.VideoCapture(1)
detector = FaceMeshDetector(maxFaces = 1)
emo_detector = FER(mtcnn = True)
plotY = LivePlot(640, 360, [20, 50], invert = True)
plotFace = LivePlot(640, 360, [300, 400], invert = True)


idList = [22, 24, 158, 160, 130, 243]
calibList = []
blinkCounter = 0
counter = 0
captured_emotions = []
dominant_emotion = ""

def getDominantEmotion(img):
    try:
        dominant_emotion, emotion_score = emo_detector.top_emotion(img)
        captured_emotions = emo_detector.detect_emotions(img)[0]["emotions"]
        captured_emotions_list = sorted(captured_emotions, key = captured_emotions.get, reverse = True)
        print(captured_emotions)
    except Exception as e:
        print(e)
        dominant_emotion = ""
        captured_emotions = ""
        captured_emotions_list = []
    
    return dominant_emotion, captured_emotions, captured_emotions_list


def eyeRatio(face):
    return (detector.findDistance(face[158], face[22])[0] + detector.findDistance(face[160], face[24])[0]) / detector.findDistance(face[130], face[243])[0] / 2 * 100

while True:
     
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw = True)

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 5, (255, 0, 255), cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]

        cv2.circle(img, face[152], 5, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, face[10], 5, (255, 0, 255), cv2.FILLED)
        dist = detector.findDistance(face[152], face[10])[0]

        cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

        #if cap.get(cv2.CAP_PROP_POS_FRAMES) % 60 == 0:
        #    dominant_emotion, captured_emotions, captured_emotions_list = getDominantEmotion(img)

        ratio = eyeRatio(face)
        calibList.append(ratio)
        if len(calibList) > 12:
            calibList.pop(0)
        calibAvg = sum(calibList) / len(calibList)
        if ratio < 35 and counter == 0:
            blinkCounter += 1
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0

        cvzone.putTextRect(img, f'Blink count: {blinkCounter}', (50,100))
        if dominant_emotion != "" and captured_emotions != []:
            cvzone.putTextRect(img, f'Dominate expression: {dominant_emotion}', (50,150))
            for emotion in captured_emotions_list:
                cvzone.putTextRect(img, f'{emotion}: {captured_emotions[emotion]}', (50, 200 + captured_emotions_list.index(emotion) * 50))

        imgPlot = plotY.update(ratio)
        facePlot = plotFace.update(dist)
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
        

    else:
        imgStack = cvzone.stackImages([img, img], 2, 1) 

    imgStack = cv2.resize(imgStack, (2560, 720))
    cv2.imshow("Img", imgStack)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    

    

    

