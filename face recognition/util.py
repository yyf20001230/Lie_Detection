import cv2
import numpy as np
from fer import FER

#get dominant emotion from captured img
def getDominantEmotion(img, emo_detector):
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

#get eye image
def getEyeImage(img, minX, minY, maxX, maxY):
    #return minY - 10 to ensure that entire eye is captured
    return img[minY - 10 : maxY, minX : maxX]


#get eyeball
def blob_process(img, threshold):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3), np.uint8)
    img = cv2.bilateralFilter(img, 10, 15, 15)
    img = cv2.erode(img, kernel, iterations=3)
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy  = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours  = sorted(contours, key = cv2.contourArea, reverse = True)
    return img, contours

#get eye ratio
def eyeRatio(face, detector):
    return (detector.findDistance(face[158], face[22])[0] + detector.findDistance(face[160], face[24])[0]) / detector.findDistance(face[130], face[243])[0] / 2 * 100

#calibrate the best threshold
def calibration(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3), np.uint8)
    img = cv2.bilateralFilter(img, 10, 15, 15)
    img = cv2.erode(img, kernel, iterations=3)

    threshold_window = []
    
    #return the threshold that matches the best with the true pupil pixel ratio
    for i in range(20, 70, 5):
        _, new_img = cv2.threshold(img, i, 255, cv2.THRESH_BINARY_INV)
        contours, _  = cv2.findContours(new_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours  = sorted(contours, key = cv2.contourArea, reverse = True)
        if len(contours) > 0:
            threshold_window.append(abs(cv2.contourArea(contours[0]) - 20000))
        else:
            threshold_window.append(15000)
    return threshold_window.index(min(threshold_window)) * 5 + 20
