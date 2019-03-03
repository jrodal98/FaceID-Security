# coding: utf-8
import cv2
# import os module for reading training data directories and paths
import os

import numpy as np
from PIL import Image

subjects = ["Unidentified", "Jake Rodal", "Kevin Melloy", "Kaan Katircioglu", "Harun Feraidon"]
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("model")


def detect_faces(img):
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    # face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    grays = [gray[y:y+w, x:x+h] for (x, y, w, h) in faces]

    return grays, faces


def predict(test_img):
    # make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    # detect face from the image
    faces, rects = detect_faces(img)
    # predict the image using our face recognizer
    for i in range(len(faces)):
        face = faces[i]
        rect = rects[i]
        label, confidence = face_recognizer.predict(face)
        # get name of respective label returned by face recognizer
        print(confidence)
        if confidence > 50:
            label = 0
        label_text = subjects[label]

        # draw a rectangle around face detected
        draw_rectangle(img, rect)
        # draw name of predicted person
        draw_text(img, label_text, rect[0], rect[1]-5)

    return img


cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
# vc.set(cv2.CAP_PROP_FPS, 240)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    img = Image.fromarray(frame, 'RGB')
    img.save('tmp.png')
    try:
        predicted_img = predict(cv2.imread("tmp.png"))
        cv2.imshow("preview", predicted_img)
    except TypeError:
        cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()
