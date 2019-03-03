from gtts import gTTS

# coding: utf-8
import cv2
# import os module for reading training data directories and paths
import os

import numpy as np
from PIL import Image

subjects = ["Unidentified", "Kevin Melloy", "Kevin Melloy", "Kevin Melloy", "Kevin Melloy"]


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


def prepare_training_data(data_folder_path):

    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    for dir_name in dirs:

        # our subject directories start with letter 's' so
        # ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue

        label = int(dir_name.replace("s", ""))

        # build path of directory containin images for current subject subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # ------STEP-3--------
        # go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            # display an image window to show the image
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)

            # detect face
            face, rect = detect_faces(image)

            # ------STEP-4--------
            # for the purpose of this tutorial
            # we will ignore faces that are not detected
            if face is not None:
                # add face to list of faces

                faces.append(face[0])
                # add label for this face
                labels.append(label)
        # break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


# print("Preparing data...")
# faces, labels = prepare_training_data("training-data")
# print("Data prepared")
#
# #print total faces and labels
# print("Total faces: ", len(faces))
# print("Total labels: ", len(labels))
#
# x = cv2.face.LBPHFaceRecognizer_create()
# x.train(faces, np.array(labels))
#
# x.save("model")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("model")


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# function to draw text on give image starting from
# passed (x, y) coordinates.


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


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

    return img, subjects[label]

# Now that we have the prediction function well defined, next step is to actually call this function on our test images and display those test images to see if our face recognizer correctly recognized them. So let's do it. This is what we have been waiting for.

# In[10]:


# print("Predicting images...")

# load test images
# test_img1 = cv2.imread("test-data/test1.jpg")
# test_img2 = cv2.imread("test-data/test2.jpg")
#
# # perform a prediction
# predicted_img1 = predict(test_img1)
# predicted_img2 = predict(test_img2)
# print("Prediction complete")

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
# vc.set(cv2.CAP_PROP_FPS, 240)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

name = ""
last_seen = "dummy"
identified = []
while rval:
    img = Image.fromarray(frame, 'RGB')
    img.save('tmp.png')
    try:
        predicted_img, name = predict(cv2.imread("tmp.png"))
        if name != last_seen:
            last_seen = name
            identified = []
            identified.append(name)
            if name != "Unidentified":
                tts = gTTS(text=f"Welcome home, {name}", lang='en')
            else:
                tts = gTTS(text=f"Unidentified", lang='en')
            tts.save("pcvoice.mp3")
            # to start the file from python
            os.system("start pcvoice.mp3")

        cv2.imshow("preview", predicted_img)


    except TypeError:
        cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
    elif key == 13:
        tts = gTTS(text=f"Welcome home, {name}", lang='en')
        tts.save("pcvoice.mp3")
        # to start the file from python
        os.system("start pcvoice.mp3")

cv2.destroyWindow("preview")
vc.release()


# combined_image = cv2.imread("test-data/test2_photo.jpg")
# predicted_img = predict(combined_image)
# # display both images
# # cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
# # cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
# cv2.imshow("test", cv2.resize(predicted_img, (800, 1000)))
#
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)
# cv2.destroyAllWindows()
