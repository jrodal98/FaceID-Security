# pip3 install opencv-python

import cv2

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
# cv2.set(cv2.CAP_PROP_FPS, fps)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    im = cv2.imread("test_output.jpg")
    cv2.imshow("preview", im)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()
