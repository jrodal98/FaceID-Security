# pip3 install opencv-python

import cv2
from PIL import Image

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
# cv2.set(cv2.CAP_PROP_FPS, fps)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

i = 1
while rval and i < 551:
    if (i > 50):
        img = Image.fromarray(frame, 'RGB')
        img.save(f'training-data/s1/{i-50}.png')
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
    i += 1

cv2.destroyWindow("preview")
vc.release()
