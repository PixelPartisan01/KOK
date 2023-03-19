import cv2
import keras.models
import numpy as np
import tensorflow as tf
import numpy
import os
from gender import *

def find_face(model):
    i = "test_22.jpg"
    img = cv2.imread(i)
    image_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    front_face_cas = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    front_faces = front_face_cas.detectMultiScale(image_g, 1.06, 8)


    for (x, y, w, h) in front_faces:
        cv2.rectangle(img, (x, y), (x + w + 2, y + h + 2), (107, 235, 52), 2)
        roi_gray = image_g[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        determine_gender(img,x ,y, w, h, model)

    cv2.namedWindow(i, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(i, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(i, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()