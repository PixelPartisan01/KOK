from gender import *
import cv2
import numpy

TRAIN = "Images\\Humans\\Training"
VALIDATION = "Images\\Humans\\Validation"
HEIGHT = 150
WIDTH = 150

def find_face(model, files):
    i = "test_22.jpg"

    for n in files:
        img = cv2.imread(n)
        image_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        front_face_cas = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
        front_faces = front_face_cas.detectMultiScale(image_g, 1.06, 8)

        for (x, y, w, h) in front_faces:
            cv2.rectangle(img, (x, y), (x + w + 2, y + h + 2), (107, 235, 52), 2)

            determine_gender(img,x ,y, w, h, model)

        cv2.namedWindow(n.split('/')[-1], cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(n.split('/')[-1], cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(n.split('/')[-1], img)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
