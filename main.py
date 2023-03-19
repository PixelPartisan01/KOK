# TODO SOURCE: https://www.kaggle.com/code/taha07/gender-classification-using-opencv/notebook
# TODO SOURCE: https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0

import cv2
import keras.models
import numpy as np
import tensorflow as tf
import numpy
import os
from gender import *
from face import *


def main():
    if not os.path.isfile("Gender.h5"):
        print("Generate model...\n\n\n")
        model = generate_model()

    else:
        print("Model already exist!\n\n\n")
        model = load_model()

    find_face(model)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if __name__ == "__main__":
    main()

