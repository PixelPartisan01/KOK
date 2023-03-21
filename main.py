# SOURCE: https://www.kaggle.com/code/taha07/gender-classification-using-opencv/notebook
# SOURCE: https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0
# SOURCE: https://github.com/opencv/opencv_zoo/blob/master/models/face_detection_yunet/demo.py

from face import *
import tkinter
from tkinter import filedialog

def main():
    if not os.path.isfile("Gender.h5"):
        print("Generate model...\n\n\n")
        model = generate_model()

    else:
        print("Model already exist!\n\n\n")
        model = load_model()

    root = tkinter.Tk()
    root.withdraw()

    file_extensions = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    file_path = filedialog.askopenfilename(initialdir= "./", title= "Select Image", filetypes= [(file_extensions, file_extensions)], multiple = True)
    find_face(model, file_path)

    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if __name__ == "__main__":
    main()

