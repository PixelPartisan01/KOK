import cv2
import keras.models
import numpy as np
import tensorflow as tf
import numpy
import os

TRAIN = "Images\\Training"
VALIDATION = "Images\\Validation"
HEIGHT = 150
WIDTH = 150

def generate_model():
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    epo = 50
    batch_size = 70

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator \
        (rescale = 1/255.0,
         rotation_range = 45,
         height_shift_range = 0.2,
         shear_range = 0.2,
         zoom_range = 0.2,
         validation_split = 0.2,
         horizontal_flip = True)

    train_data = train_datagen.flow_from_directory \
        (directory = TRAIN,
         target_size = (HEIGHT, WIDTH),
         class_mode = "categorical",
         batch_size = batch_size,
         subset = "training")

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)

    val_data = train_datagen.flow_from_directory \
        (directory=TRAIN,
         target_size=(HEIGHT, WIDTH),
         class_mode="categorical",
         batch_size = batch_size,
         subset="validation")

    mobilenet = tf.keras.applications.mobilenet_v2.MobileNetV2 \
        (weights="imagenet",
         include_top=False,
         input_shape=(HEIGHT, WIDTH, 3))

    for l in mobilenet.layers:
        l.trainable = False

    model = tf.keras.models.Sequential()
    model.add(mobilenet)
    model.add(tf.keras.layers.Dense(64, activation="relu"))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2, activation="softmax"))  # megegyzeik az osztályok darabszámával

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint \
        ("Gender.h5",
         monitor="val_accuracy",
         save_best_only=True,
         verbose=1)

#    earlystop = tf.keras.callbacks.EarlyStopping \
#        (monitor="val_accuracy",
#        patience= epo // 3,
#         verbose=1)

    history = model.fit \
        (train_data,
         steps_per_epoch=len(train_data) // batch_size,
         epochs= epo,
         validation_data=val_data,
         validation_steps=len(val_data) // batch_size,
        # callbacks=[checkpoint, earlystop],
         callbacks=[checkpoint],
         verbose=1)

    model.evaluate(val_data)
    return model

def load_model():
    model = keras.models.load_model("Gender.h5")
    return model

def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace = cv2.FONT_HERSHEY_TRIPLEX, fontScale = scale / 10, thickness = 1)
        new_width = textSize[0][0]
        if new_width <= width:
            return scale/15

    return 1/new_width

def determine_gender(img, x, y, w, h, model):
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (150, 150))
    scaled = face/255.0
    reshape = numpy.reshape(scaled, (1, 150, 150, 3))
    img_vs = np.vstack([reshape])
    pred = model.predict(img_vs)
    font_scale = 3 * (img.shape[1] // 6)

    print(pred)

    if pred[0][0] > pred[0][1]:
        t = "FEMALE [{:0.2f}%]".format((pred[0][0]) * 100) if pred[0][0] >= .6 else "FEMALE {{?}} [{:0.2f}%]".format((pred[0][0]) * 100)
        font_size = get_optimal_font_scale(t, font_scale)
        cv2.putText(img, t, (x,y-10), cv2.FONT_HERSHEY_TRIPLEX, font_size/2,(69, 47, 235), 1)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif pred[0][1] > pred[0][0]:
        t = "MALE [{:0.2f}%]".format((pred[0][1]) * 100) if pred[0][1] >= .6 else "MALE {{?}} [{:0.2f}%]".format((pred[0][1]) * 100)
        font_size = get_optimal_font_scale(t, font_scale)
        cv2.putText(img, t, (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, font_size/2, (235, 72, 47), 1)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)