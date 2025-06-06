import os
from tkinter import Image

os.environ["TF_ENABLE_ONEDNN_OPTS"] = 0
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
TRAIN_DATA_DIR = "./train"
VALIDATION_DATA_DIR = "./valid"
TRAIN_SAMPLES = 500
VALIDATION_SAMPLES = 500
NUM_CLASSES = 2
IMG_WIDTH,IMG_HIGHT=224
BATCH_SIZE = 64
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
train_datagen = image.ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range = 0.2
)
val_datagen = Image.ImageDataGenerator(preprocessing_function = preprocess_input)
train_genetarot = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size = (224,224),
    batch_size = BATCH_SIZE,
    shuffle = false,
    class_mode = "categorical"
)
val_genetarot = train_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size = (224,224),
    batch_size = BATCH_SIZE,
    shuffle = false,
    class_mode = "categorical"
)

from tensorflow.keras.layers import (Input, Flatten, Dense, Dropout, GlobalAveragePooling2D)
from tensorflow.keras.models import Model
def model_maker():
    base_model = MobileNet(include_top = False, input_shape = (244,244))
    for layer in base_model.layers[:]:
        layer.trainable = False
    input = Input(shape = (244,244,3))
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling2D()(custom_model)
    custom_model = Dense(64,activation = "relu")(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    prediction = Dense(NUM_CLASSES, activation = "softmax")(custom_model)
    return Model(inputs = input, output=prediction)

model = model_maker()
import math
num_steps = math.ceil(float(TRAIN_SAMPLES)/BATCH_SIZE)
from tensorflow.keras.optimizers import Adam
model.compile(loss="categorical_crossentry"
              optimizer = "adam"
              metrics=["acc"])
model.fit(train_genetarot, steps_per_epoch = num_steps, epochs = 10, validation_data = val_generator, validation_steps = num_steps)
img = image.load_img("picture.png",target_size=(224,224))
import numpy as np
img_array = image.img_to_array(img)
img_batch=np.expand_dims(img_array, axis=0)
from tensorflow.keras.applications.resnet50 import preprocessing_input

plt.imshow(img)
plt.plot()

img_processed = preprocess_input(img_batch)
from tensorflow.keras.applications.resnet50 import ResNet50

model = ResNet50()
prediction = model.predict(img_processed)

#nasha neuronka
#test modeli
from keras.models import load_model
model = load_model("./model")
prediction = model.predict(img_processed)
print(prediction)

