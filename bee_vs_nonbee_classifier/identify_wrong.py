import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from shutil import copy
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout, GlobalAveragePooling2D

tf.compat.v1.disable_eager_execution()


# own_model = load_model('./saved-model-50-0.88.h5')

# with open('saved-model-50-0.88_config.json') as json_file:
#     json_config = json_file.read()
# own_model = model_from_json(json_config)
# Load weights
# own_model.load_weights('saved-model-50-0.88_weights.h5')

model = load_model('models/freezed-model-70-0.85_BEST.h5')


path = "./cam_images"
datagen = ImageDataGenerator(rescale=1.0/255.0)
batch_size, target_size, class_mode = 8, (224, 224), 'binary'

right_count = 0
wrong_count = 0

val_it = datagen.flow_from_directory(
    path + '/',
	class_mode=class_mode,
    batch_size=batch_size,
    target_size=target_size,
    shuffle=False)
filenames = val_it.filenames

Y_pred = model.predict(val_it)
y_pred = [0 if pred[0] > pred[1] else 1 for pred in Y_pred]


for ind, filename in enumerate(filenames):
    print("{} -- {}".format(filename, y_pred[ind]))
	# testing bee

    if y_pred[ind] == val_it.labels[ind]:

        right_count += 1

    	# For mimic bee
        # copy("dataset/input_for_prediction/{}".format(filename), "dataset/output_for_prediction/correct/")


    else:
        wrong_count += 1

        # if val_it.labels[ind] == 0:
        #     copy("dataset/val/{}".format(filename), "dataset/wrong_pred/bee/")
        # else:
        #     copy("dataset/val/{}".format(filename), "dataset/wrong_pred/nobee/")


    	# For mimic bee
        # copy("dataset/input_for_prediction/{}".format(filename), "dataset/output_for_prediction/wrong/")


print("Right count: {}, Wrong count: {}, Accuracy: {:.6f}".format(
    right_count,
    wrong_count,
    (100*right_count)/(right_count+wrong_count)
))
