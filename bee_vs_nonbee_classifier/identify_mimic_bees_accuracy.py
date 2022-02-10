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

model = load_model('models_color_gray/unfreezed-saved-model-220-0.92_BEST.h5')


path = "./mimic_input_images/"
datagen = ImageDataGenerator(rescale=1.0/255.0)
batch_size, target_size, class_mode = 8, (224, 224), 'binary'

for node in os.listdir(path):
    mimic_bee_dir = os.path.join(path, node)
    print('')
    print(mimic_bee_dir)
    if os.path.isdir(mimic_bee_dir):
        print('{}\n-------------'.format(node))
        right_count = 0
        wrong_count = 0

        mimic_bee_dir

        val_it = datagen.flow_from_directory(
            mimic_bee_dir,
        	class_mode=class_mode,
            batch_size=batch_size,
            target_size=target_size,
            shuffle=False)
        filenames = val_it.filenames

        Y_pred = model.predict(val_it)
        # print(Y_pred)
        y_pred = [0 if pred[0] > pred[1] else 1 for pred in Y_pred]


        for ind, filename in enumerate(filenames):
            # print("{} -- {}".format(filename, y_pred[ind]))
        	# testing bee

            if y_pred[ind] == val_it.labels[ind]:
                right_count += 1
            	# For mimic bee
                # copy("dataset/input_for_prediction/{}".format(filename), "dataset/output_for_prediction/correct/")

            else:
                wrong_count += 1

            	# For mimic bee
                # copy("dataset/input_for_prediction/{}".format(filename), "dataset/output_for_prediction/wrong/")


        print("Right count: {}, Wrong count: {}, Accuracy: {:.6f}%".format(
            right_count,
            wrong_count,
            (100*right_count)/(right_count+wrong_count)
        ))
