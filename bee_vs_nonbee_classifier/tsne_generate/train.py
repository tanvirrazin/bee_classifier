import sys
from matplotlib import pyplot
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras import optimizers, Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, GlobalMaxPooling2D, ZeroPadding2D, Convolution2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math, datetime, time
from tensorflow.keras.models import load_model
import argparse


datagen = ImageDataGenerator(rescale=1.0/255.0)

batch_size, target_size, class_mode = 8, (224, 224), 'binary'
number_of_epochs = 1000
# prepare iterators
path = "./data/"

train_it = datagen.flow_from_directory(path + './',
	batch_size=batch_size, target_size=target_size)
val_it = datagen.flow_from_directory(path + './',
	batch_size=batch_size, target_size=target_size)


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False
try:
  for layer in base_model.layers:
      layer.trainable = False
except: print("Freezing")

x = GlobalAveragePooling2D()(base_model.output)
# x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(12, activation='softmax')(x)
base_model = Model(inputs=base_model.input, outputs=predictions)
base_model.compile(
	optimizer=Adam(learning_rate=0.0005),
	loss='categorical_crossentropy',
	metrics=['accuracy'])

checkpoint = ModelCheckpoint(
        "models/freezed-model-{epoch:02d}-{val_acc:.2f}.h5",
        monitor='val_acc',
        verbose=1,
        save_best_only=False,
        mode='max',
        period=5
)

print(base_model.summary())
print(len(train_it))

history = base_model.fit(
        train_it,
        steps_per_epoch=len(train_it),
        validation_data=val_it,
        validation_steps=len(val_it),
        epochs=number_of_epochs,
        callbacks=[checkpoint],
        verbose=1)

base_model.save('models/model_freezed.h5')
