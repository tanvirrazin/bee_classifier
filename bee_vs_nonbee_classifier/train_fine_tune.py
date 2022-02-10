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
path = "./dataset/"

train_it = datagen.flow_from_directory(path + 'train_color_gray/',
	batch_size=batch_size, target_size=target_size)
val_it = datagen.flow_from_directory(path + 'val_color_gray/',
	batch_size=batch_size, target_size=target_size)

model = load_model('models_color_gray/freezed-model-30-0.82.h5')

# Unfreeze model
model.trainable = True
try:
  for layer in model.layers:
    layer.trainable = True
except: print('Unfreezing')


model.compile(
	optimizer=Adam(learning_rate=0.00001),
	loss='binary_crossentropy',
	metrics=['accuracy'])

checkpoint = ModelCheckpoint(
        "models_color_gray/unfreezed-saved-model-{epoch:02d}-{val_acc:.2f}.h5",
        monitor='val_acc',
        verbose=1,
        save_best_only=False,
        mode='max',
        period=5
)

history = model.fit(
	train_it,
	steps_per_epoch=len(train_it),
	validation_data=val_it,
	validation_steps=len(val_it),
	epochs=number_of_epochs,
        callbacks=[checkpoint],
	verbose=1)

model.save('models_color_gray/model_unfreezed.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.savefig('images/accuracies_unfreezed.png')
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('images/loss_unfreezed.png')
