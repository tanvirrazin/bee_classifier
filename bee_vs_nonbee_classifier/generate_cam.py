import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input


tf.compat.v1.disable_eager_execution()

model = load_model('./models/freezed-model-70-0.85_BEST.h5')
# img_path = './image.jpg'

for f in os.listdir('./cam_images'):
    img_path = './cam_images/'+f
    clr_img = cv2.imread(img_path)
    gray = cv2.cvtColor(clr_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./cam_images/{}_gray.jpg".format(f.split('.')[0]), gray)

for f in os.listdir('./cam_images'):
    img_path = './cam_images/'+f
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    argmax = np.argmax(preds[0])
    output = model.output[:, argmax]
    print("{} -- {}".format(f, preds))

    last_conv_layer = model.get_layer('block5_conv3')
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])

    # for i in range(512):
    #     conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)


    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    hif = 0.8
    superimposed_img = heatmap * hif + img

    cv2.imwrite("./cam_images/{}_heatmap.jpg".format(f.split('.')[0]), superimposed_img)
