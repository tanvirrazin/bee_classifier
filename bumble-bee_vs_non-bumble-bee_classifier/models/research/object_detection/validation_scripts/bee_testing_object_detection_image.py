######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description:
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import datetime
import cv2
import numpy as np
import tensorflow as tf
import sys
import xml.etree.ElementTree as ET

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util


class BeeImageDetector(object):

    def __init__(self):
        self.correct_prediction = 0
        self.wrong_prediction = 0

    def take_input(self):
        self.IMAGE_NAME = sys.argv[1]

    def setup_model(self):
        # Name of the directory containing the object detection module we're using
        MODEL_NAME = ''

        # Grab path to current working directory
        self.CWD_PATH = os.getcwd()

        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        PATH_TO_CKPT = os.path.join(self.CWD_PATH,MODEL_NAME,'frozen_inference_graph_07_18_2021_color_gray.pb')

        # Path to label map file
        PATH_TO_LABELS = os.path.join(self.CWD_PATH, 'bumble_or_not_label_map.pbtxt')

        # Number of classes the object detector can identify
        NUM_CLASSES = 2

        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(
            self.label_map,
            max_num_classes=NUM_CLASSES,
            use_display_name=True
        )
        self.category_index = label_map_util.create_category_index(self.categories)

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.compat.v1.Session(graph=detection_graph)

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def detect(self, image_path=None):
        self.detect_from_dir('./unseen_new_gray')

    def detect_from_dir(self, dir_name=None):
        for filename in os.listdir(dir_name):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                print(filename, end=' -- ')
                self.detect_from_file(os.path.join(dir_name, filename))

    def detect_from_file(self, image_name=None):
        # Path to image
        PATH_TO_IMAGE = os.path.join(self.CWD_PATH,image_name)

        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image = cv2.imread(PATH_TO_IMAGE)
        image_expanded = np.expand_dims(image, axis=0)
        # print(image_expanded)
        # print(image_expanded.shape)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = self.sess.run(
            [
                self.detection_boxes,
                self.detection_scores,
                self.detection_classes,
                self.num_detections
            ],
            feed_dict={self.image_tensor: image_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        # print(len(boxes))
        # print(scores)
        # print(classes)
        predicted_image, classes = vis_util.visualize_boxes_and_labels_on_image_array__changed(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=4,
            min_score_thresh=0.90)

        # print("Image name: {}".format(image_name))
        class_text = classes.split(': ')[0]

        print("Classes: {}".format(class_text), end="")
        gt_classes = self.parse_xml(PATH_TO_IMAGE)
        pt_classes = [cl.split(': ')[0] for cl in classes.split('; ')]

        # print(pt_classes)
        # print(gt_classes)

        num_of_intersection = len(set(pt_classes).intersection(gt_classes))

        if num_of_intersection > 0:
            print(" -- Correct")
            self.correct_prediction += 1
        else:
            print(" -- Wrong")
            self.wrong_prediction += 1

        # # All the results have been drawn on image. Now display the image.
        # cv2.imshow('Object Detection', image)
	#
        # cv2.imwrite('image_written.jpg', image)
	#
        # # Press any key to close the image
        # cv2.waitKey(0)
	#
        # # Clean up
        # cv2.destroyAllWindows()

        return classes

    def parse_xml(self, image_file_name):
        gt_classes = []
        xml_file_name = image_file_name.rsplit('.', 1)[0] + '.xml'
        tree = ET.parse(xml_file_name)
        root = tree.getroot()

        for child in root:
            if child.tag == 'object':
                for ch in child:
                    if ch.tag == 'name':
                        if ch.text == 'bumblebee':
                            gt_classes.append('bumblebee')
                        else:
                            gt_classes.append('not_bumblebee')

        return gt_classes


if __name__ == "__main__":
    bd = BeeImageDetector()
    bd.setup_model()
    # bd.take_input()
    bd.detect()
    print("Correct Prediction: {}, Wrong Prediction: {}, Accuracy: {:.6f} %".format(
        bd.correct_prediction,
        bd.wrong_prediction,
        100 * bd.correct_prediction / (bd.correct_prediction + bd.wrong_prediction)
    ))
