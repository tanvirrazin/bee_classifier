import os 
import glob
import cv2
import numpy as np
import sys
import tensorflow as tf
from object_detection.utils import label_map_util

SCORE_THRESHOLD = 0.8

files = []
files =  glob.glob('../../../images/unseen_new/*.jpg')
files.extend(glob.glob('../../../images/unseen_new/*.jpeg'))
files.extend(glob.glob('../../../images/unseen_new/*.png'))


class BeeImageDetector(object):

    def take_input(self, image_path=None):
        self.IMAGE_NAME = image_path

    def setup_model(self):
        # Name of the directory containing the object detection module we're using
        MODEL_NAME = '../../../inference_graph/bumble_or_not_faster_rcnn_resnet101_coco_06_25_2020/'

        # Grab path to current working directory
        self.CWD_PATH = os.getcwd()

        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        PATH_TO_CKPT = os.path.join(self.CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

        # Path to label map file
        PATH_TO_LABELS = os.path.join(self.CWD_PATH,'../../../training','bumble_or_not_label_map.pbtxt')

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

    def detect(self, image_name=None):
        # Path to image
        PATH_TO_IMAGE = os.path.join(self.CWD_PATH,image_name)
        file_content = ""

        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image = cv2.imread(PATH_TO_IMAGE)
        (im_height, im_width, im_depth) = image.shape
        
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
        # print(len(boxes[0]))
        # print(len(scores[0]))
        # print(len(classes[0]))
        
        for i in range(len(scores[0])):
            if scores[0][i] > SCORE_THRESHOLD:
                """
                print(
                    self.category_index[classes[0][i]]['name'], 
                    scores[0][i]*100, 
                    boxes[0][i][0]*im_height,    # ymin
                    boxes[0][i][1]*im_width,     # xmin
                    boxes[0][i][2]*im_height,    # ymax
                    boxes[0][i][3]*im_width,     # xmax
                )
                """
                file_content += "{} {} {} {} {} {}\n".format(
                    self.category_index[classes[0][i]]['name'], 
                    int(scores[0][i]*100),
                    int(boxes[0][i][1]*im_width),     # xmin 
                    int(boxes[0][i][0]*im_height),    # ymin
                    int(boxes[0][i][3]*im_width),     # xmax
                    int(boxes[0][i][2]*im_height),    # ymax
                )
        
        with open('../../input/detection-results/{}.txt'.format(image_name.split('/')[-1].split('.')[0]), 'w') as f:
            f.write(file_content)
            f.close()

        print(image_name.split('/')[-1], file_content)
        print("\n\n")

        """
        predicted_image, classes = vis_util.visualize_boxes_and_labels_on_image_array__changed(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=4,
            min_score_thresh=0.90)

        print("Image name: {}".format(image_name))
        print("Classes: {}".format(classes))

        # All the results have been drawn on image. Now display the image.
        cv2.imshow('Object Detection', image)

        cv2.imwrite('image_written.jpg', image)

        # Press any key to close the image
        cv2.waitKey(0)

        # Clean up
        cv2.destroyAllWindows()
        """

        return classes

bd = BeeImageDetector()
bd.setup_model()
# bd.take_input()
# bd.detect(bd.IMAGE_NAME)

for filename in files:
    
    bd.detect(filename)

    # break

