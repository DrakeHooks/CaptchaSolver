import cv2
import numpy as np
import os
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Avoid GPU errors by running on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#Declare the title of the window, and the path to the frozen detection graph. Also, specify the path to the label map, and the number of classes the object detector can identify.
title = "CAPTCHA"
PATH_TO_FROZEN_GRAPH = 'CAPTCHA_frozen_inference_graph.pb'
PATH_TO_LABELS = 'CAPTCHA_labelmap.pbtxt'
NUM_CLASSES = 37

# Load the label map and create a category index.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Set the detection_graph to the default graph of tensorflow.
detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    # use with to open the frozen graph and create a graph definition from it.
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Captcha_detection() function takes in parameters for the image, and the average distance between two consecutive letters.
# with detection_graph.as_default(): and with tf.Session(graph=detection_graph) as sess: are used to load the frozen graph into memory.
def Captcha_detection(image, average_distance_error=3):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_np = cv2.imread(image)
            # Resize image to improve detection speed.
            image_np = cv2.resize(image_np, (0, 0), fx=5, fy=5)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            #Use numpy to convert the image into a form tensorflow can understand. Also, expand the dimensions of the image so that it represents a batch of size 1.
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Define the input and output tensors (i.e. data) for the object detection classifier.
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # feed_dict is used to feed the image into the object detection classifier.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            
            #The function below takes in the bounding box coordinates, and returns the captcha text and the confidence of the prediction.
            # use numpy squeeze function to get rid of single-dimensional entries from the shape of an array.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                # use .astype() to convert the data type of an array from float32 to int32.
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=2)

            # This creates a window and displays the image with the detections.
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.imshow(title, cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            cv2.resizeWindow(title, 1400, 400)
            cv2.imwrite("Predicted_captcha.jpg", cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))



            #use an an array and a for loop to extract the characters one by one. Then, sort the array based on the x coordinate of the bounding box. 
            # This ensures that we get the characters from left to right.
            captcha_array = []

            for i, b in enumerate(boxes[0]):
                for Symbol in range(37):
                    # Use the variable mid_x to hold the mid point of the box in the x axis. Then, use the variable scores to hold the confidence score of the prediction.
                    if classes[0][i] == Symbol and scores[0][i] >= 0.65:
                        mid_x = (b[1] + b[3]) / 2
                        captcha_array.append([category_index[Symbol].get('name'), mid_x, scores[0][i]])

            # Sort the array in ascending order based on the x coordinate of the bounding box. Lambda is used to specify the sorting parameter.
            captcha_array.sort(key=lambda x: x[1])

            # Use the variable average to hold the average distance between two consecutive characters. 
            average = sum(captcha_array[i+1][1] - captcha_array[i][1] 
                          for i in range(len(captcha_array) - 1)) / len(captcha_array)

            #This is to filter out some false positives. If the distance between two consecutive characters is less than the average width, then we only keep the character with the higher confidence.
            captcha_array_filtered = [captcha_array[0]]
            for i in range(1, len(captcha_array)):
                if captcha_array[i][1] - captcha_array[i-1][1] >= average:
                    captcha_array_filtered.append(captcha_array[i])

            # use .join() to concatenate all the predicted captcha letters into the final captcha text.
            captcha_string = "".join(symbol[0] for symbol in captcha_array_filtered)
            return captcha_string
