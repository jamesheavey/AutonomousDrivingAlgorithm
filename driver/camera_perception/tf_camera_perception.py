from .camera_perception_abc import CameraPerceptionABC
from driver.tf_model.utils import cv_utils
from driver.tf_model.utils import operations as ops
from driver.tf_model.utils import tf_utils
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# TF Parameters
FROZEN_GRAPH_PATH = '/home/ros/fs_ws/src/f21ai-sim/driver/driver/tf_model/models/ssd_mobilenet_v1/frozen_inference_graph.pb'        # noqa: E501

YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)

Y_CROP = [200, 400]
CROP_SIZE = Y_CROP[1] - Y_CROP[0]
CROP_STEP = CROP_SIZE - 20

SCORE_THRESHOLD = 0.3
NON_MAX_SUPPRESSION_THRESHOLD = 0.2


class TFCameraPerception(CameraPerceptionABC):
    def __init__(self):
        self.detection_graph = tf_utils.load_model(FROZEN_GRAPH_PATH)

    def get_cones(self, frame, parameters={}):
        """
        Find cones using a tensorflow neural net and return them in the world frame
        Ref: https://github.com/fediazgon/cone-detector-tf
        """
        blue_points = []
        yellow_points = []

        section = frame[Y_CROP[0]:Y_CROP[1], 0:frame.shape[1]]

        with tf.Session(graph=self.detection_graph) as sess:

            crops, crops_coordinates = ops.extract_crops(
                section, CROP_SIZE, CROP_SIZE, CROP_STEP, CROP_STEP)

            detection_dict = tf_utils.run_inference_for_batch(crops, sess)

            # The detection boxes obtained are relative to each crop. Get
            # boxes relative to the original image
            # IMPORTANT! The boxes coordinates are in the following order:
            # (ymin, xmin, ymax, xmax)
            boxes = []
            for box_absolute, boxes_relative in zip(
                    crops_coordinates, detection_dict['detection_boxes']):
                boxes.extend(ops.get_absolute_boxes(
                    box_absolute,
                    boxes_relative[np.any(boxes_relative, axis=1)]))
            if boxes:
                boxes = np.vstack(boxes)

            # Remove overlapping boxes
            boxes = ops.non_max_suppression_fast(
                boxes, NON_MAX_SUPPRESSION_THRESHOLD)

            # Get scores to display them on top of each detection
            boxes_scores = detection_dict['detection_scores']
            boxes_scores = boxes_scores[np.nonzero(boxes_scores)]

            for box, score in zip(boxes, boxes_scores):
                if score <= SCORE_THRESHOLD:
                    continue

                ymin, xmin, ymax, xmax = box

                color_detected_rgb = cv_utils.predominant_rgb_color(
                    section, ymin, xmin, ymax, xmax)
                text = '{:.2f}'.format(score)

                ymin = ymin + Y_CROP[0]
                ymax = ymax + Y_CROP[0]

                cv_utils.add_rectangle_with_text(
                    frame, ymin, xmin, ymax, xmax,
                    color_detected_rgb, text)

                # area for weighting, could also use score
                area = (ymax-ymin)*(xmax-xmin)
                x = xmin+(xmax-xmin)/2
                y = ymax

                # if area > 100:
                if color_detected_rgb == YELLOW:
                    yellow_points.append((x, y, min(area/200, 1)))

                if color_detected_rgb == BLUE:
                    blue_points.append((x, y, min(area/200, 1)))

        return frame, blue_points, yellow_points
