from .camera_perception_abc import CameraPerceptionABC
import numpy as np
import cv2

# HSV Cone colour ranges
BLUE_RANGE = [np.array([50, 50, 50]), np.array([130, 200, 200])]
YELLOW_RANGE = [np.array([25, 50, 50]), np.array([40, 255, 255])]

# Morphology Kernel
SIZE = 51
KERNEL = np.zeros((SIZE, SIZE), np.uint8)
KERNEL[0:SIZE, SIZE//2] = 1

# Cone area thresholds
MIN_AREA = 30
MAX_AREA = 600
# Smoothing Factor
S_F = 5

# Proportion of the top of the frame to exclude
EXCLUDED_TOP_Y_PROPORTION = 1/4

# Value to scale down cone areas for probability bounds (0-1)
CONE_PROBABILITY_SCALE = 1000


class CVCameraPerception(CameraPerceptionABC):
    def __init__(self):
        super().__init__()

    def get_cones(self, frame, parameters={}):
        """
        Find cones within certain colour range and return them in the world frame
        """
        s_f = parameters.get("smoothing_factor", S_F)
        cone_prob_scale = parameters.get("cone_probability_scale", CONE_PROBABILITY_SCALE)

        # Smooth the image to account for fuzz
        frame = cv2.filter2D(frame, -1, np.ones((s_f, s_f), np.float32)/(s_f**2))

        # Convert bgr frame to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Mask frame with limits
        blue = cv2.inRange(hsv, *BLUE_RANGE)
        yellow = cv2.inRange(hsv, *YELLOW_RANGE)

        # Close vertical masks to define full cones
        blue_cone = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, KERNEL)
        yellow_cone = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, KERNEL)

        # Find contours on the mask
        b_contours, _ = cv2.findContours(
            blue_cone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        y_contours, _ = cv2.findContours(
            yellow_cone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        blue_contours = [(cnt, 'b') for cnt in b_contours]
        yellow_contours = [(cnt, 'y') for cnt in y_contours]

        # Draw the contours on the frame
        frame = cv2.drawContours(frame, b_contours, -1, (0, 255, 0), 3)
        frame = cv2.drawContours(frame, y_contours, -1, (255, 0, 255), 3)

        b_points, y_points = [], []

        for cnt, colour in blue_contours + yellow_contours:
            # Only include contours have an area within the specified bounds
            area = cv2.contourArea(cnt)
            if MAX_AREA < area or area < MIN_AREA:
                continue

            # Calculate contour centre of mass
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])

            # Approximate a polygon to the contour
            approx = cv2.approxPolyDP(
                cnt, 0.009 * cv2.arcLength(cnt, True), True)

            # Locate the highest pixel y-coordinate
            cy = max(c[0][1] for c in approx)

            if cy < frame.shape[0] * EXCLUDED_TOP_Y_PROPORTION:
                continue

            # Draw dot on contour center
            frame = cv2.circle(frame, (cx, cy), 3, [0, 0, 255], -1)

            if colour == 'b':
                b_points.append((cx, cy, min(area/cone_prob_scale, 1)))
            elif colour == 'y':
                y_points.append((cx, cy, min(area/cone_prob_scale, 1)))

        return frame, b_points, y_points
