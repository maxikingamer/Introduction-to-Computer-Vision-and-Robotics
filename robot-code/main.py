import cv2
import sys
import time
import numpy as np
import jsonpickle
from message import Message
from camera import Camera
from publisher import Publisher
from keypress_listener import KeypressListener

class Main():
    def __init__(self) -> None:
        # keypress listener will break the terminal if we don't close it on exception
        sys.excepthook = self.except_hook

        self.keypress_listener = KeypressListener()
        self.publisher = Publisher()
        self.camera = Camera()

        print("camera_matrix", self.camera.camera_matrix)
        print("camera_", self.camera.dist_coeffs)

        self.run()

    ##### -------OUR CODE--------- ######
    def transformImage(self, img):
        # img = np.flipud(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)        
        parameters = cv2.aruco.DetectorParameters_create()  
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
        tvecs = []

        if np.all(ids is not None):
            for j in range(0, len(ids)):
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[j], 1, self.camera.camera_matrix, self.camera.dist_coeffs)
                (rvec - tvec).any()
                tvecs.append([tvec[0][0][0], tvec[0][0][1]])
                cv2.aruco.drawDetectedMarkers(img, corners)
                cv2.drawFrameAxes(img, self.camera.camera_matrix, self.camera.dist_coeffs, rvec, tvec, 1)

        return img, ids, tvecs

    ##### -------OUR CODE--------- ######

    def run(self):
        # counter
        count = 0
        start = True
        while True:
            try:
                print("count:", count)
                # maybe we want to get keypresses in the terminal, for debugging
                self.parse_keypress()

                # get image from the camera
                _, raw_img = self.camera.read()
                
                # get the time
                time0 = time.time()

                # imaginary processing
                img, ids, positions = self.transformImage(raw_img)

                # create message
                msg = Message(
                    id = count,
                    timestamp = time0,
                    start = True,
                    
                    landmark_ids = ids,
                    landmark_rs = [],
                    landmark_alphas = [],
                    landmark_positions = positions,

                    landmark_estimated_ids = [],
                    landmark_estimated_positions = [],
                    landmark_estimated_stdevs = [],

                    robot_position = np.array([0.0, 0.0]),
                    robot_theta = 0.0,
                    robot_stdev = [0.5, 0.5, 0.5],
                )

                # pickle message
                msg_str = jsonpickle.encode(msg)

                # publish message and image
                self.publisher.publish_img(msg_str, img)

                count += 1
                start = False

            except KeyboardInterrupt:
                # tidy up
                self.close()
                break
        
        # tidy up
        self.close()

    def except_hook(self, type, value, tb):
        self.close()

    def close(self):
        self.keypress_listener.close()
        self.camera.close()

    def parse_keypress(self):
        char = self.keypress_listener.get_keypress()
        if char is not None:
            print("char:", char)

if __name__ == '__main__':
    main = Main()