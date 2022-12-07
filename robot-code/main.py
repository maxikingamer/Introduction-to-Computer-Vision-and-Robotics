import cv2
import sys
import time
import numpy as np
import jsonpickle
from message import Message
from camera import Camera
from publisher import Publisher
from keypress_listener import KeypressListener
import ev3_dc as ev3
from thread_task import Periodic, Task, Repeated, STATE_STARTED, STATE_FINISHED, STATE_STOPPED, STATE_TO_STOP


class Main():
    def __init__(self) -> None:
        # keypress listener will break the terminal if we don't close it on exception
        sys.excepthook = self.except_hook

        self.keypress_listener = KeypressListener()
        self.publisher = Publisher()
        self.camera = Camera()
        self.camera_height = 0.285

        print("camera_matrix", self.camera.camera_matrix)
        print("camera_", self.camera.dist_coeffs)

        self.run()

    ##### -------OUR CODE--------- ######
    def transformImage(self, img):
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)        
        parameters = cv2.aruco.DetectorParameters_create()  
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
        positions = []
        distances = []

        if ids is not None:
            for j in range(0, len(ids)):
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[j], 0.048, self.camera.camera_matrix, self.camera.dist_coeffs)
                # (rvec - tvec).any()

                hypothenuse = np.linalg.norm(tvec)
                distance = np.sqrt(hypothenuse**2 - self.camera_height**2)
                distances.append(distance)
                pos_x = tvec[0][0][0]
                pos_y = np.sqrt(distance**2 - pos_x**2)

                positions.append([pos_x, pos_y])
                cv2.aruco.drawDetectedMarkers(img, corners)
                cv2.drawFrameAxes(img, self.camera.camera_matrix, self.camera.dist_coeffs, rvec, tvec, 0.048)
        else:
            ids = []


        return img, ids, positions, distances

    ##### -------OUR CODE--------- ######

    def run(self):
        # counter
        count = 0
        start = True

        try:
            with ev3.TwoWheelVehicle(
                0.0210,  # radius_wheel
                0.1350,  # tread
                protocol=ev3.USB
            ) as vehicle:
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
                        img, ids, positions, distance = self.transformImage(raw_img)

                        if len(distance) > 0:
                            if min(distance)<0.4:
                                Task(vehicle.drive_turn(180,0,speed=30)).start(thread=False)
                            else:
                                Task(vehicle.drive_straight(0.01)).start(thread=False)
                        else:
                            Task(vehicle.drive_straight(0.01)).start(thread=False)
                        

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
        except Exception as e:
            print(e)

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