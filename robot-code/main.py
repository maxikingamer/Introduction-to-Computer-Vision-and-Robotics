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
from threading import Event, Thread
from slam import EKFSLAM
import math
from multiprocessing.pool import ThreadPool


class Main():
    def __init__(self) -> None:
        # keypress listener will break the terminal if we don't close it on exception
        sys.excepthook = self.except_hook

        self.keypress_listener = KeypressListener()
        self.publisher = Publisher()
        self.camera = Camera()
        self.camera_height = 0.285
        self.count = 0
        self.slam = EKFSLAM(0.01,0.01,0.01,0.01)

        print("camera_matrix", self.camera.camera_matrix)
        print("camera_", self.camera.dist_coeffs)

        self.run()

    ##### -------OUR CODE--------- ######
    def transformImage(self, img):
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)        
        parameters = cv2.aruco.DetectorParameters_create()  
        positions = []
        distances = []
        angles = []
        if img is not None:
            corners, ids, rejected_img_points = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
            
            if ids is not None:
                for j in range(0, len(ids)):
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[j], 0.048, self.camera.camera_matrix, self.camera.dist_coeffs)
                    # (rvec - tvec).any()

                    hypothenuse = np.linalg.norm(tvec)
                    while True:
                        try:    
                            distance = np.sqrt(hypothenuse**2 - self.camera_height**2)
                            break
                        except:
                            continue
                    distances.append(distance)
                    pos_x = tvec[0][0][0]
                    pos_y = np.sqrt(distance**2 - pos_x**2)
                    angle = (np.arctan2(-pos_x,pos_y))
                    angles.append(angle)


                    positions.append([pos_x, pos_y])
                    cv2.aruco.drawDetectedMarkers(img, corners)
                    cv2.drawFrameAxes(img, self.camera.camera_matrix, self.camera.dist_coeffs, rvec, tvec, 0.048)
            else:
                ids = []
        else:
            ids = []


        return img, ids, positions, distances, angles

    def checkDistance(self, vehicle):
        try:
            # get image from the camera
            _, raw_img = self.camera.read()
            
            # get the time
            time0 = time.time()

            # imaginary processing
            img, ids, positions, distances, angles = self.transformImage(raw_img)

            #########SLAM#######
            uncertainty = [0.1,0.1]
            if len(ids) > 0:
                self.slam.predict(vehicle, 0.1350, "straight", 0.001)
                for i, id in enumerate(ids):
                    if self.slam.id_never_seen_before(id[0]):
                        self.slam.add_landmark(id[0],positions[i],uncertainty)
                    self.slam.correction(id[0],positions[i])
                    # print("landmark estimated positions:", self.slam.get_landmark_positions())
            ####################

            # curr_state = drive.state

            slam_position, slam_sigma = self.slam.get_robot_pose()
            estimated_landmark_positions, landmark_estimated_stdevs = self.slam.get_landmark_positions()

            # create message
            msg = Message(
                id = self.count,
                timestamp = time0,
                start = True,
                
                landmark_ids = ids,
                landmark_rs = distances,
                landmark_alphas = angles,
                landmark_positions = positions,

                # landmark_estimated_ids = [],
                # landmark_estimated_positions = [],
                # landmark_estimated_stdevs = [],

                # robot_position = np.array([0.0, 0.0]),
                # robot_theta = 0.0,
                # robot_stdev = [0.5, 0.5, 0.5],

                landmark_estimated_ids = list(self.slam.map.keys()),
                landmark_estimated_positions = estimated_landmark_positions,
                landmark_estimated_stdevs = landmark_estimated_stdevs,

                robot_position = np.array([slam_position[1], slam_position[0]]),
                robot_theta = math.radians(slam_position[2]+90),
                robot_stdev = slam_sigma.diagonal(),
            )
            self.count += 1

            # pickle message
            msg_str = jsonpickle.encode(msg)

            # publish message and image
            self.publisher.publish_img(msg_str, img)

            # if curr_state == STATE_STOPPED:
            #     return True

            if len(distances) > 0:
                return min(distances)
        
            return np.inf

        except KeyboardInterrupt as e:
            # tidy up
            print(e)
            self.close()

    ##### -------OUR CODE--------- ######

    def run(self):
        vehicle = ev3.TwoWheelVehicle (
            0.0210,  # radius_wheel
            0.1350,  # tread
            protocol=ev3.USB
        ) 

        stopped = False
        try:
            while not stopped:
                pool = ThreadPool(processes=2)
                watch_results = pool.apply_async(self.checkDistance, (vehicle,))
                
                if vehicle._current_movement is None:
                    pool.apply_async(vehicle.drive_straight(0.1).start)

                distance = watch_results.get()
                if distance < 0.4:
                    pool.close()
                    stopped = True
        except Exception as e:
            print(e)

        vehicle.stop()
        
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