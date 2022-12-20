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
        self.slam = EKFSLAM(0,0,0,0)
        self.old_pos = [0,0,0]
        self.seen_ids = {}

        print("camera_matrix", self.camera.camera_matrix)
        print("camera_", self.camera.dist_coeffs)

        self.run()

    ##### -------OUR CODE--------- ######
    def transformImage(self, img, robot_position):
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)        
        parameters = cv2.aruco.DetectorParameters_create()
        parameters.errorCorrectionRate = 0.1  
        positions = []
        distances = []
        angles = []
        rotation_matrix = np.array([[np.cos(robot_position[2]),-np.sin(robot_position[2])],[np.sin(robot_position[2]),np.cos(robot_position[2])]])

        if img is not None:
            corners, ids, rejected_img_points = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
            accepted_ids = []
            if ids is not None:
                for j in range(0, len(ids)):
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[j], 0.048, self.camera.camera_matrix, self.camera.dist_coeffs)
                    # (rvec - tvec).any()

                    hypothenuse = np.linalg.norm(tvec)
                    
                    distance = np.sqrt(np.abs(hypothenuse**2 - self.camera_height**2))
                    distances.append(distance)
                    pos_x = tvec[0][0][0]
                    pos_y = np.sqrt(np.abs(distance**2 - pos_x**2))

                    direction_global_coords = rotation_matrix @ np.array([pos_x,pos_y]) 
                    pos_x = direction_global_coords[0]
                    pos_y = direction_global_coords[1]

                    angle = (np.arctan2(-pos_x,pos_y))
                    angles.append(angle + robot_position[2])

                    positions.append([pos_x + robot_position[1], pos_y + robot_position[0]])
                    cv2.aruco.drawDetectedMarkers(img, corners)
                    cv2.drawFrameAxes(img, self.camera.camera_matrix, self.camera.dist_coeffs, rvec, tvec, 0.048)

                    accepted_ids.append(ids[j])
                
                not_so_young_ids1 = ids
                not_so_young_ids2 = not_so_young_ids1

            else:
                ids = []
        else:
            ids = []


        return img, ids, positions, distances, angles

    def checkDistance(self, vehicle, angle=0, radius=0):
        try:
            new_pos = vehicle.position
            distance_travelled = np.sqrt((new_pos[0]-self.old_pos[0])**2 + (new_pos[1]-self.old_pos[1])**2)
            self.old_pos = new_pos

            # get image from the camera
            _, raw_img = self.camera.read()
            
            # get the time
            time0 = time.time()

            x, y, angle = vehicle.position

            # imaginary processing
            img, ids, positions, distances, angles = self.transformImage(raw_img, [x,y,angle])

            #########SLAM#######
            uncertainty = [0,0]
            if len(ids) > 0:
                if angle == 0 and radius == 0:
                    self.slam.predict(vehicle, 0.1350, "straight", distance_travelled)
                else:
                    self.slam.predict(vehicle, 0.1350, "turn", [angle,radius])
                for i, id in enumerate(ids):
                    if self.slam.id_never_seen_before(id[0]):
                        self.slam.add_landmark(id[0],positions[i],uncertainty)
                    self.slam.correction(id[0],positions[i])
                    # print("landmark estimated positions:", self.slam.get_landmark_positions())
            ####################

            slam_position, slam_sigma = self.slam.get_robot_pose()
            landmark_estimated_positions, landmark_estimated_stdevs = self.slam.get_landmark_positions()

            landmark_estimated_ids = list(self.slam.map.keys())
            landmark_estimated_positions = landmark_estimated_positions.reshape(int(len(landmark_estimated_ids)),2)
            landmark_estimated_stdevs = landmark_estimated_stdevs.diagonal().reshape(int(len(landmark_estimated_ids)),2)

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

                landmark_estimated_ids = landmark_estimated_ids,
                landmark_estimated_positions = landmark_estimated_positions,
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
            0.0280,  # radius_wheel
            0.1392,  # tread
            protocol=ev3.USB
        ) 

        turned = False
        pool = ThreadPool(processes=2)
        while True:
            try:
                print("Current movement: ", vehicle._current_movement)
                if turned:
                    watch_results = pool.apply_async(self.checkDistance, (vehicle,180,0))
                else:
                    watch_results = pool.apply_async(self.checkDistance, (vehicle,))
            
                if vehicle._current_movement is None:
                    turned = False
                    print("straight!")
                    pool.apply_async(vehicle.move(10,0))

                distance = watch_results.get()
                if distance < 0.35:
                    print("too close man")
                    vehicle.stop()
                    if vehicle._current_movement is None:
                        print("turning!")
                        vehicle.drive_turn(180,0).start(thread=False)
                        print("Current movement: ", vehicle._current_movement)
                        vehicle.stop()
                        print("Current movement: ", vehicle._current_movement)
                        turned = True
                            
            except Exception as e:
                print(e)

        vehicle.stop()
        vehicle.close()
    
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