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
from slam import EKFSLAM
import math
from multiprocessing.pool import ThreadPool
import curses


class Main():
    def __init__(self, stdscr) -> None:
        # keypress listener will break the terminal if we don't close it on exception
        sys.excepthook = self.except_hook

        self.keypress_listener = KeypressListener()
        self.publisher = Publisher()
        self.camera = Camera()
        self.camera_height = 0.285
        self.count = 0
        self.slam = EKFSLAM(0.05,0.05,0.07,1*np.pi/180)
        self.old_pos = [0,0,0]
        self.input = stdscr
        self.speed = 0
        self.turn = 0
        self.tread = 0.1392
        self.radius = 0.0280 
        self.seen_ids = {}

        # print("camera_matrix", self.camera.camera_matrix)
        # print("camera_", self.camera.dist_coeffs)
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        #params.minThreshold = 1
        #params.maxThreshold = 256
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 10**2 * np.pi
        params.maxArea = 100**2 * np.pi
        # Filter by Color (black=0)
        params.filterByColor = False
        params.blobColor = 0
        #Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.8
        params.maxCircularity = 10000
        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.9
        params.maxConvexity = 1000
        # Filter by InertiaRatio
        params.filterByInertia = True
        params.minInertiaRatio = 0.1
        params.maxInertiaRatio = 1
        # Distance Between Blobs
        params.minDistBetweenBlobs = 0.1
        # Do blob detecting 
        self.detector = cv2.SimpleBlobDetector_create(params)

        self.run()

    ##### -------OUR CODE--------- ######
    def transformImage(self, img, robot_position):
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)        
        parameters = cv2.aruco.DetectorParameters_create()
        parameters.errorCorrectionRate = 0.001  
        positions = []
        distances = []
        angles = []
        accepted_ids = []
        rotation_matrix = np.array([[np.cos(robot_position[2]),
                                    -np.sin(robot_position[2])],
                                    [np.sin(robot_position[2]),
                                    np.cos(robot_position[2])]])

        if img is not None:

            # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # smoothed = cv2.GaussianBlur(gray, (5,5), sigmaX=9, sigmaY=9, borderType = cv2.BORDER_DEFAULT)
            # thresh = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, 10)
            # Set up the SimpleBlobdetector with default parameters.
            
            # Get keypoints
            # keypoints = self.detector.detect(thresh)
            # # detect green
            # lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            # # store the a-channel
            # a_channel = lab[:,:,1]
            # # Automate threshold using Otsu method
            # th = cv2.threshold(a_channel,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
            
            # detect aruco markers
            corners, ids, rejected_img_points = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
            if ids is not None:
                for j in range(0, len(ids)):
                    if not(str(ids[j][0]) in list(self.seen_ids.keys())):
                        self.seen_ids[str(ids[j][0])] = 1
                    else:
                        self.seen_ids[str(ids[j][0])] = self.seen_ids[str(ids[j][0])] + 1

                    if self.seen_ids[str(ids[j][0])] < 5:
                        continue

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

                    angle = (np.arctan2(pos_y,pos_x))
                    angles.append(angle)

                    positions.append([pos_x + robot_position[1], pos_y + robot_position[0]])
                    cv2.aruco.drawDetectedMarkers(img, corners)
                    cv2.drawFrameAxes(img, self.camera.camera_matrix, self.camera.dist_coeffs, rvec, tvec, 0.048)

                    accepted_ids.append(ids[j])

            # for keypoint in keypoints:
            #     x = int(keypoint.pt[0])
            #     y = int(keypoint.pt[1])
            #     s = keypoint.size
            #     r = int(math.floor(s/2))
            #     if th[y-r:y+r,x-r:x+r].sum() > 100:
            #         cv2.circle(img,(x, y), r, color=(0,255,0))
            #     else:
            #         cv2.circle(img,(x, y), r, color=(0,0,255))

            else:
                ids = []
        else:
            ids = []


        return img, accepted_ids, positions, distances, angles

    def react(self, c):
        '''
        reacts on keyboard arrow key events by modifying speed and turn
        '''
        if c == curses.KEY_LEFT:
            self.turn += 5
            self.turn = min(self.turn, 200)
        elif c == curses.KEY_RIGHT:
            self.turn -= 5
            self.turn = max(self.turn, -200)
        elif c == curses.KEY_UP:
            self.speed += 5
            self.speed = min(self.speed, 100)
        elif c == curses.KEY_DOWN:
            self.speed -= 5
            self.speed = max(self.speed, -100)

    def checkDistance(self, vehicle):
        try:
            pos, sigma = self.slam.get_robot_pose()
            x, y, angle = pos
            new_pos =[x,y,angle]
            x, y, angle = vehicle.position
            ev3_pos = [x,y,math.radians(angle)]
            #if np.linalg.norm(np.array(new_pos[0:2]) - np.array(self.old_pos[0:2])) < 0.01:
            #    new_pos = self.old_pos
            self.input.addstr(5, 0, f"SLAM Position: [{new_pos[0]},{new_pos[1]},{new_pos[2]}]")
            # new_pos = [x,y,angle]
            self.input.addstr(6, 0, f"EV3 Position: {ev3_pos}")

            direction = np.arctan2(new_pos[1]-self.old_pos[1],new_pos[0]-self.old_pos[0])

            # get image from the camera
            _, raw_img = self.camera.read()
            
            # get the time
            time0 = time.time()

            # imaginary processing
            img, ids, positions, distances, angles = self.transformImage(raw_img, new_pos)

            #########SLAM#######
            uncertainty = [0.1,0.1]

            """ 
            if np.abs(self.old_pos[2] - new_pos[2]) <= math.pi/180:
                self.input.addstr(3, 0, f"driving straight")
                # self.input.addstr(0, 0, f"[{x},{y},{angle}]")
                if backwards:
                    self.slam.predict(vehicle, self.tread, "straight", -distance_travelled)
                else:
                    self.slam.predict(vehicle, self.tread, "straight", distance_travelled)
            else:
                # center_x = (new_pos[1] - np.tan((math.pi/2) - new_pos[2]) - self.old_pos[1] + np.tan((math.pi/2) - self.old_pos[2])) / (np.tan((math.pi/2) - new_pos[2]) - np.tan((math.pi/2) - self.old_pos[2]))
                # center_y = (np.tan((math.pi/2) - new_pos[2]) * center_x + new_pos[1] - np.tan((math.pi/2) - new_pos[2]))
                # R = np.linalg.norm(np.array([center_x, center_y]) - np.array([self.old_pos[0], self.old_pos[1]]))
                # alpha = 2 * (np.arctan2(new_pos[0] - self.old_pos[0], new_pos[1] - self.old_pos[1]) - new_pos[2])

                b_new = new_pos[1] - np.tan(math.pi/2 + new_pos[2]) * new_pos[0]
                b_old = self.old_pos[1] - np.tan(math.pi/2 + self.old_pos[2]) * self.old_pos[0]
                m_new = np.tan(math.pi/2 + new_pos[2])
                m_old = np.tan(math.pi/2 + self.old_pos[2])

                center_x = (b_new - b_old) / (m_new - m_old)
                center_y = m_new * center_x + b_new
                R = np.linalg.norm(np.array([center_x, center_y]) - np.array([self.old_pos[0], self.old_pos[1]]))
                alpha = (2 * np.sin(distance_travelled/(2*R)))  % (2 * np.pi)
                self.input.addstr(20, 0, f"old angle form: {alpha}")
                #alpha =  (2* (math.radians(math.degrees(direction) - math.degrees(self.old_pos[2])))) % (2* np.pi)
                #self.input.addstr(21, 0, f"new angle form: {alpha}")
                self.input.addstr(3, 0, f"turniiiiiiing!!!") 
            """
            self.slam.predict(vehicle, self.tread)

            self.old_pos = new_pos
            
            
            if len(ids) > 0:
                for i, id in enumerate(ids):
                    if self.slam.id_never_seen_before(id[0]):
                        self.slam.add_landmark(id[0],positions[i],uncertainty)
                    # measured_r, measured_alpha, est_r, est_alpha, quark = self.slam.correction(id[0],positions[i])
                    # print("landmark estimated positions:", self.slam.get_landmark_positions())

                    # self.input.addstr(15, 0, f"r-diff={measured_r-est_r}")
                    # self.input.addstr(16, 0, f"alpha-diff={measured_alpha-est_alpha}")
                    # self.input.addstr(17, 0, f"quark={quark}")

        

            print_robot_pos, _ = self.slam.get_robot_pose()
            ####################


            slam_position, slam_sigma = self.slam.get_robot_pose()
            # slam_position = vehicle.position
            landmark_estimated_positions, landmark_estimated_stdevs = self.slam.get_landmark_positions()

            landmark_estimated_ids = list(self.slam.map.keys())
            landmark_estimated_positions = landmark_estimated_positions.reshape(int(len(landmark_estimated_ids)),2)
            landmark_estimated_positions = [[landmark_estimated_positions[i][1],landmark_estimated_positions[i][0]] for i in range(len(landmark_estimated_positions))]
            landmark_estimated_stdevs = landmark_estimated_stdevs.diagonal().reshape(int(len(landmark_estimated_ids)),2)
            landmark_estimated_stdevs = [[landmark_estimated_stdevs[i][1],landmark_estimated_stdevs[i][0]] for i in range(len(landmark_estimated_stdevs))]
            # create message
            msg = Message(
                id = self.count,
                timestamp = time0,
                start = True,
                
                landmark_ids = ids,
                landmark_rs = distances,
                landmark_alphas = [angle - math.pi/2 for angle in angles],
                landmark_positions = positions,

                landmark_estimated_ids = landmark_estimated_ids,
                landmark_estimated_positions = landmark_estimated_positions,
                landmark_estimated_stdevs = landmark_estimated_stdevs,

                robot_position = np.array([-slam_position[1], slam_position[0]]),
                robot_theta = slam_position[2] + math.pi/2,
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
            self.input.addstr(22, 0, "Error 1")
            self.close()

    def controlRobot(self, vehicle):
        c = self.input.getch()  # catch keyboard event
        if c in (
            curses.KEY_RIGHT,
            curses.KEY_LEFT,
            curses.KEY_UP,
            curses.KEY_DOWN
        ):
            self.react(c)
            vehicle.move(self.speed, self.turn)  # modify movement
            self.input.addstr(1, 0, f'speed: {self.speed:4d}, turn: {self.turn:4d}')
        elif c == ord('p'):
            self.speed = 0
            self.turn = 0
            vehicle.stop()  # stop movement
            pos = vehicle.position
            self.input.addstr(5, 0, f'x: {pos.x:5.2f} m, y: {pos.y:5.2f} m, o: {pos.o:4.0f} °')

        elif c in (ord('q'), 27):
            self.input.addstr(7, 0, f'Stopping')
            vehicle.stop()  # finally stop movement
            landmarks = np.array(self.slam.mu[3:]) * 100
            landmarks = landmarks.reshape(int(landmarks.shape[0]/2), 2)
            landmarks = np.append(landmarks, np.array(list(self.slam.map.keys())).reshape(landmarks.shape[0],1), axis=1)
            np.savetxt("landmarks.csv", landmarks, delimiter=",")
            self.close()

    ##### -------OUR CODE--------- ######

    def run(self):
        try:
            vehicle = ev3.TwoWheelVehicle (
                self.radius,  # radius_wheel
                self.tread,  # tread
                protocol=ev3.USB
            ) 

            timepoint1 = time.time()
            timepoint2 = time.time()
            while True:
                try:
                    pool = ThreadPool(processes=2)
                    pool.apply_async(self.controlRobot(vehicle))
                    if timepoint2 - timepoint1 >= 0.2:
                        timepoint1 = time.time()
                        watch_results = pool.apply_async(self.checkDistance, (vehicle,))
                        self.input.addstr(2, 0, f'distance: {watch_results.get()}')
                    pool.close()
                    timepoint2 = time.time()
                                
                except Exception as e:
                    print(e) 
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
        # if char is not None:
        #     print("char:", char)

if __name__ == '__main__':
    curses.wrapper(Main)
    # main = Main()