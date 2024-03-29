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
from slam import EKFSLAM
import math
from multiprocessing.pool import ThreadPool
from imageProcessing import automatic_brightness_and_contrast, detectRedCircle, detectGreenCircle
import curses
from discretizer import Discretizer
from a_star import A_Star
import matplotlib.pyplot as plt
from matplotlib import colors
from decimal import Decimal
from collections import namedtuple

MotorPositions = namedtuple('MotorPositions', [
        'left',
        'right'
])


class Main():
    """
    Contains the main robot loop
    """
    def __init__(self, stdscr) -> None:
        sys.excepthook = self.except_hook
        self.input = stdscr
        self.keypress_listener = KeypressListener()
        self.publisher = Publisher()
        self.camera = Camera()
        self.camera_height = 0.248
        self.count = 0
        self.slam = EKFSLAM(0.6,0.6,1,60*np.pi/180)
        # self.tread = 0.13047   # tread - Floor
        # self.radius = 0.02775  # radius - Floor 
        # self.radius = 0.0224  # radius - Carpet
        # self.tread = 0.1116   # tread - Carpet
        self.radius = 0.02705625    # radius - new Floor
        self.tread = 0.12796984462  # tread - new Floor

        self.seen_ids = {}
        self.markerPositions_3d = []                                                                                                                                                                                                                                                                                                                                                                                                                            
        self.markerPositions_2d = []
        self.exploredMarkers = []
        self.tvec_sol = np.loadtxt(r'./tvec_sol.csv',delimiter=",",dtype=float)
        self.rvec_sol = np.loadtxt(r'./rvec_sol.csv',delimiter=",",dtype=float)
        self.green_positions = []
        self.red_positions = []
        self.start = []
        self.end = []
        self.returning_point = []
        self.middle_point = []
        self.grid_size = 2
        self.marker_distance = .35 #.5 
        self.exploration = True
        self.stop_vehicle = False
        self.turning = False
        self.free = False
        self.finished = False

        self.run()

    def transformImage(self, img, robot_position,vehicle):
        """
        Image processing pipeline has the following steps:
        1. Enhance image contrast
        2. Detect red and green discs
        3. Detect aruco markers 
        4. Estimate the position of the aruco markers
        5. Estimate the position of the discs
        """
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
            # Get keypoints
            img = automatic_brightness_and_contrast(img)
            keypoints_red, thresh_red = detectRedCircle(img)
            keypoints_green, thresh_green = detectGreenCircle(img)
            keypoints = keypoints_green + keypoints_red
            
            # detect aruco markers
            corners, ids, rejected_img_points = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

            if ids is not None:
                self.markerPositions_2d = []
                self.markerPositions_3d = []
                for j in range(0, len(ids)):
                    if ids[j] == 0 or ids[j] > 1000:
                        continue

                    if not(str(ids[j][0]) in list(self.seen_ids.keys())):
                        self.seen_ids[str(ids[j][0])] = 1
                    else:
                        self.seen_ids[str(ids[j][0])] = self.seen_ids[str(ids[j][0])] + 1

                    if self.seen_ids[str(ids[j][0])] < 5:
                        continue
                    marker_box = np.array([[int(corners[j][0][0][0]), int(corners[j][0][0][1])], [int(corners[j][0][2][0]), int(corners[j][0][2][1])]])
                    # cv2.rectangle(img, (int(corners[j][0][0][0]), int(corners[j][0][0][1])), (int(corners[j][0][2][0]), int(corners[j][0][2][1])), color = (0, 255, 0), thickness=2)
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[j], 0.048, self.camera.camera_matrix, self.camera.dist_coeffs)
                    # (rvec - tvec).any()

                    hypothenuse = np.linalg.norm(tvec)
                    
                    distance = np.sqrt(hypothenuse**2 - self.camera_height**2)
                    distances.append(distance)
                    # pos_x = tvec[0][0][0]
                    # pos_y = np.sqrt(distance**2 - pos_x**2)
                    pos_y = -tvec[0][0][0]
                    pos_x = np.sqrt(distance**2 - pos_y**2)

     

                    angle = np.arctan2(pos_y,pos_x) % (2*np.pi)
                    angles.append(angle)

                    direction_global_coords = rotation_matrix @ np.array([pos_x,pos_y]) 
                    pos_x = direction_global_coords[0]
                    pos_y = direction_global_coords[1]

                    positions.append([pos_x + robot_position[0], pos_y + robot_position[1]])
                    cv2.aruco.drawDetectedMarkers(img, corners)
                    cv2.drawFrameAxes(img, self.camera.camera_matrix, self.camera.dist_coeffs, rvec, tvec, 0.048)

                    middle_point = corners[j][0][0] + (corners[j][0][2]- corners[j][0][0])/2
                    self.markerPositions_2d.append([middle_point[0], middle_point[1]])
                    self.markerPositions_3d.append([pos_x + robot_position[0], pos_y + robot_position[1], 0])
                    
                    #img = cv2.circle(img, self.markerPositions_2d[j], 2, (0,0,0), 20)
                    
                    accepted_ids.append(ids[j])
                
            else:
                ids = []

            discs = []



            # if len(self.markerPositions_2d) > 0:
            #     self.markerPositions_2d = np.array(self.markerPositions_2d)
            #     self.markerPositions_2d = self.markerPositions_2d.reshape(self.markerPositions_2d.shape[0],2)
            #     self.markerPositions_3d = np.array(self.markerPositions_3d)
            #     self.markerPositions_3d = self.markerPositions_3d.reshape(self.markerPositions_3d.shape[0],3)
            #     ret, self.rvec_sol, self.tvec_sol = cv2.solvePnP(self.markerPositions_3d, self.markerPositions_2d, self.camera.camera_matrix, self.camera.dist_coeffs)
            #     np.savetxt("rvec_sol.csv", self.rvec_sol, delimiter=",")
            #     np.savetxt("tvec_sol.csv", self.tvec_sol, delimiter=",")


 
            for i, keypoint in enumerate(keypoints):
                x = int(keypoint.pt[0])
                y = int(keypoint.pt[1])
                s = keypoint.size
                r = int(math.floor(s/2))

                
                uvPoint = (x,y,1)
                cameraMatrix_inv = np.linalg.inv(self.camera.camera_matrix)
                dst, _ = cv2.Rodrigues(self.rvec_sol)
                dst_inv = np.linalg.inv(dst)
                leftSideMat  = dst_inv @ cameraMatrix_inv @ uvPoint
                rightSideMat = dst_inv @ self.tvec_sol
                s = (rightSideMat[2])/leftSideMat[2]

                
                marker_worldCoords = dst_inv @ (s * cameraMatrix_inv @ uvPoint - self.tvec_sol)
                marker_worldCoords = rotation_matrix @ np.array([marker_worldCoords[0],marker_worldCoords[1]])
                position = [marker_worldCoords[0]+ robot_position[0],marker_worldCoords[1]+ robot_position[1]]     
                distance = np.sqrt(marker_worldCoords[0]**2 + marker_worldCoords[1]**2)
                print("HERE!")
                print("9")
                if not self.exploration and not self.turning:    
                    if keypoint in keypoints_green:
                        cv2.circle(img,(x, y), r, color=(255,255,255)) # green circle -> white
                        if distance < 0.4:
                            if not np.any([np.allclose(position,old_markers,atol=0.5,rtol=0) for old_markers in self.green_positions]):
                                self.green_positions.append(position)

                    else:
                        cv2.circle(img,(x, y), r, color=(0,0,0)) # red circle -> black
                        print("red circle stuff!!!")
                        if distance < 0.4:
                            if not np.any([np.allclose(position,old_markers,atol=0.5,rtol=0) for old_markers in self.red_positions]):
                                self.free = False
                                # if not vehicle._current_movement is None:
                                self.input.addstr(14, 0, f'Stopping vehicle: {i}\t\t\t')
                                self.stop_vehicle = True
                                print("STOPPING BECAUSE OF RED DISC!")
                                vehicle.stop(brake=True)

                                if vehicle._current_movement is None:
                                    self.stop_vehicle = False
                                    self.red_positions.append(position)
                                else:
                                    self.red_positions.append(position)
                                    vehicle._target_motor_pos = MotorPositions(Decimal(vehicle._current_movement['last_motor_pos'][0]),Decimal(vehicle._current_movement['last_motor_pos'][1]))
                                    vehicle._current_movement['speed_left'] = 0
                                    vehicle._current_movement['speed_right'] = 0
                                    # diff_pos_l = vehicle._current_movement['last_motor_pos'][0] - vehicle._current_movement['target_motor_pos'][0]
                                    # diff_pos_r = vehicle._current_movement['last_motor_pos'][1] - vehicle._current_movement['target_motor_pos'][1]
                                    vehicle._target_motor_pos = MotorPositions(vehicle._target_motor_pos.left, vehicle._target_motor_pos.right)

                                    # vehicle._target_motor_pos = MotorPositions(Decimal(vehicle._current_movement['last_motor_pos'][0]),Decimal(vehicle._current_movement['last_motor_pos'][1]))
                                    vehicle._current_movement['step1_left'] = 0
                                    vehicle._current_movement['step2_left'] = 0
                                    vehicle._current_movement['step3_left'] = 0
                                    vehicle._current_movement['step1_right'] = 0
                                    vehicle._current_movement['step2_right'] = 0
                                    vehicle._current_movement['step3_right'] = 0

                                    vehicle._current_movement = None
                                    vehicle.drive_turn(2, 0.0, speed=20).start(thread=False)
                                    self.red_positions.append(position)
                                    self.stop_vehicle = False

                            else:
                                print("Tried to add new disk but disk already exists!")
                

                distance = 0  
                angle = 0    
                discs.append([position, distance, angle])
                print("finished!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("10")

        else:
            ids = []
            self.markerPositions_2d = []
            self.markerPositions_3d = []


        return img, accepted_ids, positions, distances, angles, discs

    def doVisualizer(self, img, ids, positions, distances, angles, discs):
        """
        Create and send the message required by the visualizer.
        """
        try:
            ####################
            # get the time
            time0 = time.time()

            slam_position, slam_sigma = self.slam.get_robot_pose()
            # slam_position = vehicle.position
            landmark_estimated_positions, landmark_estimated_stdevs = self.slam.get_landmark_positions()

            landmark_estimated_ids = list(self.slam.map.keys())
            landmark_estimated_positions = landmark_estimated_positions.reshape(int(len(landmark_estimated_ids)),2)
            # landmark_estimated_positions = [[landmark_estimated_positions[i][1],landmark_estimated_positions[i][0]] for i in range(len(landmark_estimated_positions))]
            landmark_estimated_stdevs = landmark_estimated_stdevs.diagonal().reshape(int(len(landmark_estimated_ids)),2)
            # landmark_estimated_stdevs = [[landmark_estimated_stdevs[i][1],landmark_estimated_stdevs[i][0]] for i in range(len(landmark_estimated_stdevs))]
            # create message
            distances.extend([disc[1] for disc in discs])
            angles.extend([disc[2] for disc in discs])
            # angles = [angle - math.pi/2 for angle in angles]

            positions.extend([disc[0] for disc in discs])
            ids.extend([420+disc_id for disc_id, disc in enumerate(discs)])

            landmark_estimated_positions = landmark_estimated_positions.tolist()
            landmark_estimated_positions.extend([disc[0] for disc in discs])

            landmark_estimated_ids.extend([420+disc_id for disc_id, disc in enumerate(discs)])

            landmark_estimated_stdevs = landmark_estimated_stdevs.tolist()
            landmark_estimated_stdevs.extend([[0,0] for disc in discs])

            msg = Message(
                id = self.count,
                timestamp = time0,
                start = True,
                
                landmark_ids = ids,
                landmark_rs = distances,
                landmark_alphas = angles,
                # landmark_positions = [[-position[1], position[0]] for position in positions],
                landmark_positions = positions,

                landmark_estimated_ids = landmark_estimated_ids,
                # landmark_estimated_positions = [[-landmark_estimated_position[1],landmark_estimated_position[0]]  for landmark_estimated_position in landmark_estimated_positions],
                landmark_estimated_positions = landmark_estimated_positions,
                landmark_estimated_stdevs = landmark_estimated_stdevs,

                robot_position = np.array([slam_position[0], slam_position[1]]),
                # robot_position = np.array([-slam_position[1], slam_position[0]]),
                # robot_theta = np.where(slam_position[2] + np.pi/2 > np.pi, slam_position[2] + np.pi/2 - 2*np.pi, slam_position[2] + np.pi/2),
                robot_theta = slam_position[2],
                robot_stdev = slam_sigma.diagonal(),
            )
            self.count += 1

            # pickle message
            msg_str = jsonpickle.encode(msg)

            # publish message and image
            self.publisher.publish_img(msg_str, img)

        except Exception as e:
            self.input.addstr(20, 0, f'Visualiter Exception: {e}\t\t\t')


    def doSLAM(self, vehicle):
        """
        Call the prediction and correction step of the SLAM object. Initilize message sending procedure.
        """
        while True:
            try:
                pos, sigma = self.slam.get_robot_pose()
                x, y, angle = pos
                new_pos =[x,y,angle]
                # get image from the camera
                _, raw_img = self.camera.read()
                # imaginary processing
                img, ids, positions, distances, angles, discs = self.transformImage(raw_img, new_pos,vehicle)
                

                uncertainty = [0.6,0.6]
                self.slam.predict(vehicle, self.tread)
        
                if len(ids) > 0:
                    for i, id in enumerate(ids):
                        if self.slam.id_never_seen_before(id[0]):
                            self.slam.add_landmark(id[0],positions[i],uncertainty)
                        self.slam.correction_direct(id[0],(distances[i],angles[i]))
                self.input.addstr(0, 0, f'Robot Data.\t\t\t')
                self.input.addstr(1, 0, f'Position: [{round(x,2)}, {round(y,2)}]\t\t\t')
                self.input.addstr(2, 0, f'Angle: {round(np.degrees(angle),2)}\t\t\t')
                self.input.refresh()
                self.doVisualizer(img, ids, positions, distances, angles, discs)
                # self.saveMap()
            except Exception as e:
                self.input.addstr(20, 0, f'SLAM Exception: {e}\t\t\t')

    def saveMap(self):
        """
        Transform the landmark coordinates obtained via SLAM and save them as a CSV file.
        """
        landmarks = np.array(self.slam.mu[3:]) * 100
        landmarks = landmarks.reshape(int(landmarks.shape[0]/2), 2)
        object_ids = []
        for key in np.array(list(self.slam.map.keys())):
            if key > 33 and key < 99:
                object_ids.append(6)
            elif key <= 33:
                object_ids.append(-1)
            else:
                object_ids.append(int(key % 3))
        landmarks = np.append(landmarks, np.array(object_ids).reshape(int(landmarks.shape[0]),1), axis=1)

        np.savetxt("landmarks.csv", landmarks, delimiter=",")
        return landmarks

    def filterPositions(self, marker_list):
        """
        Given a list of landmarks filter only relevant (in front of us) landmarks.
        """
        x_bot, y_bot, theta_bot = self.slam.get_robot_pose()[0]
        dx = marker_list[:,1] - x_bot
        dy = marker_list[:,2] - y_bot

        angles = np.degrees(((np.arctan2(dy,dx) % (2*np.pi))- theta_bot) % (2*np.pi))
        angles = np.where(angles > 180, angles - 360, angles)
        angles = angles.reshape(angles.shape[0], 1)
        marker_list = np.hstack([marker_list, angles])

        marker_list = marker_list[((marker_list[:,4] < 90) & (marker_list[:,4] > -90))]

        return marker_list


    def calcAngle(self, distances, angles):
        """
        Calculate the distance weighted sum of a provided list of distances and landmarks
        """
        distances = distances / np.sum(distances)
        angles = angles * (distances)
        if len(angles) == 0: angle = 0
        else: angle = np.sum(angles)/2
        return angle

    def explore(self, vehicle):
        """
        Do exploration step.
        """
        counter = 0
        self.start = self.slam.get_robot_pose()[0][:2]
        while True:
            try:               
                self.input.addstr(4, 0, f'Phase: Exploring')
                self.input.refresh()
                self.input.refresh()
                c = self.input.getch()
                self.input.refresh()
                if c == ord('q'):
                    self.input.addstr(4, 0, f'Stopping: Exploration')
                    self.input.refresh()
                    time.sleep(4)
                    break
                markers, _ = self.slam.get_landmark_positions()
                if isinstance(markers,list): markers = np.array(markers)
                markers = markers.reshape(int(markers.shape[0]/2), 2)
                ids = np.array(list(self.slam.map.keys()))
                ids = ids.reshape(ids.shape[0], 1)
       
                marker_list = np.hstack([ids, markers])

                left_markers = marker_list[np.where(marker_list[:,0] % 3 == 0)[0],:]
                middle_markers = marker_list[np.where(marker_list[:,0] % 3 == 1)[0],:]
                right_markers = marker_list[np.where(marker_list[:,0] % 3  == 2)[0],:]

                pos, _ = self.slam.get_robot_pose()
                distances_left = np.sqrt((pos[0] - left_markers[:,1])**2 + (pos[1] - left_markers[:,2])**2)
                distances_right = np.sqrt((pos[0] - right_markers[:,1])**2 + (pos[1] - right_markers[:,2])**2)
                distances_middle = np.sqrt((pos[0] - middle_markers[:,1])**2 + (pos[1] - middle_markers[:,2])**2)

                distances_left = distances_left.reshape(distances_left.shape[0], 1)
                left_markers = np.hstack([left_markers, distances_left])
                distances_right = distances_right.reshape(distances_right.shape[0], 1)
                right_markers = np.hstack([right_markers, distances_right])
                distances_middle = distances_middle.reshape(distances_middle.shape[0], 1)
                middle_markers = np.hstack([middle_markers, distances_middle])
                
                right_markers = right_markers[right_markers[:, 3].argsort()]
                left_markers = left_markers[left_markers[:, 3].argsort()]
                middle_markers = middle_markers[middle_markers[:, 3].argsort()]

                try:
                    self.input.addstr(18, 0, f'{right_markers[0][3]}, {left_markers[0][3]}, {middle_markers[0][3]}\t\t\t')
                except Exception as e:
                    pass

                markers_infront_left = self.filterPositions(left_markers)
                markers_infront_right = self.filterPositions(right_markers)
                markers_infront_middle = self.filterPositions(middle_markers)

                markers_infront_right = markers_infront_right[markers_infront_right[:,3]<self.marker_distance]
                markers_infront_left = markers_infront_left[markers_infront_left[:,3]<self.marker_distance]
                markers_infront_middle = markers_infront_middle[markers_infront_middle[:,3]<self.marker_distance]

                if ((markers_infront_right.shape[0] == 0) & (markers_infront_left.shape[0] == 0)):
                    self.input.addstr(5, 0, f'No markers in range - driving straight.\t\t\t')
                    self.input.refresh()
                    angle=0.0
                elif (markers_infront_right.shape[0] == 0):
                    self.input.addstr(5, 0, f'Right boarder in front - turn anticlockwise.\t\t\t')     # anticlockwise
                    self.input.refresh()
                    angle = -90
                elif (markers_infront_left.shape[0] == 0):
                    self.input.addstr(5, 0, f'Left boarder in front - turn clockwise.\t\t\t')    # clockwise
                    self.input.refresh()
                    angle = 90
                else:
                    self.input.addstr(5, 0, f'Markers detected - calculate new direction.\t\t\t')
                    self.input.refresh()
                    markers_infront_right = markers_infront_right[:min(len(markers_infront_right), len(markers_infront_left))]
                    markers_infront_left = markers_infront_left[:min(len(markers_infront_right), len(markers_infront_left))]
            
                    marker_list = np.concatenate([markers_infront_right, markers_infront_left])
                    angle = self.calcAngle(marker_list[:,3], marker_list[:,4])
                
                length = .1
                speed = 20

                if len(right_markers) > 0:
                    right_dist = right_markers[0][3]
                else: right_dist = np.inf
                if len(left_markers) > 0:
                    left_dist = left_markers[0][3]
                else: left_dist = np.inf
                if len(middle_markers) > 0:
                    middle_dist = middle_markers[0][3]
                else: middle_dist = np.inf


                if angle == 0:
                    self.input.addstr(6, 0, f'Turn by {angle} degrees.\t\t\t')
                    self.input.addstr(7, 0, f'Drive by {length * 100}cm.\t\t\t')
                    self.input.refresh()
                    vehicle.drive_straight(length, speed=speed, brake=True).start(thread=False)
                elif angle > 15 or angle < -15:
                    self.input.addstr(6, 0, f'Turn by {angle} degrees.\t\t\t')
                    self.input.addstr(7, 0, f'Drive by {length * 100}cm.\t\t\t')
                    self.input.refresh()
                    vehicle.drive_turn(angle, 0.0, brake=True).start(thread=False)
                    vehicle.drive_straight(length, speed=speed, brake=True).start(thread=False)
                else:
                    self.input.addstr(6, 0, f'Turn by {angle} degrees => Too close for turning\t\t\t')
                    self.input.addstr(7, 0, f'Drive by {length * 100}cm.\t\t\t')
                    vehicle.drive_straight(length, speed=speed, brake=True).start(thread=False)

                # time.sleep(0.5)

                if counter % 5 == 0: 
                    self.input.refresh() 
                    if counter == 0:
                        vehicle.drive_straight(0.2, speed=speed, brake=True).start(thread=False)
                        vehicle.drive_turn(360,0.0, brake=True, speed = 10).start(thread=False)
                        counter += 1
                        continue

                    self.input.addstr(19, 0, f'{right_dist}, {left_dist}, {middle_dist}\t\t\t')

                    if min([right_dist, left_dist, middle_dist]) > 0.15:
                        self.input.addstr(20, 0, f'Turn by 360 degrees.\t\t\t')
                        if counter % 2 == 0:
                            vehicle.drive_turn(360,0.0, brake=True).start(thread=False)
                        else:
                            vehicle.drive_turn(-360,0.0, brake=True).start(thread=False)
                    else:
                        self.input.addstr(20, 0, f'Dont turn by 360 degrees.\t\t\t')
                        counter -= 1
                counter += 1
                    
            except Exception as e:
                print(e)
                self.input.addstr(10, 0, f'Exploration Exception: {e}\t\t\t')
                self.input.refresh()
                break
        return self.saveMap()    
        

    def solve_task(self, vehicle):  
        """
        Hub calling the exploration step and later the driving step.
        """
        landmarks = self.explore(vehicle)
        self.input.addstr(4, 0, f'Done with exploration phase.\t\t\t')
        self.input.refresh()
        landmarks = np.loadtxt("landmarks.csv", delimiter=",")
        self.input.addstr(4, 0, f'Drive to new starting position.\t\t\t')
        self.input.refresh()
        self.drive_track(vehicle,landmarks)

    def plan_line(self, start, end):
        """
        Given two points compute the angle and distance between them.
        """
        distance = (np.sqrt((start[0] - end[0])**2 + (start[1]-end[1])**2)) * self.grid_size/ 100
        angle = np.arctan2(end[0]-start[0], end[1]-start[1])
        hit_obstacle = False
        # if np.degrees(angle) < 10:
        #     angle = 2*np.pi - angle 
        if distance == 0: distance = 0.01
        return distance, angle, hit_obstacle

    def to_map_space(self, point, numpy_array=False):
        """
        Transform the coordinates from the SLAM coordinate system to the map coordinate system
        """
        if not numpy_array:
            point = np.array(point)
        point = (np.array([1,1]) + (point*100 + self.middle_point) / self.grid_size).astype(int)
        return list(point)

    def travel_for(self,distance,vehicle):
        try:
            self.input.addstr(4, 0, f'Free: {self.free}.\t\t\t')
            self.free = True
            pool2 = ThreadPool(processes=1)
            pool2.apply_async(self.drive_straight, (vehicle,))
            motor_l_total, motor_r_total = vehicle.motor_pos
            old_angles = [motor_l_total,motor_r_total]
            l_dist_list = []
            r_dist_list = []
            while True:
                l,r,old_angles = self.get_motor_movement_2(old_angles,vehicle)
                l_dist_list.append(l)
                r_dist_list.append(r)
                distance_traveled = (np.sum(l_dist_list) + np.sum(r_dist_list)) / 2
                if (distance_traveled >= distance) or (self.free == False):
                    self.free = False
                    pool2.close()
                    self.input.addstr(4, 0, f'Free: {self.free}.\t\t\t')
                    break
        except Exception as e:
            print(e)

    def get_motor_movement_2(self,old_angles, vehicle):
        try:
            """Read out the movements of the left and right wheels from the motor rotation."""
            motor_l_total, motor_r_total = vehicle.motor_pos
            motor_l = motor_l_total - old_angles[0]
            motor_r = motor_r_total - old_angles[1]
            l = motor_l * np.pi * 0.056 / 360
            r = motor_r * np.pi * 0.056 / 360
            return l, r, [motor_l_total, motor_r_total]
        except Exception as e:
            print(e)

    def drive_to_start(self, vehicle):
        """
        Drive from the current position to a position from which to start the driving phase.
        """
        self.input.addstr(9, 0, f"Returning to: {self.start} \t\t\t'")
        time.sleep(2)
        pos_x, pos_y , theta = self.slam.get_robot_pose()[0]
        self.input.addstr(15, 0, f'current direction 1: {np.degrees(2*np.pi - theta)}\t\t\t')
        # if theta > np.pi: vehicle.drive_turn(np.degrees(-theta), 0.0, brake=True).start(thread=False)
        # else: vehicle.drive_turn(np.degrees(2*np.pi - theta), 0.0, brake=True).start(thread=False)
        #vehicle.drive_turn(np.degrees(2*np.pi - theta), 0.0, brake=True).start(thread=False)
        current_angle = np.degrees(theta)
        if (360-current_angle > 2) or (360-current_angle < -2):
            vehicle.drive_turn(360-current_angle, 0.0, brake=True).start(thread=False)

        time.sleep(4)
        pos_x, pos_y , theta = self.slam.get_robot_pose()[0]
        self.returning_point = self.to_map_space([pos_x, pos_y])
        self.returning_point = [self.returning_point[1], self.returning_point[0]]
        self.input.addstr(10, 0, f"From:  {self.returning_point} \t\t\t")

        distance, angle, hit_obstacle = self.plan_line(self.returning_point, self.start)
        self.input.addstr(11, 0, f"Turn: {np.degrees(angle)} / Distance: {distance} \t\t\t")
        if (np.degrees(angle) > 2) or (np.degrees(angle) < -2):
            vehicle.drive_turn(np.degrees(angle), 0.0, brake=True).start(thread=False)
        vehicle.drive_straight(distance, speed=25, brake=False).start(thread=False)

        time.sleep(2)
        # pos_x, pos_y , theta = self.slam.get_robot_pose()[0]
        # self.input.addstr(15, 0, f'current direction 2: {np.degrees(theta)}\t\t\t')
        # # if theta > np.pi: vehicle.drive_turn(np.degrees(-theta), 0.0, brake=True).start(thread=False)
        # # else: vehicle.drive_turn(np.degrees(2*np.pi - theta), 0.0, brake=True).start(thread=False)
        # # vehicle.drive_turn(np.degrees(2*np.pi - theta), 0.0, brake=True).start(thread=False)
        # current_angle = np.degrees(theta)
        # vehicle.drive_turn(360-current_angle, 0.0, brake=True).start(thread=False)
        if (np.degrees(-angle) > 2) or (np.degrees(-angle) < -2):
            vehicle.drive_turn(np.degrees(-angle), 0.0, brake=True).start(thread=False)


        self.input.addstr(15, 0, f'Went home\t\t\t')

    def drive_track(self,vehicle,landmarks):
        """
        Plan and drive the track
        """

        self.input.addstr(4, 0, f'Phase: Planning \t\t\t')
        self.input.refresh()
        self.grid_size=3
        size_x = int((max(np.abs(landmarks[:,0])))+10)
        size_y = int((max(np.abs(landmarks[:,1])))+10)
        world_coords=[max(size_x, size_y)*2, max(size_x, size_y)*2]
        self.input.addstr(5, 0, f'Discritize Map.\t\t\t')
        self.input.refresh()
        discretizer = Discretizer(landmarks, grid_size=self.grid_size, world_coords=world_coords)
        discretizer.createMap(surrounding=6)

        self.input.addstr(5, 0, f'Plan Path.\t\t\t')
        self.input.refresh()         

        self.middle_point = np.array([int(world_coords[0]/2), int(world_coords[1]/2)])
        self.start = self.to_map_space(self.start)

        # self.start =  [self.start[1],  self.start[0]]
        self.drive_to_start(vehicle)
        
        current_pos = list(np.array([1,1]) + ((self.slam.get_robot_pose()[0][:2]*100 + self.middle_point)/self.grid_size).astype(int))
        current_pos = [current_pos[1],current_pos[0]]
        self.input.addstr(15, 0, f'CURRENT POSITION: {current_pos}\t\t\t')      

        self.input.addstr(15, 0, f'Starting Astar \t\t\t')
        self.returning_point = [self.returning_point[0]+1, self.returning_point[1]+1]
        self.end = current_pos
        a_star = A_Star(current_pos, self.returning_point, self.end, discretizer.world_map.transpose(1,0))
        path = a_star.find_path()
        trajectory = a_star.plan_trajectory()
        self.input.addstr(15, 0, f'SAVING MAP!!!!!!!!!!!\t\t\t')
        cmap = colors.ListedColormap(['white', 'green', 'red', 'blue', 'black', 'orange', 'brown', 'purple', 'gray'])
        plt.figure('Shortest path')
        plt.imshow(a_star.world_map + discretizer.mask.transpose(1,0), origin="lower", cmap=cmap, vmin=0, vmax=7)
        plt.plot(path[1],path[0], color="black", lw=2)
        plt.plot(a_star.start[1]+1,a_star.start[0]+1,'g.',markersize=20)
        plt.plot(a_star.goal[1]+1,a_star.goal[0]+1,'r.',markersize=20)
        plt.plot(self.start[1]+1, self.start[0]+1, 'y.',markersize=20 )
        plt.plot(self.returning_point[1],self.returning_point[0], 'r.',markersize=20,alpha=0.5 )
        self.input.addstr(15, 0, f'SAFE MAP!!!!!!!!!!!!\t\t\t')
        plt.savefig('initial_map.png')

        self.input.addstr(4, 0, f'Phase: Driving\t\t\t')
        self.input.refresh()
        x, y, theta = self.slam.get_robot_pose()[0]
        trajectory[0][3] = trajectory[0][3] - 90 - np.degrees(theta)
        if trajectory[0][3] > 180: trajectory[0][3] = trajectory[0][3] - 360
        elif trajectory[0][3] < -180: trajectory[0][3] = trajectory[0][3] + 360

        len_red_discs = 0
        len_green_discs = 0
        plt.figure('Shortest path')

        while True:
            c = self.input.getch()
            self.input.refresh()
            if c == ord('s'):
                self.exploration = False
                print("1")
                self.input.addstr(4, 0, f'Start Racing Mode')
                break
        print("2")
        random_colors = ["red", "blue", "green", "purple"]
        random_index = 0
        last_move = 0
        added_red_disc = False
        print("3")

        while True:
            try:
                print("4")
                while (self.stop_vehicle == True):
                    self.input.addstr(24, 0, f'Waiting...\t\t\t')
                    print("waiting to be released!")
                    self.input.refresh()
                    print("5")

                while len_red_discs < len(self.red_positions):
                    added_red_disc = True
                    # print("Adding Disk ", len_red_discs)
                    check = discretizer.add_red_disc(np.array(self.red_positions[len_red_discs]), surrounding=12)
                    if check: 
                        self.input.addstr(20+len_red_discs, 0, f'Added new red disc at: {self.to_map_space(self.red_positions[len_red_discs])}\t\t\t')

                    len_red_discs += 1
                print("6")
                if added_red_disc:
                    print("red discs stuff")
                    plt.imshow(discretizer.world_map.transpose(1,0), origin="lower", cmap=cmap, vmin=0, vmax=8)
                    plt.savefig("second_map.png")
                    robot_pose = self.slam.get_robot_pose()[0]      
                    current_pos = list(np.array([1,1]) + ((robot_pose[:2]*100 + self.middle_point)/self.grid_size).astype(int))
                    current_pos = [current_pos[1],current_pos[0]] 
                    self.input.addstr(15, 0, f'Starting Astar \t\t\t')

                    a_star = A_Star(current_pos, self.returning_point, self.end, discretizer.world_map.transpose(1,0))
                    path = a_star.find_path()
                    plt.plot(path[1],path[0], color="yellow", lw=2)

                    self.input.addstr(15, 0, f'Done with Astar \t\t\t')
                    self.input.addstr(15, 0, f'Robot angle {np.degrees(robot_pose[2])} \t\t\t')
                    trajectory = a_star.plan_trajectory()

                    new_angle = trajectory[0][3] - 90 + np.degrees(robot_pose[2])
                    # new_angle = trajectory[0][3]
                    if new_angle > 180: new_angle = new_angle - 360
                    elif new_angle < -180: new_angle = new_angle + 360

                    trajectory[0][3] = new_angle
                    last_move = 0

                    self.input.addstr(14, 0, f'First trajectory turn {trajectory[0][3]} \t\t\t')
                    plt.plot(current_pos[1]+1, current_pos[0]+1, 'y.',markersize=20)

                    random_index += 1
                    random_index %= 4
                    added_red_disc = False
                print("7")
                print("Moves: ", len(trajectory))
                if len(trajectory) > 0:
                    move = trajectory.pop(0)
                else: 
                    break

                
                speed = 30
                turning_speed = 15
                ramp_up = 80
                ramp_down = 40
                print("Start move")
                print("8")
                if -move[3]+last_move < 15 and -move[3]+last_move > -15:
                    self.input.addstr(6, 0, f'Turn by {np.round(-move[3]+last_move,2)} degrees. => Too small\t\t\t')
                    self.input.refresh()
                    vehicle.drive_straight(move[2]*self.grid_size, speed=speed, ramp_up=ramp_up, ramp_down=ramp_down, brake=True).start(thread=False)
                    # self.travel_for(move[2]*self.grid_size,vehicle)
                    last_move += -move[3]
                    # vehicle.drive_turn(-move[3], 0.0).start(thread=False)
                else:
                    self.input.addstr(6, 0, f'Turn by {np.round(-move[3] + last_move,2)} degrees.\t\t\t')
                    self.input.refresh()
                    self.turning = True
                    time.sleep(.1)
                    print("Turning: ", -move[3] + last_move)
                    # vehicle.stop()
                    # vehicle.move(0,0)
                    vehicle.drive_turn(-move[3] + last_move, 0.0, speed=turning_speed, brake=True).start(thread=False)
                    last_move = 0
                    self.turning = False
                    
                    self.input.addstr(6, 0, f'Drive by {np.round(move[2]*100,2)} cm.\t\t\t')
                    self.input.refresh()
                    vehicle.drive_straight(move[2]*self.grid_size, speed=speed, ramp_up=ramp_up, ramp_down=ramp_down,brake=True).start(thread=False)
                    # self.travel_for(move[2]*self.grid_size,vehicle)
                    last_move = 0
                plt.plot([move[0][1], move[1][1]], [move[0][0], move[1][0]], color=random_colors[random_index], lw=2)
            except Exception as e:
                print("move exception")
                self.input.addstr(35, 0, f'{vehicle._current_movement}')
                print(e)

        plt.imshow(discretizer.world_map.transpose(1,0), origin="lower", cmap=cmap, vmin=0, vmax=9)
        plt.savefig("add_red_map.png")
        self.input.addstr(15, 0, f'DONE \t\t\t')
        vehicle.drive_straight(0.1, speed=speed, brake=True, ramp_up=ramp_up, ramp_down=ramp_down).start(thread=False)
        # self.travel_for(move[2]*self.grid_size,vehicle)
        self.finished = True

    def drive_straight(self,vehicle):
        if self.free:
            vehicle.move(15,0)
            self.free = False
        else:
            pass

    def run(self):
        """
        Take care of threading.
        """
        try:
            vehicle = ev3.TwoWheelVehicle (
                self.radius,  # radius_wheel
                self.tread,  # tread
                protocol=ev3.USB
            ) 
            t1_slam = time.time()
            t2_slam = time.time()
            t1_ex = time.time()
            t2_ex = time.time()
            current_ids = []
            try:
                pool = ThreadPool(processes=2)
                pool.apply_async(self.solve_task, (vehicle,))
                t1_slam = time.time()
                result = pool.apply_async(self.doSLAM, (vehicle,))

                # while True:
                #     pool.apply_async(self.drive_straight, (vehicle,))
                #     if self.finished:
                #         break
                
                current_ids = result.get()
                pool.close()
                pool.join()
            except Exception as e:
                self.input.addstr(20, 0, f'Threading Exception: {e}\t\t\t')
            
        except Exception as e:
            self.input.addstr(20, 0, f'Vehicle Exception: {e}\t\t\t')

        vehicle.stop()
    
        # tidy up
        self.close()



    def except_hook(self, type, value, tb):
        self.close()

    def close(self):
        self.keypress_listener.close()
        self.camera.close()

if __name__ == '__main__':
    curses.wrapper(Main)
    # main = Main()
