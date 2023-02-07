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
from threading import Thread
from imageProcessing import automatic_brightness_and_contrast, detectRedCircle, detectGreenCircle


class Main():
    def __init__(self) -> None:

        self.keypress_listener = KeypressListener()
        self.publisher = Publisher()
        self.camera = Camera()
        self.camera_height = 0.248
        self.count = 0
        self.slam = EKFSLAM(0.6,0.6,1,60*np.pi/180)
        self.tread = 0.12832
        self.radius = 0.02968 
        self.seen_ids = {}
        self.markerPositions_3d = []                                                                                                                                                                                                                                                                                                                                                                                                                            
        self.markerPositions_2d = []
        self.exploredMarkers = []
        self.tvec_sol = np.loadtxt(r'./tvec_sol.csv',delimiter=",",dtype=float)
        self.rvec_sol = np.loadtxt(r'./rvec_sol.csv',delimiter=",",dtype=float)

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

                if keypoint in keypoints_green: cv2.circle(img,(x, y), r, color=(255,255,255)) # green circle -> white
                else: cv2.circle(img,(x, y), r, color=(0,0,0)) # red circle -> black

                uvPoint = (x,y,0)
                cameraMatrix_inv = np.linalg.inv(self.camera.camera_matrix)
                dst, _ = cv2.Rodrigues(self.rvec_sol)
                dst_inv = np.linalg.inv(dst)
                leftSideMat  = dst_inv @ cameraMatrix_inv @ uvPoint
                rightSideMat = dst_inv @ self.tvec_sol
                s = (rightSideMat[2])/leftSideMat[2]

                
                marker_worldCoords = dst_inv @ (s * cameraMatrix_inv @ uvPoint - self.tvec_sol)

                position = [marker_worldCoords[0], marker_worldCoords[1]]         
                distance = 0  
                angle = 0    
                discs.append([position, distance, angle])

        else:
            ids = []
        self.markerPositions_2d = []
        self.markerPositions_3d = []

        return img, accepted_ids, positions, distances, angles, discs

    def doVisualizer(self, img, ids, positions, distances, angles, discs):
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


    def doSLAM(self, vehicle):
        while True:
            try:
                pos, sigma = self.slam.get_robot_pose()
                x, y, angle = pos
                new_pos =[x,y,angle]
                # get image from the camera
                _, raw_img = self.camera.read()
                # imaginary processing
                img, ids, positions, distances, angles, discs = self.transformImage(raw_img, new_pos)

                uncertainty = [0.6,0.6]
                self.slam.predict(vehicle, self.tread)
        
                if len(ids) > 0:
                    for i, id in enumerate(ids):
                        if self.slam.id_never_seen_before(id[0]):
                            self.slam.add_landmark(id[0],positions[i],uncertainty)
                        self.slam.correction_direct(id[0],(distances[i],angles[i]))
                self.doVisualizer(img, ids, positions, distances, angles, discs)
            except Exception as e:
                print(e)

    def saveMap(self):
        try:
            landmarks = np.array(self.slam.mu[3:]) * 100
            landmarks = landmarks.reshape(int(landmarks.shape[0]/2), 2)
            object_ids = []
            for key in np.array(list(self.slam.map.keys())):
                object_ids.append(int(key % 3))
            landmarks = np.append(landmarks, np.array(object_ids).reshape(int(landmarks.shape[0]),1), axis=1)
            np.savetxt("landmarks.csv", landmarks, delimiter=",")
        except Exception as e:
            pass

    def calcAngle(self,marker_list, vehicle):
        # x_bot, y_bot, _ = vehicle.position
        x_bot, y_bot, theta_bot = self.slam.get_robot_pose()[0]
        print("X: ", x_bot)
        print("Y: ", y_bot)
        print("Theta: ", theta_bot)

        dx = marker_list[:,1] - x_bot
        dy = marker_list[:,2] - y_bot
        print("Marker List: ", marker_list)
        angles_infront = np.degrees(((np.arctan2(dy,dx) % (2*np.pi))- theta_bot)) # % (2*np.pi))
        print("Robot Angle: ", theta_bot)
        # angles_infront = angles_infront[((angles_infront < 90) | (angles_infront > 270))]
        angles_infront = angles_infront[((angles_infront < 90) & (angles_infront > -90))]
        print("Angles Infront!: ", angles_infront.astype(int))
        print("Markers infront of us!: ", angles_infront.shape[0])
        print("Markers behind: ", marker_list.shape[0] - angles_infront.shape[0])
        angles_infront = np.where(angles_infront > 180, angles_infront - 360, angles_infront)
        print()

        if len(angles_infront) == 0: angle = 0
        else: angle = np.sum(angles_infront)/2

        return angle

    def explore(self, vehicle):
        while True:
            try:               
                print("Start Exploring")
                markers, _ = self.slam.get_landmark_positions()
                markers = markers.reshape(int(markers.shape[0]/2), 2)
                ids = np.array(list(self.slam.map.keys()))
                ids = ids.reshape(ids.shape[0], 1)
                
                marker_list = np.hstack([ids, markers])

                left_markers = marker_list[np.where(marker_list[:,0] % 3 == 0)[0],:]
                right_markers = marker_list[np.where(marker_list[:,0] % 3  == 2)[0],:]
                
                print("Before everything!")
                print("Left Markers: ", left_markers.shape[0])
                print("Right Markers: ", right_markers.shape[0])
                print()

                pos, _ = self.slam.get_robot_pose()
                distances_left = np.sqrt((pos[0] - left_markers[:,1])**2 + (pos[1] - left_markers[:,2])**2)
                distances_right = np.sqrt((pos[0] - right_markers[:,1])**2 + (pos[1] - right_markers[:,2])**2)

                distances_left = distances_left.reshape(distances_left.shape[0], 1)
                left_markers = np.hstack([left_markers, distances_left])
                distances_right = distances_right.reshape(distances_right.shape[0], 1)
                right_markers = np.hstack([right_markers, distances_right])
                
                right_markers = right_markers[right_markers[:, 2].argsort()]
                left_markers = left_markers[left_markers[:, 2].argsort()]


                # right_markers = right_markers[right_markers[:,2]<.5]
                # left_markers = left_markers[left_markers[:,2]<.5]
                print("Close by markers (<.5)!")
                print("Left Markers: ", left_markers.shape[0])
                print("Right Markers: ", right_markers.shape[0])
                print()

                if ((right_markers.shape[0] == 0) & (left_markers.shape[0] == 0)):
                    print("Drive Straight - no markers close by")
                    angle=0.0
                elif (right_markers.shape[0] == 0):
                    print("Perpendicular Turn - Back to track")     # anticlockwise
                    angle = 90
                elif (left_markers.shape[0] == 0):
                    print("Perpendicular Turn - Back to track")     # clockwise
                    angle = - 90
                else:
                    right_markers = right_markers[:min(len(right_markers), len(left_markers))]
                    left_markers = left_markers[:min(len(right_markers), len(left_markers))]
            
                    marker_list = np.concatenate([right_markers, left_markers])
                    angle = self.calcAngle(marker_list, vehicle)
                
                length = .15
                if angle == 0:
                    print("No Markers infront - Drive forward")
                    vehicle.drive_straight(length, speed=20).start(thread=False)
                elif angle > 10 or angle < -10:
                    print(f"turn with {-angle} degrees")
                    vehicle.drive_turn(-angle, 0.0).start(thread=False)
                    print("Drive forward")
                    vehicle.drive_straight(length, speed=20).start(thread=False)
                else:
                    print(f"turn with 360 + {-angle} degrees")
                    vehicle.drive_turn(360 + (-angle), 0.0).start(thread=False)
                    print("Drive forward")
                    vehicle.drive_straight(length, speed=20).start(thread=False)

                time.sleep(4)
                    
            except Exception as e:
                print("Already-Running")
                print(e)
                break

        return None

    # def run1(self):
    #     print("Run started")
    #     try:
    #         vehicle = ev3.TwoWheelVehicle (
    #             self.radius,  # radius_wheel
    #             self.tread,  # tread
    #             protocol=ev3.USB
    #         ) 
    #         print("Vehicle initialized")
    #         current_ids = []
    #         pool_slam=None
    #         pool_explore=None
    #         while True:
    #             if pool_slam is None:
    #                 pool_slam = ThreadPool(processes=1)
    #                 result = pool_slam.apply_async(self.doSLAM, (vehicle,))
    #                 current_ids = result.get()
    #                 pool_slam.close()
    #                 pool_slam.join()
    #                 pool_slam=None
    #             if pool_explore is None:
    #                 pool_explore = ThreadPool(processes=1)
    #                 pool_explore.apply_async(self.explore, (vehicle,)) 
    #                 pool_explore.close()
    #                 pool_explore.join()
    #                 pool_explore=None
          
    #     except Exception as e:
    #         print(e)


    def run(self):
        print("Run started!")
        try:
            vehicle = ev3.TwoWheelVehicle (
                self.radius,  # radius_wheel
                self.tread,  # tread
                protocol=ev3.USB
            ) 
            print("Vehicle initialized")
            t1_slam = time.time()
            t2_slam = time.time()
            t1_ex = time.time()
            t2_ex = time.time()
            current_ids = []
            try:
                pool = ThreadPool(processes=2)
                print("########################")   
                pool.apply_async(self.explore, (vehicle,))
                t1_slam = time.time()
                result = pool.apply_async(self.doSLAM, (vehicle,))
                current_ids = result.get()
                pool.close()
                pool.join()
                self.saveMap()
            except Exception as e:
                print(e) 
                

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

if __name__ == '__main__':
    main = Main()
