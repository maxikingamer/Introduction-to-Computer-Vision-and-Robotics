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

class Main():
    def __init__(self, stdscr) -> None:
        sys.excepthook = self.except_hook
        self.input = stdscr
        self.keypress_listener = KeypressListener()
        self.publisher = Publisher()
        self.camera = Camera()
        self.camera_height = 0.248
        self.count = 0
        self.slam = EKFSLAM(0.6,0.6,1,60*np.pi/180)
        # self.tread = 0.12832   # radius - Floor
        # self.radius = 0.02968  # tread - Floor 
        self.radius = 0.0224  # radius - Carpet
        self.tread = 0.1116   # tread - Carpet

        self.seen_ids = {}
        self.markerPositions_3d = []                                                                                                                                                                                                                                                                                                                                                                                                                            
        self.markerPositions_2d = []
        self.exploredMarkers = []
        self.tvec_sol = np.loadtxt(r'./tvec_sol.csv',delimiter=",",dtype=float)
        self.rvec_sol = np.loadtxt(r'./rvec_sol.csv',delimiter=",",dtype=float)
        self.green_positions = []
        self.red_positions = []


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

                if keypoint in keypoints_green:
                    cv2.circle(img,(x, y), r, color=(255,255,255)) # green circle -> white
                    if not np.any([np.allclose(position, old_markers,atol=0.03,rtol=0) for old_markers in self.green_positions]):
                        self.green_positions.append(position)

                else:
                    cv2.circle(img,(x, y), r, color=(0,0,0)) # red circle -> black
                    if not np.any([np.allclose(position, old_markers,atol=0.03,rtol=0) for old_markers in self.red_positions]):
                        self.red_positions.append(position)

                distance = 0  
                angle = 0    
                discs.append([position, distance, angle])

        else:
            ids = []
        self.markerPositions_2d = []
        self.markerPositions_3d = []

        return img, accepted_ids, positions, distances, angles, discs

    def doVisualizer(self, img, ids, positions, distances, angles, discs):
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
                self.input.addstr(0, 0, f'Robot Data.\t\t\t')
                self.input.addstr(1, 0, f'Position: [{round(x,2)}, {round(y,2)}]\t\t\t')
                self.input.addstr(2, 0, f'Angle: {round(np.degrees(angle),2)}\t\t\t')
                self.input.refresh()
                self.doVisualizer(img, ids, positions, distances, angles, discs)
                #self.saveMap()
            except Exception as e:
                self.input.addstr(20, 0, f'SLAM Exception: {e}\t\t\t')

    def saveMap(self):
        landmarks = np.array(self.slam.mu[3:]) * 100
        landmarks = landmarks.reshape(int(landmarks.shape[0]/2), 2)
        object_ids = []
        for key in np.array(list(self.slam.map.keys())):
            if key < 100:
                object_ids.append(6)
            else:
                object_ids.append(int(key % 3))
        landmarks = np.append(landmarks, np.array(object_ids).reshape(int(landmarks.shape[0]),1), axis=1)
        
        np.savetxt("landmarks.csv", landmarks, delimiter=",")
        return landmarks

    def filterPositions(self, marker_list):
        x_bot, y_bot, theta_bot = self.slam.get_robot_pose()[0]
        dx = marker_list[:,1] - x_bot
        dy = marker_list[:,2] - y_bot

        angles = np.degrees(((np.arctan2(dy,dx) % (2*np.pi))- theta_bot) % (2*np.pi))
        angles = np.where(angles > 180, angles - 360, angles)
        angles = angles.reshape(angles.shape[0], 1)
        marker_list = np.hstack([marker_list, angles])

        marker_list = marker_list[((marker_list[:,4] < 90) & (marker_list[:,4] > -90))]

        return marker_list


    def calcAngle(self,angles):
        if len(angles) == 0: angle = 0
        else: angle = np.sum(angles)/2
        return angle

    def explore(self, vehicle):
        counter = 0
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
                    break
                markers, _ = self.slam.get_landmark_positions()
                if isinstance(markers,list): markers = np.array(markers)
                markers = markers.reshape(int(markers.shape[0]/2), 2)
                ids = np.array(list(self.slam.map.keys()))
                ids = ids.reshape(ids.shape[0], 1)
       
                marker_list = np.hstack([ids, markers])

                left_markers = marker_list[np.where(marker_list[:,0] % 3 == 0)[0],:]
                right_markers = marker_list[np.where(marker_list[:,0] % 3  == 2)[0],:]

                pos, _ = self.slam.get_robot_pose()
                distances_left = np.sqrt((pos[0] - left_markers[:,1])**2 + (pos[1] - left_markers[:,2])**2)
                distances_right = np.sqrt((pos[0] - right_markers[:,1])**2 + (pos[1] - right_markers[:,2])**2)

                distances_left = distances_left.reshape(distances_left.shape[0], 1)
                left_markers = np.hstack([left_markers, distances_left])
                distances_right = distances_right.reshape(distances_right.shape[0], 1)
                right_markers = np.hstack([right_markers, distances_right])
                
                right_markers = right_markers[right_markers[:, 2].argsort()]
                left_markers = left_markers[left_markers[:, 2].argsort()]

                markers_infront_left = self.filterPositions(left_markers)
                markers_infront_right = self.filterPositions(right_markers)

                markers_infront_right = markers_infront_right[markers_infront_right[:,3]<.5]
                markers_infront_left = markers_infront_left[markers_infront_left[:,3]<.5]

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
                    angle = self.calcAngle(marker_list[:,4])
                
                length = .15
                speed = 10
                if angle == 0:
                    self.input.addstr(6, 0, f'Turn by {angle} degrees.\t\t\t')
                    self.input.addstr(7, 0, f'Drive by {length * 100}cm.\t\t\t')
                    self.input.refresh()
                    vehicle.drive_straight(length, speed=speed).start(thread=False)
                elif angle > 15 or angle < -15:
                    self.input.addstr(6, 0, f'Turn by {angle} degrees.\t\t\t')
                    self.input.addstr(7, 0, f'Drive by {length * 100}cm.\t\t\t')
                    self.input.refresh()
                    vehicle.drive_turn(angle, 0.0).start(thread=False)
                    vehicle.drive_straight(length, speed=speed).start(thread=False)
                else:
                    self.input.addstr(6, 0, f'Turn by {angle} degrees. Too small\t\t\t')
                    self.input.addstr(7, 0, f'Drive by {length * 100}cm.\t\t\t')
                    self.input.refresh()
                    vehicle.drive_straight(length, speed=speed).start(thread=False)
                if counter % 10 == 0: 
                    self.input.addstr(6, 0, f'Turn by 360 degrees.\t\t\t')
                    self.input.refresh()
                    vehicle.drive_turn(360,0.0).start(thread=False)
                counter += 1
                    
            except Exception as e:
                print(e)
                self.input.addstr(10, 0, f'Exploration Exception: {e}\t\t\t')
                self.input.refresh()
                break
        return self.saveMap()    
        

    def solve_task(self, vehicle):  
        #landmarks = self.explore(vehicle)

        self.input.addstr(4, 0, f'Done with exploration phase.\t\t\t')
        self.input.refresh()    
        landmarks = np.loadtxt("landmarks.csv", delimiter=",")
        self.input.addstr(4, 0, f'Drive to new starting position.\t\t\t')
        self.input.refresh()
        self.drive_track(vehicle,landmarks)

    def plan_line(self, start, end):
        distance = (np.sqrt((start[0] - end[0])**2 + (start[1]-end[1])**2)) 
        angle = np.arctan2(end[1]-start[1], end[0]-start[0])
        hit_obstacle = False
        if angle == 0: angle=0.01
        if distance == 0: distance = 0.01
        return distance, angle, hit_obstacle

    def drive_to_start(self, vehicle):
        pos_x, pos_y , theta = self.slam.get_robot_pose()[0]
        vehicle.drive_turn(np.degrees(2*np.pi - theta), 0.0).start(thread=False)
        pos_x, pos_y , theta = self.slam.get_robot_pose()[0]
        distance, angle, hit_obstacle = self.plan_line([pos_x, pos_y], [0,0])
        vehicle.drive_turn(np.degrees(angle), 0.0).start(thread=False)
        vehicle.drive_straight(distance, speed=10).start(thread=False)
        pos_x, pos_y , theta = self.slam.get_robot_pose()[0]
        vehicle.drive_turn(np.degrees(2*np.pi - theta), 0.0).start(thread=False)

    def drive_track(self,vehicle,landmarks):
        self.input.addstr(4, 0, f'Phase: Planning \t\t\t')
        self.input.refresh()
        grid_size=2
        size_x = int((max(np.abs(landmarks[:,0])))+10)
        size_y = int((max(np.abs(landmarks[:,1])))+10)
        world_coords=[max(size_x, size_y)*2, max(size_x, size_y)*2]
        self.input.addstr(5, 0, f'Discritize Map.\t\t\t')
        self.input.refresh()
        discretizer = Discretizer(landmarks, grid_size=grid_size, world_coords=world_coords)
        discretizer.createMap(surrounding=8)

        self.input.addstr(5, 0, f'Plan Path.\t\t\t')
        self.input.refresh()    
        
        #self.drive_to_start(vehicle)

        middle_point = np.array([int(world_coords[0]/2), int(world_coords[1]/2)])
        current_pos = list(np.array([1,1]) + ((self.slam.get_robot_pose()[0][:2] + middle_point)/grid_size).astype(int))
        end = [current_pos[0], current_pos[1]-6]

        separating_middle = [current_pos[0]-3, current_pos[1]]
        separaing_line = []

        i = 0
        while True:
            new_point = [separating_middle[0], separating_middle[1]-i]
            if discretizer.world_map[new_point[0],new_point[1]] == 5:
                break

            separaing_line.append(new_point)
            discretizer.world_map[new_point[0],new_point[1]] = 4
            i +=1

        i = 0
        while True:
            new_point = [separating_middle[0], separating_middle[1]+i]
            if discretizer.world_map[new_point[0],new_point[1]] == 5:
                break
            
            separaing_line.append(new_point)
            discretizer.world_map[new_point[0],new_point[1]] = 4
            i +=1



        a_star = A_Star(current_pos, end, discretizer.world_map.transpose(1,0))
        path = a_star.find_path()
        trajectory = a_star.plan_trajectory(5)


        cmap = colors.ListedColormap(['white', 'green', 'red', 'blue', 'black', 'orange', 'brown', 'pink'])
        plt.figure('Shortest path')
        plt.imshow(a_star.world_map + discretizer.mask.transpose(1,0), origin="lower", cmap=cmap, vmin=0, vmax=7)
        plt.plot(path[1],path[0], color="black", lw=2)
        plt.plot(a_star.start[1]+1,a_star.start[0]+1,'g.',markersize=20)
        plt.plot(a_star.goal[1]+1,a_star.goal[0]+1,'r.',markersize=20)
        plt.savefig('initial_map.png')

        self.input.addstr(4, 0, f'Phase: Driving\t\t\t')
        self.input.refresh()

        speed = 10
        trajectory = self.fix_trajectory(trajectory,grid_size)
   
        
        len_red_discs = 0
        len_green_discs = 0
        while True:

            if len_red_discs != len(self.red_positions):
                print(len_red_discs, len(self.red_positions))
                print(np.array(self.red_positions[len_red_discs]))
                discretizer.add_red_disc(np.array(self.red_positions[len_red_discs]), surrounding=8)
                print("test")
                len_red_discs += 1
                plt.figure('Shortest path')
                plt.imshow( discretizer.world_map.transpose(1,0), origin="lower", cmap=cmap, vmin=0, vmax=7)
                plt.savefig('add_red_map.png')
                #current_pos = list(np.array([1,1]) + ((self.slam.get_robot_pose()[0][:2] + middle_point)/grid_size).astype(int))
                #a_star = A_Star(current_pos, end, discretizer.world_map.transpose(1,0))
                #path = a_star.find_path()
                #trajectory = a_star.plan_trajectory(5)
                #trajectory = self.fix_trajectory(trajectory,grid_size)


            # if len_green_discs != len(self.green_positions):
            move = trajectory.pop(0)
            if np.degrees(move[3]) > 1 or np.degrees(move[3]) < -1 :
                self.input.addstr(6, 0, f'Turn by {-np.degrees(move[3])} degrees.\t\t\t')
                self.input.refresh()
                vehicle.drive_turn(-np.degrees(move[3]), 0.0).start(thread=False)
            if move[2]>0.01:
                self.input.addstr(7, 0, f'Drive by {move[2] * 100}cm.\t\t\t')
                self.input.refresh()
                vehicle.drive_straight(move[2], speed=speed).start(thread=False)

    def fix_trajectory(self,trajectory,grid_size):
        trajectory[0][3] = trajectory[0][3] - (np.pi/2)
        for i,move in enumerate(trajectory):
            trajectory[i][2] *= grid_size
            if move[3] < -np.pi:
                trajectory[i][3] += 2*np.pi
            
        return trajectory

    def run(self):
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
