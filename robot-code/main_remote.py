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
from imageProcessing import automatic_brightness_and_contrast, preprocess2, detectRedCircle, detectGreenCircle


class Main():
    def __init__(self, stdscr) -> None:
        # keypress listener will break the terminal if we don't close it on exception
        sys.excepthook = self.except_hook

        self.keypress_listener = KeypressListener()
        self.publisher = Publisher()
        self.camera = Camera()
        self.camera_height = 0.285
        self.count = 0
        self.slam = EKFSLAM(0.05,0.05,1,60*np.pi/180)
        self.old_pos = [0,0,0]
        self.input = stdscr
        self.speed = 0
        self.turn = 0
        self.tread = 0.1392
        self.radius = 0.0280 
        self.seen_ids = {}
        self.translation_matrix = [[ 1.00000001e+00, 6.76287665e-10], [ 1.24710522e+00, -1.32404765e-01]]

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
            # img = automatic_brightness_and_contrast(img)
            # disc_img = np.copy(img[int(img.shape[0]/2):,:,:])

            # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # gray_blurred = cv2.GaussianBlur(gray, (3,3), sigmaX=9, sigmaY=9, borderType = cv2.BORDER_DEFAULT)
            # thresh = cv2.threshold(disc_img, 120, 255, cv2.THRESH_BINARY)[1] 
            
            # Get keypoints
            # keypoints = self.detector.detect(thresh)
            keypoints_red, thresh_red = detectRedCircle(img)
            keypoints_green, thresh_green = detectGreenCircle(img)
            self.input.addstr(10, 0, f"Green: {len(keypoints_green)}")
            self.input.addstr(11, 0, f"Red: {len(keypoints_red)}")

            keypoints = keypoints_green + keypoints_red
            # detect green
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            # store the a-channel
            a_channel = lab[:,:,1]
            # Automate threshold using Otsu method
            th = cv2.threshold(a_channel,0.5,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

            # detect aruco markers
            corners, ids, rejected_img_points = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
            # marker_box = np.zeros(1)
            # disc_box = np.zeros(1)
            if ids is not None:
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
                    cv2.rectangle(img, (int(corners[j][0][0][0]), int(corners[j][0][0][1])), (int(corners[j][0][2][0]), int(corners[j][0][2][1])), color = (0, 255, 0), thickness=2)
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

                    positions.append([pos_x + robot_position[0], pos_y + robot_position[1]])
                    cv2.aruco.drawDetectedMarkers(img, corners)
                    cv2.drawFrameAxes(img, self.camera.camera_matrix, self.camera.dist_coeffs, rvec, tvec, 0.048)

                    accepted_ids.append(ids[j])
                
            else:
                ids = []

            discs = []

            for i, keypoint in enumerate(keypoints):
                x = int(keypoint.pt[0])
                y = int(keypoint.pt[1])
                s = keypoint.size
                r = int(math.floor(s/2))

                if th[y-r:y+r,x-r:x+r].sum() > (10*r**2*math.pi)/11:
                    cv2.circle(img,(x, y), r, color=(255,255,255)) # green circle -> white
                else:
                    cv2.circle(img,(x, y), r, color=(0,0,0)) # red circle -> black

                box = np.array([[[x-r,y-r],[x-r,y+r],[x+r,y-r],[x+r,y+r]]], dtype="double")
                disc_box = self.translation_matrix @ np.array([[x-r,y-r], [x+r,y+r]])

                cv2.rectangle(img, (int(disc_box[0][0]), int(disc_box[0][1])), (int(disc_box[1][0]), int(disc_box[1][1])), color = (0, 0, 255), thickness=2)
                
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(box, 0.05, self.camera.camera_matrix, self.camera.dist_coeffs)
                hypothenuse = np.linalg.norm(tvec)
                distance = np.sqrt(np.abs(hypothenuse**2 - self.camera_height**2))
                pos_x = tvec[0][0][0]
                pos_y = np.sqrt(np.abs(distance**2 - pos_x**2))

                direction_global_coords = rotation_matrix @ np.array([pos_x,pos_y]) 
                pos_x = direction_global_coords[0]
                pos_y = direction_global_coords[1]

                position = [pos_x + robot_position[0], pos_y + robot_position[1]]
                angle = (np.arctan2(pos_y,pos_x))
                discs.append([position, distance, angle])

            # if marker_box.shape[0] > 1:
            #     if disc_box.shape[0] > 1:
            #         translation_matrix = marker_box @ np.linalg.inv(disc_box)
            #         print(translation_matrix)
            #         time.sleep(60)

        else:
            ids = []

        return img, accepted_ids, positions, distances, angles, discs

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
            self.input.addstr(6, 0, f"EV3 Position: {ev3_pos}") # fancy print

            direction = np.arctan2(new_pos[1]-self.old_pos[1],new_pos[0]-self.old_pos[0])

            # get image from the camera
            _, raw_img = self.camera.read()
            
            # get the time
            time0 = time.time()

            # imaginary processing
            img, ids, positions, distances, angles, discs = self.transformImage(raw_img, new_pos)

            #########SLAM#######
            uncertainty = [0.6,0.6]

            self.slam.predict(vehicle, self.tread)

            self.old_pos = new_pos
            
            
            if len(ids) > 0:
                for i, id in enumerate(ids):
                    if self.slam.id_never_seen_before(id[0]):
                        self.slam.add_landmark(id[0],positions[i],uncertainty)
                    if id == 11:
                        self.input.addstr(7, 0, f"Landmark positions: {positions[i]}")
                    #self.slam.correction_pro(id[0], positions[i])
                    measured_r, measured_alpha, est_r, est_alpha = self.slam.correction(id[0],positions[i])
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
            # landmark_estimated_positions = [[landmark_estimated_positions[i][1],landmark_estimated_positions[i][0]] for i in range(len(landmark_estimated_positions))]
            landmark_estimated_stdevs = landmark_estimated_stdevs.diagonal().reshape(int(len(landmark_estimated_ids)),2)
            # landmark_estimated_stdevs = [[landmark_estimated_stdevs[i][1],landmark_estimated_stdevs[i][0]] for i in range(len(landmark_estimated_stdevs))]
            # create message
            distances.extend([disc[1] for disc in discs])
            angles.extend([disc[2] for disc in discs])
            angles = [angle - math.pi/2 for angle in angles]

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
                landmark_positions = positions,

                landmark_estimated_ids = landmark_estimated_ids,
                landmark_estimated_positions = landmark_estimated_positions,
                landmark_estimated_stdevs = landmark_estimated_stdevs,

                robot_position = np.array([slam_position[1], -slam_position[0]]),
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
            object_ids = []
            for key in np.array(list(self.slam.map.keys())):
                object_ids.append(int(key % 3))
            landmarks = np.append(landmarks, np.array(object_ids).reshape(int(landmarks.shape[0]),1), axis=1)
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