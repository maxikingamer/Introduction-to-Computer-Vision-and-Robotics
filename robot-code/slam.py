import numpy as np
import math
import ev3_dc as ev3

def camera_detections(robot_pose):
    # get landmarks from camera image and return a list
    # where each item is a landmark observation:
    # [[actual_measurement, world_coords, camera_coords, id], ...]
    pass
    
class EKFSLAM:
    def __init__(self,stddev_l,stddev_r,erorr_r,erorr_alpha):
        self.mu = np.zeros(3)
        self.sigma = np.array([[0,0,0],[0,0,0],[0,0,0]])
        self.sigma_l =  stddev_l
        self.sigma_r =  stddev_r
        self.merror_r = erorr_r
        self.merror_alpha = erorr_alpha
        self.map = {}
        self.motor_l_old = 0
        self.motor_r_old = 0

    def get_motor_movement(self, vehicle):
        # returns the motor movement in this timestep
        # return (l, r)
        # x, y, angle = vehicle.position
        motor_l_total, motor_r_total = vehicle.motor_pos
        motor_l = motor_l_total- self.motor_l_old
        motor_r = motor_r_total- self.motor_r_old
        self.motor_l_old = motor_l_total
        self.motor_r_old = motor_r_total
        l = motor_l * np.pi * 0.056 / 360
        r = motor_r * np.pi * 0.056 / 360
        #print(f'the current motor_l position is: {motor_l}°')    
        #print(f'the current motor_r position is: {motor_r}°') 
        #if movement_type == "straight":
            #return movement_args, movement_args
        #else:
            #if movement_args[0] < 0:
            #tread_r = movement_args[1] + tread/2
            #tread_l = movement_args[1] - tread/2
            #else:
            #    tread_r = movement_args[1] - tread/2
            #    tread_l = movement_args[1] + tread/2
    
            
        #return movement_args[0] * tread_l, movement_args[0] * tread_r
        return l, r

    def predict(self, vehicle, tread):
        """
        Compute the prediction step as provided in the pdf.
        INPUT PARAMETERS ARE TO BE DECIDED
        SINCE WE NEED ACCESS TO THE MOVEMENT 
        """
        sigma = self.sigma[:3,:3]
        l, r = self.get_motor_movement(vehicle)
        w = tread
        theta = self.mu[2]
        alpha = (r-l)/w
        sigma_mu = np.array([[self.sigma_l**2,0],[0,self.sigma_r**2]])
        
        #create the matrixes G and V
        if l!=r:
            G = np.array([[1,0,((l/alpha)+(w/2))*(np.cos(theta+alpha)-np.cos(theta))],[0,1,((l/alpha)+(w/2))*(np.sin(theta+alpha)-np.sin(theta))],[0,0,1]])
            A = ((w*r)/(r-l)**2) * (np.sin(theta+alpha) - np.sin(theta)) - ((r+l)/2*(r-l))*np.cos(theta+alpha)
            B = ((w*r)/(r-l)**2) * (-np.cos(theta+alpha) + np.cos(theta)) - ((r+l)/2*(r-l))*np.sin(theta+alpha)
            C = -((w*r)/(r-l)**2) * (np.sin(theta+alpha) - np.sin(theta)) + ((r+l)/2*(r-l))*np.cos(theta+alpha)
            D = -((w*r)/(r-l)**2) * (-np.cos(theta+alpha) + np.cos(theta)) + ((r+l)/2*(r-l))*np.sin(theta+alpha)
            new_coords = np.array([self.mu[0],self.mu[1]]) + ((l/alpha) + (w/2)) * np.array([np.sin(theta+alpha) - np.sin(theta),-np.cos(theta+alpha)+np.cos(theta)])

        else:
            G = np.array([[1,0,-l*np.sin(theta)],[0,1,l*np.cos(theta)],[0,0,1]])
            A = 0.5*(np.cos(theta) + (l/w) * np.sin(theta))
            B = 0.5*(np.sin(theta) - (l/w) * np.cos(theta))
            C = 0.5*(np.cos(theta) - (l/w) * np.sin(theta))
            D = 0.5*(np.sin(theta) + (l/w) * np.cos(theta))
            new_coords = np.array(self.mu[:2]) + l * np.array([np.cos(theta), np.sin(theta)])
        V = np.array([[A,C],[B,D],[-1/w,1/w]])

        
        #new coords formula
        angle = (theta+alpha) % (2*np.pi)
        # if angle > np.pi:
        #     angle = angle - 2 * np.pi
        self.mu[2] = angle
        self.mu[0] = new_coords[0]
        self.mu[1] = new_coords[1]

        #new coords direct
        #self.mu[0], self.mu[1], self.mu[2] = vehicle.position
        #self.mu[2] = np.radians(self.mu[2])

        # print("\n SLAM Coords: ", new_coords)
        # print("\n Vehicle Coords: ", self.mu[:3])
        
        #sigma update
        self.sigma[:3,:3] = (G @ sigma @ G.T) + (V @ sigma_mu @ V.T)

        
    def add_landmark(self,landmark_id,coordinates,uncertainty):
        """
        Adds a new landmark to the list of positions mu and the covariance matrix sigma.
        :param landmark_id: the id of the detected aruco marker
        :param coordinates: the [x,y]-coordinates of the landmark 
        :param uncertainty: the uncertainty of the measurment as an iterable with length 2
        """
        coordinates_fixed = [coordinates[1],coordinates[0]]
        # extend mu
        self.mu = np.append(self.mu, coordinates_fixed,axis=0)
        # map the landmark id to its position in the coordinates vector
        self.map[landmark_id] = (len(self.mu)-3)/2
        
        # extend sigma
        sigma_new = np.zeros((self.sigma.shape[0]+2,self.sigma.shape[1]+2))
        sigma_new[:self.sigma.shape[0],:self.sigma.shape[1]] = self.sigma
        sigma_new[self.sigma.shape[0],self.sigma.shape[1]] = uncertainty[0]
        sigma_new[self.sigma.shape[0]+1,self.sigma.shape[1]+1] = uncertainty[1]
        self.sigma = sigma_new

    def correction(self,landmark_id, landmark_coords):
        """
        Computes the correction step as provided in the pdf.
        :param landmark_id: the id of the detected aruco marker
        :param landmark_coords: the [x,y]-coordinates of the aruco marker
        """
        landmark_coords = [landmark_coords[1],landmark_coords[0]]
        landmark_id_in_mu = int(self.map[landmark_id])
        positions, errors = self.get_landmark_x_y(landmark_id_in_mu)
        robot_positions, robot_error = self.get_robot_pose()

        #estimated distance and angle to landmark
        est_r = np.sqrt((positions[0]-robot_positions[0])**2 + (positions[1]-robot_positions[1])**2)
        est_alpha = np.arctan2((positions[1]-robot_positions[1]),(positions[0]-robot_positions[0])) - robot_positions[2] 
        #print(est_alpha)
        #measured distance and angle to landmark
        measured_r = np.sqrt((landmark_coords[0]-robot_positions[0])**2 + (landmark_coords[1]-robot_positions[1])**2)
        measured_alpha = np.arctan2((landmark_coords[1]-robot_positions[1]),(landmark_coords[0]-robot_positions[0])) - robot_positions[2]
        #print(measured_alpha)
        #compute matrix H
        H = np.zeros((2,len(self.mu)))
        H[0,0] = -(landmark_coords[0]-robot_positions[0])/measured_r
        H[0,1] = -(landmark_coords[1]-robot_positions[1])/measured_r
        H[0,2] = 0
        H[1,0] = (landmark_coords[1]-robot_positions[1])/(measured_r**2)
        H[1,1] = -(landmark_coords[0]-robot_positions[0])/(measured_r**2)
        H[1,2] = -1
        H[0,2*landmark_id_in_mu+1] = (landmark_coords[0]-robot_positions[0])/measured_r
        H[0,2*landmark_id_in_mu+2] = (landmark_coords[1]-robot_positions[1])/measured_r
        H[1,2*landmark_id_in_mu+1] = -(landmark_coords[1]-robot_positions[1])/(measured_r**2)
        H[1,2*landmark_id_in_mu+2] = (landmark_coords[0]-robot_positions[0])/(measured_r**2)
        #correction step
        Q = np.diag((self.merror_r,self.merror_alpha))
        K = self.sigma @ (H.T @ np.linalg.inv( (H @ self.sigma @ H.T) + Q))
        assert -np.pi < measured_alpha - est_alpha < np.pi
        # update mu
        # formula changed but the result should be identical to the one provided in the pdf
        # print(f"{self.mu} = {self.mu} + ({K} * {np.array([measured_r, measured_alpha])} - {np.array([est_r, est_alpha])}")
 
        #print((K @ (np.array([measured_r, measured_alpha]) - np.array([est_r, est_alpha]))))
        self.mu = self.mu + (K @ (np.array([measured_r, measured_alpha]) - np.array([est_r, est_alpha])))
        quark = (K @ (np.array([measured_r, measured_alpha]) - np.array([est_r, est_alpha])))
        #update sigma
        self.sigma = (np.identity(self.sigma.shape[0]) - (K @ H) ) @ self.sigma
        
        return measured_r, measured_alpha, est_r, est_alpha, quark

    def get_robot_pose(self):
        # read out the robot position and angle from mu variable
        # read out robot error from Sigma
        
        return self.mu[:3], self.sigma[:3,:3]


    # the two following functions can be merged
    def get_landmark_positions(self):# read out the landmark positions from mu variable
        # read out landmark error from Sigma
        return self.mu[3:], self.sigma[3:,3:]
    
    def get_landmark_x_y(self,i):
        positions, errors = self.get_landmark_positions()
        return positions[2*i-2:2*i], errors[2*i-2:2*i,2*i-2:2*i]
        
        
    def id_never_seen_before(self, id):
        return not(id in list(self.map.keys()))




slam = EKFSLAM(0.01,0.01,0.01,0.01)
timesteps = []

for timestep in timesteps:
    # movements is what is refered to as u = (l, r) in the document
    movements = get_motor_movement()
    slam.predict(movements)
    landmark_observations = camera_detections(slam.get_robot_pose())
    for landmark_observation in landmark_observations:
        actual_measurement, world_coords, camera_coords, id = landmark_observation

        # actual_measurement is the distance r and angle alpha of the landmark relative to the robot.
        # world_coords are the world coordinates of the landmark.
        # camera_coords are the camera coordinates of the landmark.
        # id is a unique number identifying the landmark.
        if slam.id_never_seen_before(id):
            slam.add_landmark(world_coords)
        slam.correction(actual_measurement, id)
        # print("landmark estimated positions:", slam.get_landmark_positions())
