import numpy as np
import math
import ev3_dc as ev3

    
class EKFSLAM:
    """
    Class that performs slam.
    """
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
        """Read out the movements of the left and right wheels from the motor rotation."""

        motor_l_total, motor_r_total = vehicle.motor_pos
        motor_l = motor_l_total- self.motor_l_old
        motor_r = motor_r_total- self.motor_r_old
        self.motor_l_old = motor_l_total
        self.motor_r_old = motor_r_total
        l = motor_l * np.pi * 0.056 / 360
        r = motor_r * np.pi * 0.056 / 360
        return l, r

    def predict(self, vehicle, tread):
        """
        Compute the prediction step as provided in the pdf.
        """
        sigma = self.sigma[:3,:3]
        l, r = self.get_motor_movement(vehicle)
        w = tread
        theta = self.mu[2]
        alpha = (r-l)/w
        sigma_mu = np.array([[self.sigma_l**2,0],[0,self.sigma_r**2]])

        theta = (theta+alpha) % (2*np.pi)   # [0,2*pi]
        self.mu[2] = theta
        
        
        #create the matrixes G and V
        if l!=r:
            new_coords = np.array([self.mu[0],self.mu[1]]) + ((l/alpha) + (w/2)) * np.array([np.sin(theta+alpha) - np.sin(theta),-np.cos(theta+alpha)+np.cos(theta)])
            G = np.array([[1,0,((l/alpha)+(w/2))*(np.cos(theta+alpha)-np.cos(theta))],[0,1,((l/alpha)+(w/2))*(np.sin(theta+alpha)-np.sin(theta))],[0,0,1]])
            A = ((w*r)/(r-l)**2) * (np.sin(theta+alpha) - np.sin(theta)) - ((r+l)/2*(r-l))*np.cos(theta+alpha)
            B = ((w*r)/(r-l)**2) * (-np.cos(theta+alpha) + np.cos(theta)) - ((r+l)/2*(r-l))*np.sin(theta+alpha)
            C = -((w*r)/(r-l)**2) * (np.sin(theta+alpha) - np.sin(theta)) + ((r+l)/2*(r-l))*np.cos(theta+alpha)
            D = -((w*r)/(r-l)**2) * (-np.cos(theta+alpha) + np.cos(theta)) + ((r+l)/2*(r-l))*np.sin(theta+alpha)
            
        else:
            new_coords = np.array(self.mu[:2]) + l * np.array([np.cos(theta), np.sin(theta)])
            G = np.array([[1,0,-l*np.sin(theta)],[0,1,l*np.cos(theta)],[0,0,1]])
            A = 0.5*(np.cos(theta) + (l/w) * np.sin(theta))
            B = 0.5*(np.sin(theta) - (l/w) * np.cos(theta))
            C = 0.5*(np.cos(theta) - (l/w) * np.sin(theta))
            D = 0.5*(np.sin(theta) + (l/w) * np.cos(theta))
            
        V = np.array([[A,C],[B,D],[-1/w,1/w]])

        self.mu[0] = new_coords[0]
        self.mu[1] = new_coords[1]
        
        #sigma update
        self.sigma[:3,:3] = (G @ sigma @ G.T) + (V @ sigma_mu @ V.T)

        return theta, alpha

        
    def add_landmark(self,landmark_id,coordinates,uncertainty):
        """
        Adds a new landmark to the list of positions mu and the covariance matrix sigma.
        :param landmark_id: the id of the detected aruco marker
        :param coordinates: the [x,y]-coordinates of the landmark 
        :param uncertainty: the uncertainty of the measurment as an iterable with length 2
        """

        # extend mu
        self.mu = np.append(self.mu, coordinates,axis=0)
        # map the landmark id to its position in the coordinates vector
        self.map[landmark_id] = (len(self.mu)-3)/2
        
        # extend sigma
        sigma_new = np.zeros((self.sigma.shape[0]+2,self.sigma.shape[1]+2))
        sigma_new[:self.sigma.shape[0],:self.sigma.shape[1]] = self.sigma
        sigma_new[self.sigma.shape[0],self.sigma.shape[1]] = uncertainty[0]
        sigma_new[self.sigma.shape[0]+1,self.sigma.shape[1]+1] = uncertainty[1]
        self.sigma = sigma_new

    
    def difference_angle(self, angle1, angle2):
        difference = (angle1 - angle2) % (2*np.pi) 
        difference = np.where(difference > np.pi, difference - 2*np.pi, difference)
        return difference
   

    def correction_direct(self,landmark_id, landmark_dist_angle):
        """
        Computes the correction step as provided in the pdf.
        :param landmark_id: the id of the detected aruco marker
        :param landmark_coords: the [x,y]-coordinates of the aruco marker
        """
        robot_positions, robot_error = self.get_robot_pose()

        #measured distance and angle to landmark
        measured_r, measured_alpha = landmark_dist_angle
        

        landmark_id_in_mu = int(self.map[landmark_id])
        positions, errors = self.get_landmark_x_y(landmark_id_in_mu)
      
        #precomputation for derrivatives  
        x_bot, y_bot, theta_bot = robot_positions
        x_lm, y_lm = positions
        dx = x_lm - x_bot
        dy = y_lm - y_bot

        #estimated distance and angle to landmark
        est_r = np.sqrt(dx**2 + dy**2) 
        est_alpha = ((np.arctan2(dy,dx) % (2*np.pi))- theta_bot) % (2*np.pi)

        #compute matrix H
        H = np.zeros((2,len(self.mu)))
        H[0,0] = -dx/est_r
        H[0,1] = -dy/est_r
        H[0,2] = 0
        H[1,0] = dy/(est_r**2)
        H[1,1] = -dx/(est_r**2)
        H[1,2] = -1
        H[0,2*landmark_id_in_mu+1] = dx/est_r
        H[0,2*landmark_id_in_mu+2] = dy/est_r
        H[1,2*landmark_id_in_mu+1] = -dy/(est_r**2)
        H[1,2*landmark_id_in_mu+2] = dx/(est_r**2)

        #correction step
        Q = np.diag([self.merror_r**2,self.merror_alpha**2])
        K = self.sigma @ (H.T @ np.linalg.inv((H @ self.sigma @ H.T) + Q))
        assert -np.pi < self.difference_angle(measured_alpha,est_alpha) < np.pi

        # update mu
        self.mu += (K @ (np.array([measured_r - est_r, self.difference_angle(measured_alpha,est_alpha)])))

        #update sigma
        self.sigma = (np.identity(self.sigma.shape[0]) - (K @ H) ) @ self.sigma
        


    def get_robot_pose(self):
        """Read out the robot position and angle from mu variable"""
        return self.mu[:3], self.sigma[:3,:3]


    def get_landmark_positions(self):
        """
        Read out the landmark positions from mu variable, read out landmark error from Sigma.
        """
        return self.mu[3:], self.sigma[3:,3:]
    
    def get_landmark_x_y(self,i):
        """
        Get the landmark possitions and covariances from the mu variable and Sigma.
        """
        positions, errors = self.get_landmark_positions()
        return positions[2*i-2:2*i], errors[2*i-2:2*i,2*i-2:2*i]
        
        
    def id_never_seen_before(self, id):
        """
        Add landmarks that wehave not seen before to the dictionary of seen landmarks.
        """
        return not(id in list(self.map.keys()))




