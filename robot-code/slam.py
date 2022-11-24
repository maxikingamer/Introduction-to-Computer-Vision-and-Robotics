import numpy as np 

# define constants:
measurement_distance_stddev = 0.02
robot_width = 0.15
measurement_angle_stddev = 0.001

# your arbitrary start position
initial_state = (420, 69)
initial_covariance = np.zeros((3,3))
def get_motor_movement():

    # returns the motor movement in this timestep
    # return (l, r)
    pass

def camera_detections(robot_pose):
    # get landmarks from camera image and return a list
    # where each item is a landmark observation:
    # [[actual_measurement, world_coords, camera_coords, id], ...]
    pass

class EKFSLAM:
    def __init__(self, ...):
        pass

    def predict(self, ...):
        pass

    def add_landmark(self, ...):
        pass

    def correction(self, ...):
        pass

    def get_robot_pose(self):
        # read out the robot position and angle from mu variable
        # read out robot error from Sigma
        pass

    def get_landmark_positions(self):
        # read out the landmark positions from mu variable
        # read out landmark error from Sigma
        pass





# # slam = EKFSLAM(...)

# for timestep:
# # movements is what is refered to as u = (l, r) in the document
# movements = get_motor_movement()
# slam.predict(movements)
# landmark_observations = camera_detections(slam.get_robot_pose())
# for landmark_observation in landmark_observations:
# actual_measurement, world_coords, camera_coords, id = landmark_observation

# # actual_measurement is the distance r and angle alpha of the landmark relative to the robot.
# # world_coords are the world coordinates of the landmark.
# # camera_coords are the camera coordinates of the landmark.
# # id is a unique number identifying the landmark.
# if id_never_seen_before(id):
# slam.add_landmark(world_coords)

# slam.correction(actual_measurement, id)


# print("landmark estimated positions:", slam.get_landmark_positions())
