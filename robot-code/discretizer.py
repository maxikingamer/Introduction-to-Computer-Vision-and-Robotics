import numpy as np 
# 1,2,3 = Markers / 4 = padding / 5 = border / 6 = finish / red_discs = 7 / 8 = start / green_disk = 9 / padding_red_disk = 10 / 0 = background
class Discretizer:
    """
    Class to perform discretization and map generation
    """
    def __init__(self, landmarks, grid_size=1, world_coords=[400,500]):
        self.grid_size = grid_size
        self.world_map = np.zeros((int(world_coords[0]/grid_size),int(world_coords[1]/grid_size)))
        self.mask = np.zeros((int(world_coords[0]/grid_size)+2,int(world_coords[1]/grid_size)+2))
        self.middle_point = [int(world_coords[0]/2), int(world_coords[1]/2), 0]
        self.landmarks = landmarks
        
    def naive_line(self, r0, c0, r1, c1):
        """
        Computes the points lying on the line between two points.
        """
        # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
        # If either of these cases are violated, do some switches.
        if abs(c1-c0) < abs(r1-r0):
            # Switch x and y, and switch again when returning.
            xx, yy = self.naive_line(c0, r0, c1, r1)
            return (yy, xx)

        # At this point we know that the distance in columns (x) is greater
        # than that in rows (y). Possibly one more switch if c0 > c1.
        if c0 > c1:
            return self.naive_line(r1, c1, r0, c0)

        # We write y as a function of x, because the slope is always <= 1
        # (in absolute value)
        x = np.arange(c0, c1+1, dtype=float)
        y = x * (r1-r0) / (c1-c0) + (c1*r0-c0*r1) / (c1-c0)

        return (np.concatenate((np.floor(y), np.floor(y)+1)).astype(int), np.concatenate((x,x)).astype(int))

    def createMap(self, surrounding):
        """
        Hub that calls all necessary functions to generate the map.
        """
        surrounding = int(surrounding/self.grid_size) 
        for landmark in self.landmarks:
            landmark += self.middle_point
            if landmark[2] > 2:
                self.world_map[int(landmark[0]/self.grid_size), int(landmark[1]/self.grid_size)] = landmark[2]
            elif landmark[2] == -1:
                self.mask[int(landmark[0]/self.grid_size), int(landmark[1]/self.grid_size)] = 8
            else:
                self.world_map[int(landmark[0]/self.grid_size)-2:int(landmark[0]/self.grid_size)+2, int(landmark[1]/self.grid_size)-2:int(landmark[1]/self.grid_size)+2] = landmark[2] + 1
        right_walls = np.array(self.landmarks[self.landmarks[:,2] == 0][:,:2] / self.grid_size).astype(int)
        obstacles = np.array(self.landmarks[self.landmarks[:,2] == 1][:,:2] / self.grid_size).astype(int)
        left_walls = np.array(self.landmarks[self.landmarks[:,2] == 2][:,:2] / self.grid_size).astype(int)
        starting_line = np.array(self.landmarks[self.landmarks[:,2] == 6][:,:2] / self.grid_size).astype(int)

        self.drawBorders(right_walls, threshold=int(25/self.grid_size))
        self.drawBorders(obstacles, wall=False, threshold=int(25/self.grid_size))
        self.drawBorders(left_walls, threshold=int(25/self.grid_size))
        self.drawBorders(starting_line, threshold=int(25/self.grid_size))
        self.pad_path(surrounding=surrounding)
    
    def checkNeighbours(self, x, y, map, surrounding):
        """
        Creates the list of neighbours of a given point /old
        """
        neighbours = [[x,y]]
        for i in range(1, surrounding+1):
            new_neighbours = []
            for (x,y) in neighbours:
                moves = [[0,i], [0,-i], [i, 0], [-i,0], [i,i], [i,-i], [-i,i], [-i,-i]]
                tmp = [[x,y]]*len(moves)
                possible_neighbours = [[x1+x2, y1+y2] for ((x1,y1),(x2,y2)) in zip(tmp, moves)]
            
                for (x_n, y_n) in possible_neighbours:
                    if ((x_n>0) & (x_n<=map.shape[0]) & (y_n>0) & (y_n<=map.shape[0])):
                        if [x_n,y_n] not in neighbours:
                            new_neighbours.append([x_n, y_n])
                
            neighbours.extend(new_neighbours)

        return neighbours

    def add_red_disc(self, pos, surrounding):
        """
        Add red discs to the map.
        """
        surrounding = int(surrounding/self.grid_size)
        pos = ((pos*100 + np.array(self.middle_point[:2]))/self.grid_size).astype(int)
        if pos[0] < self.world_map.shape[0] and pos[1] < self.world_map.shape[1]:
            self.world_map[pos[0], pos[1]] = 7
            y,x = pos
            overlay = self.world_map[y-surrounding:y+surrounding,x-surrounding:x+surrounding] == 0
            (self.world_map[y-surrounding:y+surrounding,x-surrounding:x+surrounding])[overlay] = 10
            return True
        return False

            
    def pad_path(self, surrounding):
        """
        Add padding to the map.
        """
        for y, row in enumerate(self.world_map):
            for x, pixel in enumerate(row):
                if (pixel > 0) & (pixel < 5):
                    if ((pixel == 3) | (pixel == 2)): surrounding += 1
                    elif pixel == 1: surrounding -= 3
                    elif pixel == 6: surrounding = 0
                    overlay = self.world_map[y-surrounding:y+surrounding,x-surrounding:x+surrounding] == 0
                    (self.world_map[y-surrounding:y+surrounding,x-surrounding:x+surrounding])[overlay] = 5
                    if ((pixel == 3) | (pixel == 2)): surrounding -= 1
                    elif pixel == 1: surrounding += 3
                    elif pixel == 6: surrounding = 0
    

    def drawBorders(self, landmarks, threshold=25, wall=True): 
        """
        Draw the lines collecting the landmarks on the map.
        """
        if(len(landmarks) < 2):
            return
        
        for l1 in landmarks:
            min_dist1 = np.inf
            min_dist2 = np.inf
            neighbours = [0,0]

            for l2 in landmarks:  
                if np.equal(l1,l2).all(): continue 
                dist = np.linalg.norm(l1-l2)

                if dist < min_dist1:
                    min_dist1 = dist
                    neighbours[0] = l2
                elif dist < min_dist2:
                    min_dist2 = dist
                    neighbours[1] = l2

            if wall:
                if min_dist1 > 2*threshold:
                    continue
                elif min_dist2 > 2*threshold:
                    neighbours[1] = 0
            else:
                if min_dist1 > threshold:
                    continue
                elif min_dist2 > threshold:
                    neighbours[1] = 0

            for i in range(2):
                if isinstance(neighbours[i], int): break
                line = self.naive_line(l1[0], l1[1], neighbours[i][0] , neighbours[i][1])
                for border_x, border_y in zip(line[0], line[1]):
                    if border_x < self.world_map.shape[0] and border_y < self.world_map.shape[1] and border_x > 0 and border_y > 0:
                        self.world_map[border_x, border_y] = 4
                    else:
                        print("Measurement does not fit map criteria.")

                
        
                