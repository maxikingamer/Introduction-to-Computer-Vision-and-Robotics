import numpy as np 

class Discretizer:
    def __init__(self, landmarks, grid_size=1, world_coords=[400,500]):
        self.grid_size = grid_size
        self.world_map = np.zeros((int(world_coords[0]/grid_size),int(world_coords[1]/grid_size)))
        self.middle_point = [int(world_coords[0]/2), int(world_coords[1]/2), 0]
        self.landmarks = landmarks

    def naive_line(self, r0, c0, r1, c1):
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
        for landmark in self.landmarks:
            landmark += self.middle_point
            self.world_map[int(landmark[0])-2:int(landmark[0])+3, int(landmark[1])-2:int(landmark[1])+3] = landmark[2] + 10


        # sorted_landmarks = self.landmarks[self.landmarks[:, 2].argsort()]
        # precessor = sorted_landmarks[0]

        # for landmark in sorted_landmarks[1:]:
        #     line = self.naive_line(precessor[0], precessor[1], landmark[0], landmark[1])
        #     precessor = landmark
            
        #     for border_x, border_y in zip(line[0], line[1]):
        #         self.world_map[border_x, border_y] = 2

        # # line = self.naive_line(sorted_landmarks[0][0], sorted_landmarks[0][1], sorted_landmarks[-1][0], sorted_landmarks[-1][1])
        # # self.world_map[line[0], line[1]] = 2
        
        right_walls = self.landmarks[self.landmarks[:,2] == 0]
        obstacles = self.landmarks[self.landmarks[:,2] == 1]
        left_walls = self.landmarks[self.landmarks[:,2] == 2]

        self.drawBorders(right_walls)
        self.drawBorders(obstacles)
        self.drawBorders(left_walls)
        self.pad_path(surrounding=surrounding)

    
    def checkNeighbours(self, x, y, map, surrounding):
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

            
    def pad_path(self, surrounding):
        for y, row in enumerate(self.world_map):
            for x, pixel in enumerate(row):
                if pixel == 2 or pixel >= 10:
                    overlay = self.world_map[y-surrounding:y+surrounding,x-surrounding:x+surrounding] == 0
                    (self.world_map[y-surrounding:y+surrounding,x-surrounding:x+surrounding])[overlay] = 3
                    #neighbours = self.checkNeighbours(x,y,self.world_map,surrounding)
                    #for (x_n,y_n) in neighbours:
                        #if self.world_map[y_n][x_n] == 0:
                        #    self.world_map[y_n][x_n] = 3
    


    def drawBorders(self, landmarks, threshold=25): 
        
        if(len(landmarks) < 2):
            return
        

        for l1 in landmarks:
            min_dist1 = np.inf
            min_dist2 = np.inf
            neighbours = [0,0]

            for l2 in landmarks:  
                if np.equal(l1,l2).all(): continue 
                dist = np.linalg.norm(l1-l2)

                if dist < threshold:
                    if dist < min_dist1:
                        min_dist1 = dist
                        neighbours[0] = l2
                    elif dist < min_dist2:
                        min_dist2 = dist
                        neighbours[1] = l2


            for i in range(2):
                if isinstance(neighbours[i], int): break
                line = self.naive_line(l1[0], l1[1], neighbours[i][0] , neighbours[i][1])
                for border_x, border_y in zip(line[0], line[1]):
                    if border_x < self.world_map.shape[0] and border_y < self.world_map.shape[1] and border_x > 0 and border_y > 0:
                        self.world_map[border_x, border_y] = 2
                    else:
                        print("World Map is too small or measurement is hella wrong")

                
        
                