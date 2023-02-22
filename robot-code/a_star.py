import numpy as np 
from scipy.spatial import distance
import math


class A_Star:
    def __init__(self, start, goal, origin, map):
        self.moves = np.array([(-1, 0), (0, 1), (1, 0), (0, -1), (-1,-1), (-1,1), (1,1), (1,-1)])
        # self.moves = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])
        self.start = start
        self.goal = goal
        self.world_map=np.ones((map.shape[0]+2,map.shape[1]+2))
        self.world_map[1:map.shape[0]+1,1:map.shape[0]+1]=map
        self.path = []
        self.d=np.full((self.world_map.shape[0],self.world_map.shape[1]), np.inf)
        self.d[self.start[0]+1,self.start[1]+1]=distance.euclidean(self.start, self.goal)
        # self.d[self.start[0]+1,self.start[1]+1]=distance.cityblock(self.start, self.goal)
        # self.q=np.array([self.world_map==0]).squeeze()
        self.q=np.array([((self.world_map==0) | (self.world_map==5)| (self.world_map==10))]).squeeze()
        self.q[self.start[0]+1,self.start[1]+1]=True
        self.cameFrom=np.zeros((2,self.world_map.shape[0], self.world_map.shape[1]),dtype='int')
        self.origin = origin

    def find_path(self):
        """
        Reconstructs the path given the distance matrix.
        """
        self.search_path()
        [px,py]=self.reconstruct_path()
        self.path = [px,py]
        return self.path

    def find_min_node(self):
        """
        Find the node with minimal cost from a given node.
        """
        tmp=np.array(self.d)
        # we only look at not visited nodes
        tmp[~self.q]=np.inf
        # find node wtih minimal cost
        u=np.unravel_index(tmp.argmin(), tmp.shape)
        # set this node as visited
        self.q[u[0],u[1]]=False
        return u

    # explore neighboring nodes
    def update_nodes(self,u):
        """Explore neighbouring nodes, compute cost"""
        for i in range(self.moves.shape[0]):
            # get a new node
            v=[u[0]+self.moves[i,0], u[1]+self.moves[i,1]]
            # if the new node is not an obstacle and it is not marked as visited
            if (v[0]>0) & (v[0]<self.world_map.shape[0]) & (v[1]>0) & (v[1]<self.world_map.shape[1]):
                # if not self.world_map[v[0],v[1]] and self.q[v[0],v[1]]:
                if (self.world_map[v[0],v[1]]==0 or self.world_map[v[0],v[1]]==5 or self.world_map[v[0],v[1]]==10) and self.q[v[0],v[1]]:
                    # compute distance from previuos node to the new node
                    dx=math.sqrt(pow(u[0]-v[0],2)+pow(u[1]-v[1],2))
                    # compute distance from the start node to the new node
                    g=self.d[u[0],u[1]]+dx
                    # compute distance from current node to end note by manhattan heuristic
                    # h=np.abs(v[0]-gx)+np.abs(v[1]-gy)
                    # h=distance.cityblock(v, self.goal)
                    h=distance.euclidean(v, self.goal)
                    if self.world_map[v[0],v[1]]==5: h += 999999999
                    if self.world_map[v[0],v[1]]==10: h += 500
                    f=g+h
                    # if new distance is shorter than previous distance the update
                    if f<self.d[v[0],v[1]]:
                        # update distance
                        self.d[v[0],v[1]]=f
                        # remember from which node came
                        self.cameFrom[:,v[0],v[1]]=[u[0],u[1]]

    # perform path search				
    def search_path(self):
        """
        Completes the cost matrix
        """
        # get number of not visited nodes
        k=np.count_nonzero(self.q)
        # search until all nodes are visited
        while k>0:
            # get new node with minimal cost
            u=self.find_min_node()
            # stop search if end point is reached
            if u == self.goal:
                break
            # explore neighboring nodes
            self.update_nodes(u)
            # reduce counter
            k=k-1

    # reconstruct path from visited nodes list
    def reconstruct_path(self):
        """
        Construcs the path from start to finish
        """
        T=pow(self.world_map.shape[0]-2,2)
        
        # we shift points since we have added borders
        sx=self.start[0]+1
        sy=self.start[1]+1
        gx=self.goal[0]
        gy=self.goal[1]
        
        px=[gx]
        py=[gy]

        for t in range(T):
            [x,y]=self.cameFrom[:,px[0],py[0]]
            px.insert(0,x)
            py.insert(0,y)
            if x==sx and y==sy:
                break
        return px, py

    def plan_line(self, start, end):
        """
        Computes angle and distance given two points.
        """
        distance = (np.sqrt((start[0] - end[0])**2 + (start[1]-end[1])**2)) / 100
        angle = np.arctan2(end[1]-start[1], end[0]-start[0])
        hit_obstacle = False
        if angle == 0: angle=0.01
        if distance == 0: distance = 0.01
        return distance, np.degrees(angle), hit_obstacle


    def plan_trajectory(self):
        """
        Plan the trajectory from start to finish based on the A-star path.
        """
        # num_ankers = int(len(self.path[0])/10)
        # ankers = np.linspace(0,len(self.path[0])-1, num_ankers, dtype="int64")

        ankers = self.smooth_path()
        trajectory = []
        last_angle = 0
        add_angle = 0
        add_length = 0
        passed_points = 1
        check = True

        for i, anker in enumerate(ankers):
            # if i == len(ankers)-1:
            #     end = [self.path[0][ankers[i]], self.path[1][ankers[i]]]
            #     trajectory.append([start, end, length, angle])
            #     break
            if i == len(ankers)-1:
                end = [self.origin[0], self.origin[1]]
            else:
                end = [self.path[0][ankers[i+1]], self.path[1][ankers[i+1]]]
            if check:
                start = [self.path[0][anker], self.path[1][anker]]
            
            
            length, angle, hit_obstacle = self.plan_line(start, end)
            turning_angle = angle - last_angle

            if (length < 0.05):
                check = False
            else:
                angle = turning_angle 
                last_angle += angle
                if angle > 180: angle = angle - 360
                elif angle < -180: angle = angle + 360
                trajectory.append([start, end, length, angle, last_angle])
                check = True

        return trajectory

    def smooth_path(self):
        """
        returns smoothed path which is faster than usual trajectory from above
        """
        anker_points = [0]
        vertical = (self.path[1][0] - self.path[1][1]) == 0
        diagonal = (np.abs(self.path[1][0] - self.path[1][1]) + np.abs(self.path[0][0] - self.path[0][1])) == 2
        for i, (x, y) in enumerate(zip(self.path[0], self.path[1])):
            if i > 0:
                last_place = [self.path[0][i-1], self.path[1][i-1]]
                place = [x,y]

                if vertical:
                    if (last_place[1] - place[1]) == 0:
                        continue
                    else:
                        anker_points.append(i-1)
                        vertical = False
                else:
                    if (last_place[0] - place[0]) == 0:
                        continue
                    else:
                        anker_points.append(i-1)
                        vertical = True

        if i-1 not in anker_points:
            anker_points.append(i-1)
        if i not in anker_points:
            anker_points.append(i)

        return anker_points
    