import heapq as hq
from typing import Tuple
import numpy as np
from key import Key

import rospy
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg

from occupancy_grid_2d import OccupancyGrid2d

from enum import Enum

class RecalculateEnum(Enum):
    NONE = 0
    GOAL = 1
    OFFSET = 2

class DStarLite:
    NEXT_FRAME_ID = "next_waypoint"
    ROBOT_FRAME_ID = "base_footprint"
    GLOBAL_FIXED_ID = "odom"

    def l2_norm(point1, point2):
        # print("Point 1", point1)
        # print("Point 2", point2)
        return np.linalg.norm(point1 - point2)

    # Note: points are of type two length array. 
    def __init__(self, occupancy_grid: OccupancyGrid2d, 
                goal, offset, finish_callback, 
                recalculate: RecalculateEnum):

        self.pub_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)

        # define global variables
        self.occupancy_grid = occupancy_grid
        occupancy_grid.add_d_star_lite_callback(self.callback)

        
        self.rhs = np.ones(occupancy_grid._2dmap.shape) * np.inf
        self.g = np.ones(occupancy_grid._2dmap.shape) * np.inf
        # print(self.occupancy_grid.get_current_location)
        self.start = self.occupancy_grid.get_current_location()
        self.last = self.start

        self.offset = offset
        self.goal = None
        self.original_goal = goal
        self.recalculate_mode = recalculate

        if recalculate == RecalculateEnum.GOAL:
            self.goal = self.occupancy_grid.object_center(goal) + offset
        else: 
            self.goal = goal + offset

        
        self.U = [] # priority queue to keep track of nodes to search
        self.costs = {}
        hq.heapify(self.U)
        self.km = 0

        # for vertex in np.indices(occupancy_grid.shape):
        #     self.rhs[vertex] = float('inf')
        #     self.g[vertex] = float('inf')

        self.goal = self.goal.astype(np.int)

        print(self.goal)
        
        self.rhs[self.goal[0], self.goal[1]] = 0
        hq.heappush(self.U, (Key(DStarLite.l2_norm(self.start, goal), 0), goal))

        self.finish_callback = finish_callback
        
    # v = vertex for which we are calculating the key
    def calculate_key(self, u):
        return Key(min(self.g[u[0], u[1]], self.rhs[u[0], u[1]]) + DStarLite.l2_norm(self.start, u) + self.km, 
                   min(self.g[u[0], u[1]], self.rhs[u[0], u[1]]))

    def compute_shortest_path(self):
        print("Computing Shortest Path")
        while (len(self.U) and self.U[0][0] < self.calculate_key(self.start)) or \
            self.rhs[self.start[0], self.start[1]] > self.g[self.start[0], self.start[1]]:
            k_old, u = hq.heappop(self.U) # pop from top of the heap, k_old is old key, u is node
            k_new = self.calculate_key(u) # recalculate the key

            ## If distance increases
            if k_old < k_new:
                for vertex in self.U:
                    if np.array_equal(vertex[1], u):
                        self.U.remove(vertex)
                hq.heappush(self.U, (k_new, u))

            ## If node is not "locally consistent"
            elif self.g[u[0], u[1]] > self.rhs[u[0], u[1]]:
                self.g[u[0], u[1]] = self.rhs[u[0], u[1]]
                for s in self.neighbors(u):
                    # print(self.neighbors(u))
                    if not np.array_equal(s, self.goal):
                        self.rhs[s[0], s[1]] = min(self.rhs[s[0], s[1]], 
                                                   self.occupancy_grid.get_edge_costs(s, u) + self.g[u[0], u[1]])
                    self.update_vertex(s) 
            
            ## This forces node to be essentially "overconsistent", which would trigger the elif above
            ## if the node is actually important. 
            else:
                g_old = self.g[u[0], u[1]]
                self.g[u[0], u[1]] = float('inf')
                for s in self.neighbors(u) + [u]:
                    if self.rhs[s[0], s[1]] == DStarLite.l2_norm(s, u) + g_old:
                        if s != self.goal:
                            self.rhs[s[0], s[1]] = min([self.occupancy_grid.get_edge_costs(s, s_prime) + self.g[s_prime[0], s_prime[1]] for s_prime in self.neighbors(s)])
                    self.update_vertex(s)

        path = self.get_shortest_path()

        # print(path)

        if len(path) > 5: # Can be tuned to be stupid
            self.publish_waypoint(path[5])
        else:
            self.teardown()

        print("finished compute shortest path")

    def update_vertex(self, u):
        nodes_in_pq = list(map(lambda x: x[1], self.U))
        u_in_nodes = any(map(lambda x: np.array_equal(x, u), nodes_in_pq))
        if self.g[u[0], u[1]] != self.rhs[u[0], u[1]] and u_in_nodes:
            for vertex in self.U:
                if np.array_equal(vertex[1], u):
                    self.U.remove(vertex)
            hq.heappush(self.U, (self.calculate_key(u), u))
            hq.heapify(self.U)
        elif self.g[u[0], u[1]] != self.rhs[u[0], u[1]] and not u_in_nodes:
            hq.heappush(self.U, (self.calculate_key(u), u))
        elif self.g[u[0], u[1]] == self.rhs[u[0], u[1]] and u_in_nodes:
            for vertex in self.U:
                if np.array_equal(vertex[1], u):
                    self.U.remove(vertex)
            hq.heapify(self.U)

    def neighbors(self, point):
        return self.occupancy_grid.get_edges_from_node(point)

    # ENTIRE LOGIC HERE MOVED INTO callback. Actual actuation of motors should be handled at a 
    # lower level. 
    # def main(self):
    #     while self.start != self.goal:
    #         successor = self.neighbors(self.start)
    #         self.start = np.argmin([DStarLite.l2_norm(self.start, s_prime) + self.g[s_prime] for s_prime in successor])
    #         self.start = successor[self.start] # TODO: makes no sense use numpy to fix
    #         # TODO: move robot to start
    #         # TODO: scan graph for changed edge costs
    #         costs_changed = {}
    #         if len(costs_changed.keys()) > 0:
    #             self.km = self.km + DStarLite.l2_norm(self.last, self.start)
    #             self.last = self.start
    #             for u, v in costs_changed:
    #                 c_old = DStarLite.l2_norm(u, v)
    #                 self.costs[(u, v)] = costs_changed[(u, v)]
    #                 if c_old > self.costs[(u, v)]:
    #                     if u != self.goal:
    #                         self.rhs[u] == min(self.rhs[u], self.costs[(u, v)] + self.g[v])
    #                 elif self.rhs[u] == c_old + self.g[v]:
    #                     if u != self.goal:
    #                         s_prime = u
    #                         self.rhs[u] == min(self.costs[(u, s_prime)] + self.g[s_prime])
    #                 self.update_vertex(u)
    #             self.compute_shortest_path()

    def recompute_goal(self):
        if self.recalculate_mode == RecalculateEnum.GOAL:
            self.goal = self.occupancy_grid.object_center(self.original_goal) + self.offset
        elif self.recalculate_mode == RecalculateEnum.OFFSET:
            offset_size = np.around(np.linalg.norm(self.offset))
            orientation = self.occupancy_grid.get_current_location() - self.original_goal
            orientation = np.sign(orientation) * (np.array([1, 0]) if np.abs(orientation[0]) > np.abs(orientation[1]) else np.array([0, 1]))
            self.offset = offset_size * orientation
            self.goal = self.original_goal + self.offset

    def callback(self):
        # keeping new entries of the priority queue fair by adding existing progress
        self.start = self.occupancy_grid.get_current_location()

        ## Update object center if we learn more information
        self.recompute_goal()

        if np.array_equal(self.start, self.goal):
            self.teardown()
        
        self.km = self.km + DStarLite.l2_norm(self.last, self.start) 
        self.last = self.start
        for u in range(self.occupancy_grid._changes.shape[0]):
            for v in range(self.occupancy_grid._changes.shape[1]):
                if self.occupancy_grid._changes[u, v]:
                    self.update_vertex(np.array([u, v]))
        self.compute_shortest_path()

    ## Called after updating all the g and rhs values
    def get_shortest_path(self):
        shortest_path = [self.start]
        while not np.array_equal(shortest_path[-1], self.goal):
            print(shortest_path)
            neighbors = self.occupancy_grid.get_edges_from_node(shortest_path[-1])
            neighbors = list(filter(lambda x: not any(map(lambda y: np.array_equal(x, y), shortest_path)), neighbors))
            max_index = np.argmin(list(map(lambda x: self.g[x[0], x[1]], neighbors)))
            shortest_path.append(neighbors[max_index])
        return shortest_path

    def publish_waypoint(self, point: np.ndarray):
        x, y = self.occupancy_grid.VoxelCenter2D(point[0], point[1])
        t = geometry_msgs.msg.TransformStamped()
        t.header.frame_id = DStarLite.GLOBAL_FIXED_ID
        t.header.stamp = rospy.Time.now()
        t.child_frame_id = DStarLite.NEXT_FRAME_ID
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        tfm = tf2_msgs.msg.TFMessage([t])

        print("Published Waypoint Transform:", tfm)
        self.pub_tf.publish(tfm)

    def teardown(self):
        self.occupancy_grid.remove_d_star_lite_callback(self.callback)
        self.finish_callback()    