from typing import Tuple
import rospy
import tf2_ros
import tf

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

import numpy as np

from occupancy_grid_3d import OccupancyGrid3d

class OccupancyGrid2d(OccupancyGrid3d):
    ZERO_THRESHOLD = 1e-3

    def __init__(self):
        super().__init__()
        self._name = rospy.get_name() + "/grid_map_2d"
        self._d_star_lite_callback = []

    def Initialize(self):
        super().Initialize()
        self._2dmap = np.zeros((self._x_num, self._y_num))
        self._last_2dmap = np.zeros((self._x_num, self._y_num))
        self._changes = np.zeros((self._x_num, self._y_num))
        return True

    def LoadParameters(self):
        print("Load Parameters Called")
        super().LoadParameters()
        print("Passed Super Load Parameters")
        self._vis_topic = "/vis/2dmap"
        self._z_num = 1
        return True

    def SensorCallback_2(self, msg):
        print("2d called")
        if not self._initialized:
            rospy.logerr("%s: Was not initialized.", self._name)
            return  
        
        super().SensorCallback_2(msg)

        print("Current Location:", self.get_current_location())

        self._last_2dmap = self._2dmap
        self.collapse_into_2D()

        self._changes = self._2dmap - self._last_2dmap
        self._changes[self._changes < OccupancyGrid2d.ZERO_THRESHOLD] = 0

        # self.Visualize()

        for callback in self._d_star_lite_callback:
            callback()
        
    def collapse_into_2D(self):
        self._2dmap = np.max(self._map, axis=2)

    def add_d_star_lite_callback(self, callback):
        self._d_star_lite_callback.append(callback)

    def remove_d_star_lite_callback(self, callback):
        self._d_star_lite_callback.remove(callback)

    def get_edges_from_node(self, node):
        second_node = []
        if node[0] > 0:
            second_node.append(np.array([node[0] - 1, node[1]]))
        if node[0] < self._x_num - 1:
            second_node.append(np.array([node[0] + 1, node[1]]))
        if node[1] > 0:
            second_node.append(np.array([node[0], node[1] - 1]))
        if node[1] < self._y_num - 1:
            second_node.append(np.array([node[0], node[1] + 1]))
        return second_node

    def edge_cost_function(prob1, prob2):
        ## Perhaps modify this function to be smarter
        return np.max((prob1, prob2)) * 10 + 1

    def get_edge_costs(self, first_node, second_node):
        possible_edges = self.get_edges_from_node(first_node)
        if not any(map(lambda x: np.array_equal(x, second_node), possible_edges)):
            return float("inf")
        return OccupancyGrid2d.edge_cost_function(self._map[first_node[0], first_node[1]], self._map[second_node[0], second_node[1]])

    def VoxelCenter2D(self, ii, jj):
        return super().VoxelCenter(ii, jj, 0)[:2]

    def PointToVoxel2D(self, x, y):
        return super().PointToVoxel(x, y, 0)[:2]

    # def Colormap(self, ii, jj):
    #     p = self.LogOddsToProbability(self._2dmap[ii, jj])

    #     c = ColorRGBA()
    #     c.r = p
    #     c.g = 0.1
    #     c.b = 1.0 - p
    #     c.a = 0.75
    #     return c

    def Colormap_old(self, ii, jj):
        p = self.LogOddsToProbability(self._2dmap[ii, jj])

        c = ColorRGBA()
        c.r = p
        c.g = 0.1
        c.b = 1.0 - p
        c.a = 0.1 * abs(p - 0.5) # 0.75
        # print(c)
        return c

    def Visualize(self, sensor_x, sensor_y, yaw):
        print("Visualizing")
        m = Marker()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = self._camera_frame # self._fixed_frame
        m.ns = "map"
        m.id = 0
        m.type = Marker.CUBE_LIST
        m.action = Marker.ADD
        m.scale.x = self._x_res
        m.scale.y = self._y_res
        m.scale.z = self._z_res

        try:
            # print(self._fixed_frame, self._sensor_frame)
            pose = self._tf_buffer.lookup_transform(
                self._fixed_frame, self._sensor_frame, rospy.Time())
            
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            # Writes an error message to the ROS log but does not raise an exception
            rospy.logerr("%s: Could not extract pose from TF.", self._name)
            return

        cos, sin = np.cos(-yaw), np.sin(-yaw)
        trans_x = - 1 * sensor_x
        trans_y = - 1 * sensor_y

        for ii in range(self._x_num):
            for jj in range(self._y_num):
                p = Point(0.0, 0.0, 0.0)
                (p.x, p.y) = self.VoxelCenter2D(ii, jj)
                # p.x, p.y = cos * p.x - sin * p.y, sin * p.x + cos * p.y

                p.x, p.y = cos * (p.x + trans_x) - sin * (p.y + trans_y),\
                    sin * (p.x + trans_x) + cos * (p.y + trans_y)

                m.points.append(p)
                m.colors.append(self.Colormap_old(ii, jj))

        self._vis_pub.publish(m)
    # def Visualize(self):
    #     m = Marker()
    #     m.header.stamp = rospy.Time.now()
    #     m.header.frame_id = self._fixed_frame
    #     m.ns = "map"
    #     m.id = 0
    #     m.type = Marker.CUBE_LIST
    #     m.action = Marker.ADD
    #     m.scale.x = self._x_res
    #     m.scale.y = self._y_res
    #     m.scale.z = 0.01True

    #     for ii in range(self._x_num):
    #         for jj in range(self._y_num):
    #             p = Point(0.0, 0.0, 0.0)
    #             (p.x, p.y) = self.VoxelCenter2D(ii, jj)

    #             m.points.append(p)
    #             m.colors.append(self.Colormap(ii, jj))
    #     self._vis_pub.publish(m)

    def get_current_location(self) -> np.ndarray:
        """ Returns the current location of the turtlebot in 
        terms of grid coordinates. """
        if not self._initialized:
            rospy.logerr("%s: Was not initialized.", self._name)
            return
        
        pose = None
        while not pose:
            try:
                # print(self._fixed_frame, self._sensor_frame)
                pose = self._tf_buffer.lookup_transform(
                    self._fixed_frame, self._sensor_frame, rospy.Time())
                
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                # Writes an error message to the ROS log but does not raise an exception
                rospy.logerr("%s: Could not extract pose from TF.", self._name)
        
        # Extract x, y coordinates and heading (yaw) angle of the turtlebot, 
        # assuming that the turtlebot is on the ground plane.
        sensor_x = pose.transform.translation.x # 0.01 = 1 meter??
        sensor_y = pose.transform.translation.y
        return np.array(self.PointToVoxel2D(sensor_x, sensor_y))

    def object_center(self, color: str):
        if self.clusters[color].center_voxel:
            return self.clusters[color].center_voxel[:2]
        else:
            return np.array([np.inf, np.inf])

# class OccupancyGrid2d:
#     SENSOR_FRAME_ID = RobotMover.ROBOT_FRAME_ID
#     FIXED_FRAME_ID = RobotMover.GLOBAL_FIXED_ID
#     SENSOR_TOPIC = "/vis/3dmap"
#     VIS_TOPIC = "/vis/2dmap"

#     def __init__(self):
#         self._intialized = False

#         # Set up tf buffer and listener.
#         self._tf_buffer = tf2_ros.Buffer()
#         self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
#         self._br = tf2_ros.TransformBroadcaster()
#         self._name = rospy.get_name() + "/grid_map_2d"
#         self._d_star_lite_callback = []

#         # Dimensions and bounds. FROM OCCUPANCY GRID 3D LoadParameters
#         self._x_num = 200
#         self._x_min = -2
#         self._x_max = 2
#         self._x_res  = (self._x_max - self._x_min) / self._x_num
#         self._y_num = 200
#         self._y_min = -2
#         self._y_max = 2
#         self._y_res = (self._y_max - self._y_min) / self._y_num
#         self._z_num = 1

#     def Initialize(self):
#                 # Load parameters.
#         if not self.LoadParameters():
#             rospy.logerr("%s: Error loading parameters.", self._name)
#             return False

#         # Register callbacks.
#         if not self.RegisterCallbacks():
#             rospy.logerr("%s: Error registering callbacks.", self._name)
#             return False

#         # Set up the map.
#         self._map = np.zeros((self._x_num, self._y_num, self._z_num))
#         self._rgb = np.zeros((self._x_num, self._y_num, self._z_num, 3))

#         ## 2D map part
#         self._2dmap = np.zeros((self._x_num, self._y_num))
#         self._last_2dmap = np.zeros((self._x_num, self._y_num))
#         self._changes = np.zeros((self._x_num, self._y_num))

#         self._initialized = True
#         return True

#     def register_callback(self):
#         ## Listen to 3D Occupancy Grid
#         self._sub = rospy.Subscriber(OccupancyGrid2d.SENSOR_TOPIC,
#                                      Marker,
#                                      self.callback
#                                      queue_size=1)

#         ## Publisher.
#         self._vis_pub = rospy.Publisher(OccupancyGrid2d.VIS_TOPIC,
#                                         Marker,
#                                         queue_size=1000)

#         return True

#     ### Message Processing Callback Functions
#     def collapse_into_2D(self):
#         self._2dmap = np.max(self._map, axis=2)

#     def callback(self, msg):
#         if not self._initialized:
#             rospy.logerr("%s: Was not initialized.", self._name)
#             return

        