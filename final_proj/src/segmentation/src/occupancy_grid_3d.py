################################################################################
#
# Adapted from OccupancyGrid2d from Lab 8, extended to 3D
#
################################################################################

# roslaunch realsense2_camera rs_d435_camera_with_model.launch

# roslaunch realsense2_camera rs_camera.launch enable_gyro:=true enable_accel:=true align_depth:=true unite_imu_method:="copy" initial_reset:=true filters:=pointcloud
import rospy
import tf2_ros
import tf

import tf2_msgs.msg


from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, TransformStamped
from std_msgs.msg import ColorRGBA

from sklearn.cluster import AgglomerativeClustering

import numpy as np

class ColorClusters(object):
    def __init__(self):
        self.center_voxel = None
        self.color = None
        self.cluster_indices = None


class OccupancyGrid3d(object):
    def __init__(self):
        self._intialized = False

        # Set up tf buffer and listener.
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
        self._br = tf2_ros.TransformBroadcaster()
        self._name = rospy.get_name() + "/grid_map_3d"

        self.clusters = {"purple": ColorClusters(), "green": ColorClusters()}

        self.ignore = None


    # Initialization and loading parameters.
    def Initialize(self):

        # Load parameters.
        if not self.LoadParameters():
            rospy.logerr("%s: Error loading parameters.", self._name)
            return False

        # Register callbacks.
        if not self.RegisterCallbacks():
            rospy.logerr("%s: Error registering callbacks.", self._name)
            return False

        # Set up the map.
        self._map = np.zeros((self._x_num, self._y_num, self._z_num))
        self._rgb = np.zeros((self._x_num, self._y_num, self._z_num, 3))

        self._initialized = True
        return True

    def LoadParameters(self):
        # Random downsampling fraction, i.e. only keep this fraction of rays.
        self._random_downsample = 0.2

        # Dimensions and bounds.
        self._x_num = 200
        self._x_min = -2
        self._x_max = 2
        self._x_res  = (self._x_max - self._x_min) / self._x_num
        self._y_num = 200
        self._y_min = -2
        self._y_max = 2
        self._y_res = (self._y_max - self._y_min) / self._y_num
        self._z_num = 5
        self._z_min = -0
        self._z_max = 0.1
        self._z_res = (self._z_max - self._z_min) / self._z_num

        self._min_dist = 0.1 # in METERS

        # Update parameters.
        self._occupied_update = self.ProbabilityToLogOdds(0.7)
        self._occupied_threshold = self.ProbabilityToLogOdds(0.97)
        self._free_update = self.ProbabilityToLogOdds(0.3)
        self._free_threshold = self.ProbabilityToLogOdds(0.03)

        self._sensor_frame = "base_footprint"
        self._fixed_frame = "odom"
        self._camera_frame = "camera_link"
        self._vis_topic = "/vis/3dmap"

        return True

    def RegisterCallbacks(self):
        # # Subscriber.
        # self._sensor_sub = rospy.Subscriber(self._sensor_topic,
        #                                     LaserScan,
        #                                     self.SensorCallback,
        #                                    Called queue_size=1)

        # Publisher.
        self._vis_pub = rospy.Publisher(self._vis_topic,
                                        Marker,
                                        queue_size=1000)

        self.pub_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)


        return True

    # Callback to process sensor measurements.
    def SensorCallback(self, msg):
        if not self._initialized:
            rospy.logerr("%s: Was not initialized.", self._name)
            return

        # Get our current pose from TF.
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
        #print(pose)
        # Extract x, y coordinates and heading (yaw) angle of the turtlebot, 
        # assuming that the turtlebot is on the ground plane.
        sensor_x = 0 # pose.transform.translation.x
        sensor_y = 0 # pose.transform.translation.y
        # if abs(pose.transform.translation.z) > 0.05:
        #     rospy.logwarn("%s: Turtlebot is not on ground plane.", self._name)

        # (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
        #     [pose.transform.rotation.x, pose.transform.rotation.y,
        #      pose.transform.rotation.z, pose.transform.rotation.w])
        # if abs(roll) > 0.1 or abs(pitch) > 0.1:
        #     rospy.logwarn("%s: Turtlebot roll/pitch is too large.", self._name)

        # modification to loop over pointcloud
        for idx, point in enumerate(msg):
            if np.random.rand() > self._random_downsample:
                    continue
            elif np.isnan(point).any():
                continue
            
            # TODO: transform coords to global frame
            # TODO: check if out of max range?
            curr_pos = np.array([sensor_x, sensor_y, 0])
            slope = point / np.linalg.norm(point) # (point - curr_pos) / np.linalg.norm(point - curr_pos)
            slope = np.reshape(slope, (-1, 1))
            dist = np.linalg.norm(point) # np.linalg.norm(point - curr_pos)

            if dist < self._min_dist:
                continue

            occupied = np.reshape(np.arange(0, dist, min([self._x_res, self._y_res, self._z_res])), (-1, 1))
            
            # x_arr_along_ray = curr_pos[0] + occupied * slope[0]
            # y_arr_along_ray = curr_pos[1] + occupied * slope[1]
            # z_arr_along_ray = curr_pos[2] + occupied * slope[2]
            points_along_voxel = (curr_pos + occupied @ slope.T)
            # print(occupied @ slope.T)

            voxels_along_ray = [self.PointToVoxel(*points_along_voxel[i]) for i in range(len(occupied))]
            voxels_unique_along_ray = np.unique(np.array(voxels_along_ray), axis = 0)

            # raise NotImplementedError
            # break
            # unoccupiedCalled
            for voxel_x, voxel_y, voxel_z in voxels_unique_along_ray[:-1]:
                if (voxel_x < 0 or voxel_x >= self._x_num or
                    voxel_y < 0 or voxel_y >= self._y_num or
                    voxel_z < 0 or voxel_z >= self._z_num):
                    continue
                curr_odds = self._map[voxel_x, voxel_y, voxel_z]
                new_odds = max(curr_odds + self._free_update, self._free_threshold)
                self._map[voxel_x, voxel_y, voxel_z] = new_odds
            
            # occupied
            voxel_x, voxel_y, voxel_z = voxels_unique_along_ray[-1]
            if not (voxel_x < 0 or voxel_x >= self._x_num or
                    voxel_y < 0 or voxel_y >= self._y_num or
                    voxel_z < 0 or voxel_z >= self._z_num):
                curr_odds = self._map[voxel_x, voxel_y, voxel_z]
                new_odds = min(curr_odds + self._occupied_update, self._occupied_threshold)
                self._map[voxel_x, voxel_y, voxel_z] = new_odds
                
            
        # Visualize.
        self.Visualize()
    
    # attempt to better optimize the 3d occupancy map update call
    # USING THIS ONE
    def SensorCallback_2(self, msg):
        if not self._initialized:
            rospy.logerr("%s: Was not initialized.", self._name)
            return
        
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
        
        # Extract x, y coordinates and heading (yaw) angle of the turtlebot, 
        # assuming that the turtlebot is on the ground plane.
        sensor_x = pose.transform.translation.x # 0.01 = 1 meter??
        sensor_y = pose.transform.translation.y
        
        if abs(pose.transform.translation.z) > 0.05:
            rospy.logwarn("%s: Turtlebot is not on ground plane.", self._name)

        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
            [pose.transform.rotation.x, pose.transform.rotation.y,
             pose.transform.rotation.z, pose.transform.rotation.w])
        if abs(roll) > 0.1 or abs(pitch) > 0.1:
            rospy.logwarn("%s: Turtlebot roll/pitch is too large.", self._name)

        # t = TransformStamped()
        # t.header.stamp = rospy.Time.now()
        # t.header.frame_id = self.

        # self._br.sendTransform(pose.transform.translation, pose.transform.rotation, rospy.Time.now(), self._camera_frame, "camera_rotated")
        # print(sensor_x * 1000, sensor_y * 1000)
        u_fac = 1
        # print(u_fac * sensor_x, u_fac * sensor_y)
        curr_voxel = np.array(self.PointToVoxel(u_fac * sensor_x, u_fac * sensor_y, 0)) #change
        
        #todo: transform all the points
        # occupied_voxels = [np.array(self.PointToVoxel(\
        #     np.cos(yaw) * point[0] - np.sin(yaw) * point[1],\
        #     np.sin(yaw) * point[0] + np.cos(yaw) * point[1],\s
        #     point[2])) for point in msg]

        
        # occupied_voxels = [np.array(self.PointToVoxel(\
        #     np.cos(yaw) * (point[0] + u_fac * sensor_x) - np.sin(yaw) * (point[1] + u_fac * sensor_y),\
        #     np.sin(yaw) * (point[0] + u_fac * sensor_x) + np.cos(yaw) * (point[1] + u_fac * sensor_y),\
        #     point[2])) for point in msg]
        
        occupied_voxels = [np.array(self.PointToVoxel(\
            np.cos(yaw) * (point[0]) - np.sin(yaw) * (point[1]) + u_fac * sensor_x,\
            np.sin(yaw) * (point[0]) + np.cos(yaw) * (point[1]) + u_fac * sensor_y,\
            point[2])) for point in msg]
        voxel_colors = msg[:, 3:6]

        for i, voxel in enumerate(occupied_voxels):
            if np.random.rand() > self._random_downsample:
                continue

            slope = (voxel - curr_voxel) / np.linalg.norm(voxel - curr_voxel)  # voxel / np.linalg.norm(voxel) # 
            slope = np.reshape(slope, (-1, 1))
            dist = np.linalg.norm(voxel - curr_voxel)

            if dist < self._min_dist:
                continue

            occupied = np.reshape(np.arange(0, dist + 1, 1), (-1, 1))


            voxels_along_ray = (curr_voxel + occupied @ slope.T).astype(np.int)
            voxels_unique_along_ray = np.unique(np.array(voxels_along_ray), axis = 0)
            
            # unoccupiedCalled
            for voxel_x, voxel_y, voxel_z in voxels_unique_along_ray[:-1]:
                if (voxel_x < 0 or voxel_x >= self._x_num or
                    voxel_y < 0 or voxel_y >= self._y_num or
                    voxel_z < 0 or voxel_z >= self._z_num):
                    continue
                # print(voxel_x, voxel_y, voxel_z)
                curr_odds = self._map[voxel_x, voxel_y, voxel_z]
                new_odds = max(curr_odds + self._free_update, self._free_threshold)
                self._map[voxel_x, voxel_y, voxel_z] = new_odds
            
            # occupied
            voxel_x, voxel_y, voxel_z = voxels_unique_along_ray[-1]
            if not (voxel_x < 0 or voxel_x >= self._x_num or
                    voxel_y < 0 or voxel_y >= self._y_num or
                    voxel_z < 0 or voxel_z >= self._z_num):
                curr_odds = self._map[voxel_x, voxel_y, voxel_z]
                new_odds = min(curr_odds + self._occupied_update, self._occupied_threshold)
                self._map[voxel_x, voxel_y, voxel_z] = new_odds
                self._rgb[voxel_x, voxel_y, voxel_z, :] = voxel_colors[i]
                        
        # Visualize.
        self.Visualize(sensor_x, sensor_y, yaw)
        self.UpdateClusters()
        print("After update")
        self.MaskMap()


    ###### HACK ####
    def get_transform_from_goal(self, point):
        x, y, _ = self.VoxelCenter(point[0], point[1], 0)
        t = TransformStamped()
        t.header.frame_id = "odom"
        t.header.stamp = rospy.Time.now()
        t.child_frame_id = "next_waypoint"
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

    def publish_rotation_transform(self):
        t = TransformStamped()
        t.header.frame_id = "odom"
        t.header.stamp = rospy.Time.now()
        t.child_frame_id = "next_waypoint"
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.1

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        tfm = tf2_msgs.msg.TFMessage([t])

        print("Published Waypoint Transform:", tfm)
        self.pub_tf.publish(tfm)


    #vectorize everything wheee
    def SensorCallback_3(self, msg):
        if not self._initialized:
            rospy.logerr("%s: Was not initialized.", self._name)
            return
        sensor_x = 0 # pose.transform.translation.x
        sensor_y = 0 # pose.transform.translation.y
        
        for idx, point in enumerate(msg):
            
            # TODO: transform coords to global frame
            # TODO: check if out of max range?
            curr_pos = np.array([sensor_x, sensor_y, 0])
            slope = point / np.linalg.norm(point) # (point - curr_pos) / np.linalg.norm(point - curr_pos)
            slope = np.reshape(slope, (-1, 1))
            dist = np.linalg.norm(point) # np.linalg.norm(point - curr_pos)

            if dist < self._min_dist:
                continue

            occupied = np.reshape(np.arange(0, dist, min([self._x_res, self._y_res, self._z_res])), (-1, 1))

            points_along_voxel = (curr_pos + occupied @ slope.T)

            voxels_along_ray = [self.PointToVoxel(*points_along_voxel[i]) for i in range(len(occupied))]
            voxels_unique_along_ray = np.unique(np.array(voxels_along_ray), axis = 0)

            # raise NotImplementedError
            # break
            # unoccupiedCalled
            within_bounding_box = \
                np.logical_and( \
                    np.logical_and( \
                        np.logical_and(voxels_unique_along_ray[:, 0] >= 0, voxels_unique_along_ray[:, 0] < self._x_num), \
                        np.logical_and(voxels_unique_along_ray[:, 1] >= 0, voxels_unique_along_ray[:, 1] < self._y_num)
                    ), \
                    np.logical_and(voxels_unique_along_ray[:, 2] >= 0, voxels_unique_along_ray[:, 2] < self._z_num)
                )
            
            voxels_unique_filtered = voxels_unique_along_ray[within_bounding_box]
            unoccupied_voxels = voxels_unique_filtered[:-1]
            occupied_voxel = voxels_unique_filtered[-1]

            curr_odds = self._map[unoccupied_voxels[:, 0], unoccupied_voxels[:, 1], unoccupied_voxels[:, 2]]
            new_odds = np.maximum(curr_odds + self._free_update, self._free_threshold)
            self._map[unoccupied_voxels[:, 0], unoccupied_voxels[:, 1], unoccupied_voxels[:, 2]] = new_odds

            curr_odds = self._map[occupied_voxel[0], occupied_voxel[1], occupied_voxel[2]]
            new_odds = min(curr_odds + self._occupied_update, self._occupied_threshold)
            self._map[occupied_voxel[0], occupied_voxel[1], occupied_voxel[2]] = new_odds 
            # for voxel_x, voxel_y, voxel_z in voxels_unique_along_ray[:-1]:
            #     if (voxel_x < 0 or voxel_x >= self._x_num or
            #         voxel_y < 0 or voxel_y >= self._y_num or
            #         voxel_z < 0 or voxel_z >= self._z_num):
            #         continue
            #     curr_odds = self._map[voxel_x, voxel_y, voxel_z]

            #     new_odds = max(curr_odds + self._free_update, self._free_threshold)
            #     self._map[voxel_x, voxel_y, voxel_z] = new_odds
            
            # # occupied
            # voxel_x, voxel_y, voxel_z = voxels_unique_along_ray[-1]
            # if not (voxel_x < 0 or voxel_x >= self._x_num or
            #         voxel_y < 0 or voxel_y >= self._y_num or
            #         voxel_z < 0 or voxel_z >= self._z_num):
            #     curr_odds = self._map[voxel_x, voxel_y, voxel_z]
            #     new_odds = min(curr_odds + self._occupied_update, self._occupied_threshold)
            #     self._map[voxel_x, voxel_y, voxel_z] = new_odds
            
        # Visualize.
        self.Visualize()
        
    # Convert (x, y) coordinates in fixed frame to grid coordinates.
    def PointToVoxel(self, x, y, z):
        grid_x = int((x - self._x_min) / self._x_res)
        grid_y = int((y - self._y_min) / self._y_res)
        grid_z = int((z - self._z_min) / self._z_res)

        return (grid_x, grid_y, grid_z)

    # Get the center point (x, y) corresponding to the given voxel.
    def VoxelCenter(self, ii, jj, kk):
        center_x = self._x_min + (0.5 + ii) * self._x_res
        center_y = self._y_min + (0.5 + jj) * self._y_res
        center_z = self._z_min + (0.5 + kk) * self._z_res

        return (center_x, center_y, center_z)

    # Convert between probabity and log-odds.
    def ProbabilityToLogOdds(self, p):
        return np.log(p / (1.0 - p))

    def LogOddsToProbability(self, l):
        return 1.0 / (1.0 + np.exp(-l))

    # plot occupancy grid with color of voxel
    def Colormap(self, ii, jj, kk):
        p = self.LogOddsToProbability(self._map[ii, jj, kk])

        c = ColorRGBA()
        c.r = self._rgb[ii, jj, kk, 0] / 255
        c.g = self._rgb[ii, jj, kk, 1] / 255
        c.b = self._rgb[ii, jj, kk, 2] / 255
        c.a = max(0, (p - 0.5) * 2)
        # print(c)
        return c
    
    def Colormap_old(self, ii, jj, kk):
        p = self.LogOddsToProbability(self._map[ii, jj, kk])

        c = ColorRGBA()
        c.r = p
        c.g = 0.1
        c.b = 1.0 - p
        c.a = 0.1 * abs(p - 0.5) # 0.75
        # print(c)
        return c

    # Visualize the map as a collection of flat cubes instead of
    # as a built-in OccupancyGrid message, since that gives us more
    # flexibility for things like color maps and stuff.
    # See http://wiki.ros.org/rviz/DisplayTypes/Marker for a brief tutorial.
    def Visualize(self, sensor_x, sensor_y, yaw):
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
                for kk in range(self._z_num):
                    p = Point(0.0, 0.0, 0.0)
                    (p.x, p.y, p.z) = self.VoxelCenter(ii, jj, kk)
                    # p.x, p.y = cos * p.x - sin * p.y, sin * p.x + cos * p.y

                    p.x, p.y = cos * (p.x + trans_x) - sin * (p.y + trans_y),\
                        sin * (p.x + trans_x) + cos * (p.y + trans_y)

                    m.points.append(p)
                    m.colors.append(self.Colormap(ii, jj, kk))

        self._vis_pub.publish(m)
    
    
    def SegmentVoxelByColor(self, center_colors, rg_min, rg_max, rb_min, rb_max, gb_min, gb_max, rgb_min):

        return_indices = []
        for i, color in enumerate(center_colors):
            r, g, b = color
            if r >= rgb_min and g >= rgb_min and b >= rgb_min:
                rg_ratio = r / g
                rb_ratio = r / b
                gb_ratio = g / b
                if rg_ratio > rg_min and rg_ratio < rg_max \
                    and rb_ratio > rb_min and rb_ratio < rb_max \
                    and gb_ratio > gb_min and gb_ratio < gb_max:
                    return_indices.append(i)
        return return_indices
    
    def ClusterVoxels(self):
        voxels_filtered = []
        
        for ii in range(np.shape(self._rgb)[0]):
            for jj in range(np.shape(self._rgb)[1]):
                for kk in range(np.shape(self._rgb)[2]):
                    if self.LogOddsToProbability(self._map[ii, jj, kk]) > 0.7:
                        voxels_filtered.append(np.array([ii, jj, kk]))
        voxels_filtered = np.array(voxels_filtered)
        if len(voxels_filtered) == 0:
            return None, None, None
        # print(np.shape(voxels_filtered))
        clustering = AgglomerativeClustering(n_clusters = None, distance_threshold=3, linkage = 'single').fit(voxels_filtered)
        cluster_labels = clustering.labels_
        clusters = np.unique(cluster_labels)
        # print(len(cluster_labels))

        voxel_centers, average_colors, indices_list = [], [], []
        for cluster in clusters:
            indices = np.where(cluster_labels == cluster)[0]
            if len(indices) > 10:
                cluster_voxels = voxels_filtered[indices, :]
                colors = [self._rgb[x, y, z, :] for x, y, z in cluster_voxels]
                voxel_centers.append(np.mean(cluster_voxels, axis = 0))
                average_colors.append(np.mean(colors, axis = 0))
                indices_list.append(indices)
        # print(average_colors)
        return voxel_centers, average_colors, indices_list

    def UpdateClusters(self):
        voxel_centers, center_colors, indices_list = self.ClusterVoxels()
        # print(center_colors)
        print(center_colors)
        if center_colors is not None:
            purple_indices = self.SegmentVoxelByColor(center_colors, rg_min=1.1, rg_max=2, rb_min=0.7, rb_max=1.3, gb_min = 0.5, gb_max = 1 / 1.2, rgb_min = 30)
            if len(purple_indices) > 1:
                print("more than one purple cluster found")
            green_indices = self.SegmentVoxelByColor(center_colors, rg_min=0, rg_max=0.95, rb_min=0, rb_max=10, gb_min = 1.2, gb_max = 10, rgb_min = 30)
            if len(green_indices) > 1:
                print("more than one green cluster found")
            print(green_indices)

            # self.purple_center_voxel = None
            # self.purple_center = None
            # self.purple_cluster_indices = None
            # self.green_center_voxel = None
            # self.green_center = None
            # self.green_cluster_indices = None

            if len(purple_indices):
                self.clusters["purple"].center_voxel = voxel_centers[purple_indices[0]]
                self.clusters["purple"].color = center_colors[purple_indices[0]]
                self.clusters["purple"].cluster_indices = indices_list[purple_indices[0]]
            
            if len(green_indices):
                self.clusters["green"].center_voxel = voxel_centers[green_indices[0]]
                self.clusters["green"].color = center_colors[green_indices[0]]
                self.clusters["green"].cluster_indices = indices_list[green_indices[0]]
            
            print(self.clusters["purple"].center_voxel, self.clusters["green"].center_voxel)

    def MaskMap(self):
        if self.ignore and self.clusters[self.ignore]:
            for point in self.clusters[self.ignore].cluster_indices:
                self._map[point[0], point[1], point[2]] = 0
        # print(voxels_filtered)
        # print(len(voxels_filtered))
        # if len(voxels_filtered) > 0:
        #     return np.mean(voxels_filtered, axis = 0)
        # else:
        #     return None

    def get_current_position(self):
        if not self._initialized:
            rospy.logerr("%s: Was not initialized.", self._name)
            return
        
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
        
        # Extract x, y coordinates and heading (yaw) angle of the turtlebot, 
        # assuming that the turtlebot is on the ground plane.
        sensor_x = pose.transform.translation.x # 0.01 = 1 meter??
        sensor_y = pose.transform.translation.y

        return self.PointToVoxel(sensor_x, sensor_y, 0)
