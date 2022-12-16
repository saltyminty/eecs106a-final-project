#!/usr/bin/env python
"""
copied from lab 6
"""

from __future__ import print_function
from collections import deque
from robot_mover import RobotMover
from turtlebot_controller import TurtlebotController
from d_star_lite import DStarLite, RecalculateEnum

import rospy
import message_filters
import ros_numpy
import tf
import sys

import time

from sensor_msgs.msg import Image, CameraInfo, PointCloud2

import numpy as np
import cv2

from cv_bridge import CvBridge

from image_segmentation import segment_image, edge_detect_canny
from pointcloud_segmentation import segment_pointcloud
from occupancy_grid_3d import OccupancyGrid3d
from occupancy_grid_2d import OccupancyGrid2d

def to_grayscale(rgb_img):
    return np.dot(rgb_img[... , :3] , [0.299 , 0.587, 0.114])

def edge_detect_canny(img):
    edges = cv2.Canny(to_grayscale(img), 100, 200)
    return edges

def get_camera_matrix(camera_info_msg):
    # TODO: Return the camera intrinsic matrix as a 3x3 numpy array
    # by retreiving information from the CameraInfo ROS message.
    # Hint: numpy.reshape may be useful here.
    return np.reshape(camera_info_msg.K, (3, 3))

def isolate_object_of_interest(points, image, cam_matrix, trans, rot):
    segmented_image = segment_image(image)
    points = segment_pointcloud(points, segmented_image, cam_matrix, trans, rot)
    return points

def numpy_to_pc2_msg(points):
    return ros_numpy.msgify(PointCloud2, points, stamp=rospy.Time.now(),
        frame_id='camera_depth_optical_frame')

class PointcloudProcess:
    """
    Wraps the processing of a pointcloud from an input ros topic and publishing
    to another PointCloud2 topic.

    """
    def __init__(self, points_sub_topic, 
                       image_sub_topic,
                       cam_info_topic,
                       points_pub_topic):

        self.num_steps = 0

        self.messages = deque([], 5)
        self.pointcloud_frame = None
        points_sub = message_filters.Subscriber(points_sub_topic, PointCloud2)
        image_sub = message_filters.Subscriber(image_sub_topic, Image)
        caminfo_sub = message_filters.Subscriber(cam_info_topic, CameraInfo)
        self._bridge = CvBridge()
        self.listener = tf.TransformListener()
        
        self.points_pub = rospy.Publisher(points_pub_topic, PointCloud2, queue_size=10)
        self.image_pub = rospy.Publisher('segmented_image', Image, queue_size=10)
        
        # self.occupancy_grid = OccupancyGrid3d()
        self.occupancy_grid = OccupancyGrid2d()
        if not self.occupancy_grid.Initialize():
            rospy.logerr("Failed to initialize the mapping node.")
            sys.exit(1)

        ts = message_filters.ApproximateTimeSynchronizer([points_sub, image_sub, caminfo_sub],
                                                          10, 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)

    def callback(self, points_msg, image, info):
        try:
            intrinsic_matrix = get_camera_matrix(info)
            rgb_image = ros_numpy.numpify(image)
            # print(points_msg)
            points = ros_numpy.numpify(points_msg)
            points = ros_numpy.point_cloud2.split_rgb_field(points)
        except Exception as e:
            rospy.logerr(e)
            return
        self.num_steps += 1
        self.messages.appendleft((points, rgb_image, intrinsic_matrix))

    def publish_once_from_queue(self):
        # print("entered publish once from queue")
        if self.messages:
            # print("message exists")
            points, image, info = self.messages.pop()
            try:
                trans, rot = self.listener.lookupTransform(
                                                       '/camera_color_optical_frame',
                                                       '/camera_depth_optical_frame',
                                                       rospy.Time(0))
                rot = tf.transformations.quaternion_matrix(rot)[:3, :3]
            except (tf.LookupException,
                    tf.ConnectivityException, 
                    tf.ExtrapolationException):
                return
            
            #TODO: process pointcloud
            
            
            # points = isolate_object_of_interest(points, image, info, 
            #    np.array(trans), np.array(rot))
            self.processPoints(points)

            # ####  OccupancyGrid3dmpy_to_pc2_msg(points)
            # points_msg = numpy_to_pc2_msg(points)
            # self.points_pub.publish(points_msg)
            # print("Published segmented pointcloud at timestamp:",
            #        points_msg.header.stamp.secs)

    def processPoints(self, message):
        
        xyz_rgb = np.vstack((message['z'], -1 * message['x'], -1 * message['y'], message['r'], message['g'], message['b'])).T
        #todo, transform to fixed frame
        # update occupancy grid
        # print(np.max(xyz, axis = 0), np.min(xyz, axis = 0), np.mean(xyz, axis = 0), np.median(xyz, axis = 0))
        print("Process Points: Calling Sensor Callback")
        self.occupancy_grid.SensorCallback_2(xyz_rgb)
        print("ProcessPoints: published occupancy grid")

        # # test_purple_center = self.occupancy_grid.SegmentVoxelByColor(r_min = 200, r_max = 230, g_min = 100, g_max = 120, b_min = 200, b_max = 225)
        # test_purple_center = self.occupancy_grid.SegmentVoxelByColor(r_min = 100, r_max = 230, g_min = 100, g_max = 230, b_min = 100, b_max = 225)
        # # test_green_center = self.occupancy_grid.SegmentVoxelByColor(r_min = 160, r_max = 185, g_min = 240, g_max = 265, b_min = 30, b_max = 60)
        # print(test_purple_center)

    # header = message.header
    # height = message.height
    # width = message.width
    # pointField = message.fields
    # point_step = message.point_step
    # row_step = message.row_step
    # data = message.data


def main():
    CAM_INFO_TOPIC = '/camera/color/camera_info'
    RGB_IMAGE_TOPIC = '/camera/color/image_raw'
    POINTS_TOPIC = '/camera/depth/color/points'
    POINTS_PUB_TOPIC = 'segmented_points'

    rospy.init_node('realsense_listener')
    process = PointcloudProcess(POINTS_TOPIC, RGB_IMAGE_TOPIC,
                                CAM_INFO_TOPIC, POINTS_PUB_TOPIC)
    r = rospy.Rate(1000)
    time.sleep(1)
    process.publish_once_from_queue()

    # problem = DStarLite(process.occupancy_grid, np.array([128, 93]), np.array([0, 0]), lambda : print("Finished"), recalculate=RecalculateEnum.NONE)
    # problem.compute_shortest_path()

    # mover = RobotMover(process.occupancy_grid)
    # 128.69444444  93.27777778   2.19444444
    # goal = {"chair": [np.array([0, 0]), np.array([50, 50])]}
    # mover.set_goals(goal["chair"])
    # chairs_found = mover.move_next_object()
    # while not chairs_found:
    #     r.sleep()
    #     process.publish_once_from_queue()
    #     chairs_found = mover.move_next_object()


    while not rospy.is_shutdown():
        process.publish_once_from_queue()
        # print("SLEEP")
        r.sleep()

if __name__ == '__main__':
    main()
