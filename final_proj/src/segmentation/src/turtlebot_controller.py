import rospy
import tf2_ros
import sys

import numpy as np

from collections import deque

from geometry_msgs.msg import Twist

from robot_mover import RobotMover

from enum import Enum


class TurtlebotController():
    Kp = [-0.1, 0.5]
    Ki = [0.01, 0.05]
    Kd = [-0.01, 0.05]
    Kw = [0.5, 0.5]
    

    TURTLEBOT_FRAME_ID = RobotMover.ROBOT_FRAME_ID
    GOAL_FRAME_ID = "next_waypoint"

    def __init__(self):
        # Thread.__init__(self)
        self.initialized = False
        self.int_error = np.zeros((2, ))
        self.last_error = np.zeros((2, ))
        self.ring_buff_capacity = 3
        self.ring_buff = deque([], self.ring_buff_capacity)


    def control_iterate(self):
        try:
            trans = self._tf_buffer.lookup_transform(TurtlebotController.TURTLEBOT_FRAME_ID, 
                                                    TurtlebotController.GOAL_FRAME_ID, rospy.Time(0))

            if trans.transform.translation.z > 0.001:
                control_command = Twist()# Generate this
                control_command.angular.z = 0.01
                self._pub.publish(control_command)
                print(control_command)
                return control_command
            
            error = np.array([-trans.transform.translation.x, trans.transform.translation.y])
            self.int_error = TurtlebotController.Kw * self.int_error + error
            
            curr_derivative = error - self.last_error
            self.ring_buff.append(curr_derivative)

            derivative_error = np.mean(self.ring_buff, axis=0)
            print(curr_derivative)

            self.last_error = error


            control_command = Twist()# Generate this
            control_command.linear.x = self.Kp[0] * error[0] + self.Ki[0] * self.int_error[0] + self.Kd[0] * derivative_error[0]
            control_command.angular.z = self.Kp[1] * error[1] + self.Ki[1] * self.int_error[1] + self.Kd[1] * derivative_error[1]
            
            self._pub.publish(control_command)
            print(control_command)
            return control_command

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(e)
            pass

    def run(self):
        rospy.init_node('turtlebot_controller', anonymous=True)

        self._pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self.rate = rospy.Rate(10) # 10hz

        self.initialized = True

        # ts = message_filters.ApproximateTimeSynchronizer(["\tf"],
        #                                                   10, 0.1, allow_headerless=True)
        while not rospy.is_shutdown():
            self.control_iterate()
            self.rate.sleep()

if __name__ == "__main__":
    controller = TurtlebotController()
    controller.run()
