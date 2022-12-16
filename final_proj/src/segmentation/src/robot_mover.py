import time
from typing import Dict, List, Tuple
from occupancy_grid_2d import OccupancyGrid2d
from d_star_lite import DStarLite, RecalculateEnum

import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg
import rospy

import numpy as np

from enum import Enum

class RobotState(Enum):
    INIT = 1
    ACQUIRE_OBJECT = 2
    PUSH_OBJECT = 3

class RobotMover:
    CHAIR_SIZE = 2 # Size of the chair in grid numbers (length of a side)
    TABLE_SIZE = 10 # size of the table in grid numbers
    NUM_CHAIRS = 2 # Number of chairs in the problem
    NUM_TABLES = 1 # Number of tables in the problem

    NEXT_FRAME_ID = "next_waypoint"
    ROBOT_FRAME_ID = "base_footprint"
    GLOBAL_FIXED_ID = "odom"

    def distance_function(self, x):
        return np.linalg.norm(x - self.current_location)

    def __init__(self, occupancy_grid: OccupancyGrid2d):
        self.occupancy_grid: OccupancyGrid2d = occupancy_grid
        self.goals: Dict[str, List[np.ndarray]] = {}
        self.current_objects: Dict[str, List[str]] = {"chair": ["purple", "green"]}
        self.current_location = self.occupancy_grid.get_current_location()

        self.state = RobotState.INIT
        self.cur_object = None
        self.cur_goal = None
        self.r = rospy.Rate(10)

    def set_goals(self, chair_goals: List[np.ndarray]):
        self.goals["chair"] = chair_goals
        # self.goals["table"] = table_goals

    def closest_item(self) -> Tuple[str, str]:
        self.current_location = self.occupancy_grid.get_current_location()

        if len(self.goals["chair"]) == 0:
            print("All Object Placed!")
            self.state = RobotState.INIT
            return

        key = lambda x: self.distance_function(self.occupancy_grid.object_center(x))
        chair_min = min(self.current_objects["chair"], key=key)

        return ("chair", chair_min)

    def closest_goal(self, object_type: str) -> np.ndarray:
        self.current_location = self.occupancy_grid.get_current_location()
        return min(self.goals[object_type], default=None, key=self.distance_function)

    def move_to_object(self, object_state: Tuple[str, str], orientation: np.ndarray) -> DStarLite:
        self.state = RobotState.ACQUIRE_OBJECT
        side_length: int = RobotMover.CHAIR_SIZE if object_state[0] == "chair" else RobotMover.TABLE_SIZE
        offset = (side_length // 2 + 1) * orientation
        dstarlite = DStarLite(self.occupancy_grid, object_state[1], offset, self.segment_finish, RecalculateEnum.GOAL)
        dstarlite.compute_shortest_path()

    def segment_finish(self):
        if self.state == RobotState.ACQUIRE_OBJECT:
            self.state = RobotState.PUSH_OBJECT
            self.move_to_goal()

        elif self.state ==  RobotState.PUSH_OBJECT:
            self.state = RobotState.INIT
            object_type = self.cur_object[0]
            self.goals[object_type].remove(self.cur_goal)
            self.current_objects[object_type].remove(self.cur_object)
            self.move_next_object()

    def move_next_object(self):
        self.state = RobotState.ACQUIRE_OBJECT
        if len(self.goals["chair"]) == 0:
            return True
        
        self.cur_object = self.closest_item()

        if self.distance_function(self.occupancy_grid.object_center(self.cur_object[1])) == np.inf:
            print("No Chairs found")
            time.sleep(1)
            problem = DStarLite(self.occupancy_grid, 
                                self.occupancy_grid.get_current_location(), 
                                np.zeros((2, )), lambda : print("No Chairs LOL"), 
                                recalculate=RecalculateEnum.NONE)
            problem.compute_shortest_path()
            return False
            

        self.cur_goal = self.closest_goal(self.cur_object[0])
        
        orientation = self.occupancy_grid.object_center(self.cur_object[1]) - self.cur_goal
        # orientation = self.occupancy_grid.object_center(self.cur_object) - self.cur_goal
        orientation = np.sign(orientation) * (np.array([1, 0]) if np.abs(orientation[0]) > np.abs(orientation[1]) else np.array([0, 1]))
        self.move_to_object(self.cur_object, orientation)
        return True

    def move_to_goal(self):
        dstarlite = DStarLite(self.occupancy_grid, self.cur_goal, self.turtlebot_frame, self.segment_finish, RecalculateEnum.OFFSET)
        dstarlite.compute_shortest_path()
