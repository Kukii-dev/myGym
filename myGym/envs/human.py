from typing import List, Tuple, Optional
import os
import random

import numpy as np
import importlib.resources as pkg_resources

from myGym.envs.env_object import EnvObject
from myGym.utils.helpers import get_robot_dict


def get_pointing_quaternion(p, ee_pos, target_pos):
    """
    Calculates a quaternion to orient an end effector at ee_pos to point at target_pos.
    This is achieved by calculating the required yaw and pitch from the direction vector.

    Args:
        p: PyBullet client instance.
        ee_pos (list or np.array): The end effector position.
        target_pos (list or np.array): The position to point at.

    Returns:
        list: A quaternion [x, y, z, w].
    """
    ee_pos = np.array(ee_pos)
    target_pos = np.array(target_pos)

    # Calculate the direction vector from the end effector to the target
    direction = target_pos - ee_pos

    # Handle case where the target is at the same position as the end effector
    if np.linalg.norm(direction) < 1e-6:
        return [0, 0, 0, 1]  # Return identity quaternion (no rotation)

    # Calculate Yaw (rotation around Z-axis)
    yaw = np.arctan2(direction[1], direction[0])

    # Calculate Pitch (rotation around Y-axis)
    horizontal_dist = np.sqrt(direction[0]**2 + direction[1]**2)
    pitch = np.arctan2(-direction[2], horizontal_dist)

    # Roll is 0 for simple pointing
    roll = 0

    # Convert the Euler angles (roll, pitch, yaw) to a quaternion
    return p.getQuaternionFromEuler([roll, pitch, yaw])


def _link_name_to_idx(p, body_id: int, link_name: str) -> int:
    for i in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, i)
        if info[-5].decode('utf-8') == link_name:
            return i
    exc = f"Cannot find a link index for the link with name {link_name}"
    raise Exception(exc)


class Human:
    """
    Class for control of human-environment interaction.

    Parameters:
        :param model_name: (string) Model name from the get_robot_dict() dictionary
        :param pybullet_client: Which pybullet client the environment should refer to in case of parallel existence
        of multiple instances of this environment
    """
    def __init__(self,
                 model_name: str = "human",
                 pybullet_client=None,
                 direction_point: np.array = np.array([0, 0.8, 0]),
                 links_for_direction_vector: Tuple[str, str] = ("endeffector", "r_index2")
                 ):

        self.p = pybullet_client
        self.body_id: int or None = None
        self.n_joints: int or None = None
        self.motors_indices = []
        self.n_motors: int or None = None
        self.end_effector_idx: int or None = None
        self.direction_point = direction_point  # for pointing during a testing phase

        self._load_model(model_name)
        self._set_motors()

        self.links_indices_for_direction_vector = (
            _link_name_to_idx(self.p, self.body_id, links_for_direction_vector[0]),
            _link_name_to_idx(self.p, self.body_id, links_for_direction_vector[1])
        )

    def _load_model(self, model_name):
        """
        Load SDF or URDF model of specified model and place it in the environment to specified position and orientation.

        Parameters:
            :param model_name: (string) Model name in the get_robot_dict() dictionary
        """
        robot_info = get_robot_dict()[model_name]
        path = robot_info['path']
        position = robot_info['position']
        orientation = robot_info['orientation']
        path = os.path.join(pkg_resources.files("myGym"), path.lstrip("/"))
        orientation = self.p.getQuaternionFromEuler(orientation)

        if path[-3:] == 'sdf':
            self.body_id = self.p.loadSDF(path)[0]
            self.p.resetBasePositionAndOrientation(self.body_id, position, orientation)
        else:
            self.body_id = self.p.loadURDF(path, position, orientation, useFixedBase=True) #  flags=(self.p.URDF_USE_SELF_COLLISION)

        self.n_joints = self.p.getNumJoints(self.body_id)
        for jid in range(self.n_joints):
            self.p.changeDynamics(self.body_id, jid, collisionMargin=0., contactProcessingThreshold=0.0, ccdSweptSphereRadius=0)

    def _set_motors(self):
        """
        Identify motors among all joints (fixed joints aren't motors).
        Identify index of end-effector link among all links. Uses data from human model.
        """
        for i in range(self.n_joints):
            info = self.p.getJointInfo(self.body_id, i)
            q_index = info[3]
            link_name = info[12]

            if q_index > -1:
                self.motors_indices.append(i)

            if 'endeffector' in link_name.decode('utf-8'):
                self.end_effector_idx = i

        self.n_motors = len(self.motors_indices)

        if self.end_effector_idx is None:
            print("No end effector detected. "
                  "Please define which link is an end effector by adding 'endeffector' to the name of the link")
            exit()

    def __repr__(self):
        """
        Get overall description of the human. Used mainly for debug.

        Returns:
            :return description: (string) Overall description
        """
        params = {'Id': self.body_id,
                  'Number of joints': self.n_joints,
                  'Number of motors': self.n_motors}
        description = 'Human parameters\n' + '\n'.join([k + ': ' + str(v) for k, v in params.items()])
        return description

    def _run_motors(self, motor_poses):
        """
        Move joint motors towards desired joint poses respecting model's dynamics

        Parameters:
            :param motor_poses: (list) Desired poses of individual joints
        """
        self.p.setJointMotorControlArray(self.body_id,
                                         self.motors_indices,
                                         self.p.POSITION_CONTROL,
                                         motor_poses,
                                         )

    def _calculate_motor_poses(self, end_effector_pos, orientation: Optional[list] = None):
        """
        Calculate motor poses corresponding to desired position of end-effector. Uses inverse kinematics.

        Parameters:
            :param end_effector_pos: (list) Desired position of end-effector in the environment [x,y,z]
            :param orientation: (list, optional) Desired orientation as quaternion [x,y,z,w]. If None, uses pointing IK.
        Returns:
            :return motor_poses: (list) Calculated motor poses corresponding to the desired end-effector position
        """
        if orientation is not None:
            return self.p.calculateInverseKinematics(
                self.body_id,
                self.end_effector_idx,
                end_effector_pos,
                orientation
            )
        return self.p.calculateInverseKinematics(
            self.body_id,
            self.end_effector_idx,
            end_effector_pos,
        )

    def point_finger_at(self, position=None, relative=False, use_pointing_ik=True):
        """
        Point human's finger towards the desired position using orientation-aware IK.

        Parameters:
            :param position: (list) Cartesian coordinates [x,y,z]
            :param relative: (bool) If True, position is relative to current direction_point
            :param use_pointing_ik: (bool) If True, uses pointing quaternion for orientation-aware IK
        """
        if relative:
            if position is not None:
                self.direction_point += position
                position = self.direction_point
            else:
                exc = "You must pass relative coordinates if a relative option has been chosen"
                raise Exception(exc)
        else:
            if position is None:
                position = self.direction_point

        if use_pointing_ik:
            # Get current end effector position to calculate pointing orientation
            link_state = self.p.getLinkState(self.body_id, self.end_effector_idx)
            ee_pos = link_state[0]
            # Calculate the pointing orientation quaternion
            pointing_quat = get_pointing_quaternion(self.p, ee_pos, position)
            self._run_motors(self._calculate_motor_poses(position, pointing_quat))
        else:
            self._run_motors(self._calculate_motor_poses(position))

    def find_object_human_is_pointing_at(self, objects: List[EnvObject]) -> EnvObject:
        if not objects:
            raise Exception("There are no objects!")

        i1, i2 = self.links_indices_for_direction_vector
        p1, p2 = self.p.getLinkState(self.body_id, i1)[0], self.p.getLinkState(self.body_id, i2)[0]
        p1, p2 = np.array(p1), np.array(p2)
        vec = (p1 - p2) / np.linalg.norm(p1 - p2)
        points = np.array([o.get_position() for o in objects])

        points -= p2.reshape(1, -1)  # move points to be able to compute projections (make the vector relatively centered)
        scalars = np.dot(points, vec)  # scalar product (as a part of computing a projection)
        points_proj = scalars.reshape(-1, 1) * vec.reshape(1, -1)  # projections on the vector
        points_rej = points - points_proj  # rejections
        distances = np.linalg.norm(points_rej, axis=1)
        return objects[np.argmin(distances)]

    def select_random_target_and_point(self, objects: List[EnvObject], num_targets: int = 3) -> Tuple[EnvObject, np.ndarray]:
        """
        Select a random target from a subset of objects and point at it using orientation-aware IK.
        Returns the selected object and its position as a goal for a Tiago robot.

        Parameters:
            :param objects: (List[EnvObject]) List of available objects to choose from
            :param num_targets: (int) Number of targets to consider (default 3)
        Returns:
            :return: (Tuple[EnvObject, np.ndarray]) The selected target object and its position as Tiago goal
        """
        if not objects:
            raise Exception("There are no objects!")

        # Select up to num_targets random objects from the list
        available_objects = objects[:min(num_targets, len(objects))]
        if len(objects) > num_targets:
            available_objects = random.sample(objects, num_targets)

        # Randomly select one target from the available objects
        selected_target = random.choice(available_objects)
        target_position = np.array(selected_target.get_position())

        # Point at the selected target using orientation-aware IK
        self.point_finger_at(target_position, use_pointing_ik=True)

        # Store the current target for reference
        self.direction_point = target_position
        self.current_target = selected_target

        return selected_target, target_position

    def get_tiago_goal_from_pointing(self) -> np.ndarray:
        """
        Get the current pointing target position as a goal for the Tiago robot.

        Returns:
            :return: (np.ndarray) The 3D position [x, y, z] that the human is pointing at
        """
        return np.array(self.direction_point)
