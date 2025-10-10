import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from typing import List, Dict, Tuple
import urdfpy


class URDFKinematicsSolver:
    """
    A simple kinematics solver using URDF and inverse Jacobian method.
    Uses urdfpy for URDF parsing and kinematics.
    Uses Numpy and SciPy for numerical computations. 
    """

    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self.robot = urdfpy.URDF.load(urdf_path)

        ## Extract joint infromation from URDF
        self.joints = []
        self.joint_names = []
        self.joint_limits = {}

        for joint in self.robot.joints:
            # We are only considering serial chains with joints that rotate or are linear
            # As opposed to something like a hexbot which has spherical joints (3R in one joint)
            if joint.joint_type in ['revolute', 'prismatic']:
                self.joints.append(joint)
                self.joint_names.append(joint.name)

                # Add joint limits
                if joint.limit is not None:
                   lower = joint.limit.lower if joint.limit.lower is not None else -np.pi
                   upper = joint.limit.upper if joint.limit.upper is not None else np.pi
                   self.joint_limits.append((lower, upper))
                else:
                    self.joint_limits.append((-np.pi, np.pi))

            # build serial kinematic chain, we assume one end effector 
            # for this exercise
            self.link_names = [link.name for link in self.robot.links]
            self.end_effector_link = self.link_names[-1]

            print(f"Loaded URDF with {len(self.joints)} joints.")
            print(f"End-effector link: {self.end_effector_link}")


def plan_trajectory(current_joints: dict[str, float],
                    desired_eef_pose: tuple[float, float, float, float, float, float]
                   ) -> list[dict[str, float]]:
    """
    Input:
        current_joints – mapping joint name → current angle (radians)
        desired_eef_pose – (x, y, z, roll, pitch, yaw) in XYZ + RPY convention

    Output:
        List of joint-configuration waypoints, each {joint_name: angle},
        forming a smooth path from the current configuration to one that
        achieves the desired end-effector pose.
    """
def time_parameterize_trajectory(
        waypoints: list[dict[str, float]],
        max_vel: float = 1.0,
        max_acc: float = 2.0
    ) -> list[tuple[float, dict[str, float]]]:
    """
    Input:
        waypoints – joint-space path from Part 1
        max_vel, max_acc – scalar joint-space limits

    Output:
        List of (time, joint_dict) tuples giving absolute timestamps
        that satisfy velocity and acceleration bounds.
    """


def main():
    current_joints = {
        "joint1": 0.0,
        "joint2": 0.0,
        "joint3": 0.0
    }
    desired_eef_pose = (1.0, 0.0, 0.5, 0.0, 0.0, 0.0)

    waypoints = plan_trajectory(current_joints, desired_eef_pose)
    timed_trajectory = time_parameterize_trajectory(waypoints)  
    for t, joint_angles in timed_trajectory:
        print(f"Time: {t:.2f}, Joint Angles: {joint_angles}")


if __name__ == "__main__":
    main()
    
