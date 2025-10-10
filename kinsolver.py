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

    # Returns one ndarray of shape (4,4) - a transfromation matrix that
    # encodes the position and rotation of the end effector
    def forward_kinematics(self, joint_angles: List[float]) -> np.ndarray:
        """
        Compute the forward kinematics to get the end-effector pose.
        joint_angles: List of joint angles in radians.
        Returns a 4x4 transformation matrix representing the end-effector pose.
        """
        joint_dict = {name: angle for name, angle in zip(self.joint_names, joint_angles)}
        fk = self.robot.link_fk(joint_dict)
        return fk[self.end_effector_link]
    
    def compute_jacobian(self, joint_angles: List[float]) -> np.ndarray:
        """
        Compute the Jacobian matrix at the current joint angles.
        joint_angles: List of joint angles in radians.
        Returns a 6xN Jacobian matrix, where N is the number of joints.
        """
        joint_dict = {name: angle for name, angle in zip(self.joint_names, joint_angles)}
        J = self.robot.jacobian(joint_dict, self.end_effector_link)
        return J
    
    def get_pose_from_transform(self, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract position and orientation (as quaternion) from a 4x4 transformation matrix.
        Returns: (position (3,), orientation_quat (4,))
        """
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        orientation_quat = R.from_matrix(rotation_matrix).as_quat()
        return position, orientation_quat
    
    def is_valid(self, q: List[float]) -> bool:
        """Check joint limits and basic collision avoidance"""
        # Joint limits
        for i, angle in enumerate(q):
            if i < len(self.joint_limits):
                lower, upper = self.joint_limits[i]
                if not (lower <= angle <= upper):
                    return False
        
        # Simple self-collision check (distance-based)
        try:
            joint_cfg = {name: q[i] if i < len(q) else 0.0 
                        for i, name in enumerate(self.joint_names)}
            fk = self.robot.link_fk(cfg=joint_cfg)
            positions = [T[:3, 3] for T in fk.values()]
            
            for i in range(len(positions)):
                for j in range(i + 2, len(positions)):
                    if np.linalg.norm(positions[i] - positions[j]) < 0.1:
                        return True  # Collision
            return False
        except:
            return False
    
    def inverse_kinematics(self, target_transform: np.ndarray, 
                          max_iter: int = 100) -> Tuple[List[float], bool]:
        """Solve IK using pseudo-inverse Jacobian method with quaternion-based orientation error"""
        target_pos = target_transform[:3, 3]
        target_quat = R.from_matrix(target_transform[:3, :3]).as_quat()
        
        # Try multiple initial guesses
        initial_guesses = [
            [0.0] * len(self.joint_names),  # Home position
            np.random.uniform(-1, 1, len(self.joint_names)),  # Random
            [(l + u) / 2 for l, u in self.joint_limits],  # Mid-range
        ]
        
        for initial_guess in initial_guesses:
            q = np.array(initial_guess)
            
            for _ in range(max_iter):
                T = self.forward_kinematics(q)
                pos, quat = self.get_pose(T)
                
                # Position error (same as before)
                pos_err = target_pos - pos
                
                # Quaternion-based orientation error using scipy
                # Compute rotation difference and convert to axis-angle
                R_target = R.from_quat(target_quat)
                R_current = R.from_quat(quat)
                R_error = R_target * R_current.inv()
                quat_err = R_error.as_rotvec()
                
                # Combined error
                error = np.concatenate([pos_err, 0.5 * quat_err])
                
                if np.linalg.norm(error) < 1e-4 and self.is_valid(q):
                    return q.tolist(), True
                
                J = self.compute_jacobian(q)
                J[3:] *= 0.5  # Weight orientation
                J_pinv = J.T @ np.linalg.inv(J @ J.T + 0.01 * np.eye(6))
                
                q += 0.1 * J_pinv @ error
                q = np.clip(q, [l for l, u in self.joint_limits], [u for l, u in self.joint_limits])
        
        # If no initial guess worked, return the last attempt
        return q.tolist(), False
    
    def interpolate_path(self, q_start: List[float], q_end: List[float], n: int = 20) -> List[List[float]]:
        """Linear interpolation with collision checking"""
        path = []
        for i in range(n + 1):
            alpha = i / n
            q = [(1-alpha)*s + alpha*e for s, e in zip(q_start, q_end)]
            if self.is_valid(q):
                path.append(q)
        return path if path else [q_start, q_end]


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
    
