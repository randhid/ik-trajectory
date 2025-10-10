import os
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Tuple


class URDFKinematicsSolver:
    """
    A simple kinematics solver using URDF and inverse Jacobian method.
    Uses urdfpy for URDF parsing and kinematics.
    Uses Numpy and SciPy for numerical computations. 
    """

    def __init__(self, urdf_path: str = None):
        # Initialize PyBullet in DIRECT mode (no GUI)
        self.physics_client = p.connect(p.DIRECT)
        
        # Use PyBullet's built-in xarm6 if no path specified
        if urdf_path is None or "xarm6" in urdf_path:
            import pybullet_data
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.robot_id = p.loadURDF("xarm/xarm6_robot.urdf", useFixedBase=True)
        else:
            p.setAdditionalSearchPath(os.path.dirname(urdf_path))
            self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)
        
        # Extract joint information
        num_joints = p.getNumJoints(self.robot_id)
        self.joint_names = []
        self.joint_limits = []
        self.joint_indices = []
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_type = joint_info[2]
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.joint_names.append(joint_info[1].decode('utf-8'))
                self.joint_indices.append(i)
                lower_limit = joint_info[8] if joint_info[8] != -float('inf') else -np.pi
                upper_limit = joint_info[9] if joint_info[9] != float('inf') else np.pi
                self.joint_limits.append((lower_limit, upper_limit))
        
        # Find end-effector link (last link)
        self.end_effector_index = num_joints - 1

        print(f"Loaded URDF with {len(self.joint_names)} joints: {self.joint_names}")

    def forward_kinematics(self, joint_angles: List[float]) -> np.ndarray:
        """
        Compute the forward kinematics to get the end-effector pose.
        joint_angles: List of joint angles in radians.
        Returns a 4x4 transformation matrix representing the end-effector pose.
        """
        # Set joint positions
        for i, joint_idx in enumerate(self.joint_indices):
            if i < len(joint_angles):
                p.resetJointState(self.robot_id, joint_idx, joint_angles[i])
        
        # Get end-effector pose
        link_state = p.getLinkState(self.robot_id, self.end_effector_index)
        pos = link_state[0]  # position
        orn = link_state[1]  # orientation (quaternion)
        
        # Convert to 4x4 transformation matrix
        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = R.from_quat(orn).as_matrix()
        return T
    
    def compute_jacobian(self, joint_angles: List[float]) -> np.ndarray:
        """
        Compute the Jacobian matrix at the current joint angles using numerical differentiation.
        joint_angles: List of joint angles in radians.
        Returns a 6xN Jacobian matrix, where N is the number of joints.
        """
        eps = 1e-6
        J = np.zeros((6, len(joint_angles)))
        T0 = self.forward_kinematics(joint_angles)
        pos0, quat0 = self.get_pose_from_transform(T0)
        R0 = R.from_quat(quat0).as_matrix()
        
        for i in range(len(joint_angles)):
            q_pert = joint_angles.copy()
            q_pert[i] += eps
            T_pert = self.forward_kinematics(q_pert)
            pos_pert, quat_pert = self.get_pose_from_transform(T_pert)
            R_pert = R.from_quat(quat_pert).as_matrix()
            
            # Linear velocity
            J[:3, i] = (pos_pert - pos0) / eps
            
            # Angular velocity (using rotation matrix difference)
            omega_skew = (R_pert - R0) @ R0.T / eps
            J[3:, i] = [omega_skew[2,1], omega_skew[0,2], omega_skew[1,0]]
        
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
        
        # Set joint positions for collision check
        for i, joint_idx in enumerate(self.joint_indices):
            if i < len(q):
                p.resetJointState(self.robot_id, joint_idx, q[i])
        
        # Simple collision check using PyBullet
        contacts = p.getContactPoints(self.robot_id, self.robot_id)
        return len(contacts) == 0  # No self-collision
    
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
                pos, quat = self.get_pose_from_transform(T)
                
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
    
    def __del__(self):
        """Clean up PyBullet connection"""
        try:
            p.disconnect(self.physics_client)
        except:
            pass

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
    solver = URDFKinematicsSolver()  # Use built-in xarm6
    
    # Joint name mapping
    mapping = {"base": "joint1", "shoulder": "joint2", "elbow": "joint3", 
               "wrist_pan": "joint4", "wrist_tilt": "joint5", "wrist_roll": "joint6"}
    reverse_map = {v: k for k, v in mapping.items()}
    
    # Convert to solver format
    current_q = [current_joints.get(reverse_map.get(name, name), 0.0) 
                 for name in solver.joint_names]
    
    # Convert desired pose to transformation matrix
    x, y, z, roll, pitch, yaw = desired_eef_pose
    target_transform = np.eye(4)
    target_transform[:3, 3] = [x, y, z]
    target_transform[:3, :3] = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    
    # Solve inverse kinematics (handles multiple initial guesses internally)
    target_q, success = solver.inverse_kinematics(target_transform)
    if not success:
        raise ValueError("Inverse kinematics failed to find a valid solution.")
    
    # Generate path and convert back to dict format
    path = solver.interpolate_path(current_q, target_q)
    return [{reverse_map.get(name, name): q[i] 
             for i, name in enumerate(solver.joint_names)} 
            for q in path]

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
    current_joints = {"base": 0.0, "shoulder": 0.0, "elbow": 1.578, 
                     "wrist_pan": 0.0, "wrist_tilt": -1.57, "wrist_roll": 0.0}
   
    desired_pose = (0.5, 0.3, 0.6, 0.0, 0.0, 0.0)
   
    print(f"Current: {current_joints}")
    print(f"Target:  {desired_pose}")
    
    # Plan trajectory
    waypoints = plan_trajectory(current_joints, desired_pose)
    print(f"Generated {len(waypoints)} waypoints")

if __name__ == "__main__":
    main()
    
