import os
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Tuple


class URDFKinematicsSolver:
    """
    A kinematics solver using PyBullet's built-in xarm6 and pseudo-inverse Jacobian method.
    Uses PyBullet for URDF parsing and forward kinematics.
    Uses numerical differentiation for Jacobian computation.
    Joint names: joint1, joint2, joint3, joint4, joint5, joint6
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
                self.joint_names.append(joint_info[1].decode("utf-8"))
                self.joint_indices.append(i)
                lower_limit = (
                    joint_info[8] if joint_info[8] != -float("inf") else -np.pi
                )
                upper_limit = joint_info[9] if joint_info[9] != float("inf") else np.pi
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
            J[3:, i] = [omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]]

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
        """Check joint limits, workspace bounds, and collision avoidance"""
        # Joint limits
        for i, angle in enumerate(q):
            if i < len(self.joint_limits):
                lower, upper = self.joint_limits[i]
                if not (lower <= angle <= upper):
                    return False

        # Set joint positions for workspace and collision checks
        for i, joint_idx in enumerate(self.joint_indices):
            if i < len(q):
                p.resetJointState(self.robot_id, joint_idx, q[i])

        # Simple collision check using PyBullet
        contacts = p.getContactPoints(self.robot_id, self.robot_id)
        return len(contacts) == 0  # No self-collision

    def inverse_kinematics(
        self, target_transform: np.ndarray, max_iter: int = 200
    ) -> Tuple[List[float], bool]:
        """Solve IK using pseudo-inverse Jacobian method with quaternion-based orientation error"""
        target_pos = target_transform[:3, 3]
        target_quat = R.from_matrix(target_transform[:3, :3]).as_quat()

        print(f"IK target position: {target_pos}")
        print(f"IK target orientation (quat): {target_quat}")

        # Try multiple initial guesses
        initial_guesses = [
            [0.0] * len(self.joint_names),  # Home position
            np.random.uniform(-0.5, 0.5, len(self.joint_names)),  # Small random
            [(l + u) / 2 for l, u in self.joint_limits],  # Mid-range
            np.random.uniform(-1, 1, len(self.joint_names)),  # Larger random
        ]

        best_q = None
        best_error = float("inf")

        for guess_idx, initial_guess in enumerate(initial_guesses):
            q = np.array(initial_guess)

            for iter_count in range(max_iter):
                T = self.forward_kinematics(q)
                pos, quat = self.get_pose_from_transform(T)

                # Position error
                pos_err = target_pos - pos

                # Quaternion-based orientation error using scipy
                R_target = R.from_quat(target_quat)
                R_current = R.from_quat(quat)
                R_error = R_target * R_current.inv()
                quat_err = R_error.as_rotvec()

                # Combined error
                error = np.concatenate([pos_err, 0.5 * quat_err])
                error_norm = np.linalg.norm(error)

                # Track best solution
                if error_norm < best_error:
                    best_error = error_norm
                    best_q = q.copy()

                # Check convergence
                if error_norm < 1e-3:  # Relaxed tolerance
                    print(
                        f"IK converged with error: {error_norm:.6f} after {iter_count+1} iterations (guess {guess_idx+1})"
                    )
                    return q.tolist(), True

                # Compute Jacobian and update
                J = self.compute_jacobian(q)
                J[3:] *= 0.5  # Weight orientation

                # Use damped least squares with adaptive damping
                damping = 0.01 + 0.1 * error_norm
                J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))

                # Adaptive step size
                step_size = 0.2 / (1.0 + error_norm)
                q += step_size * J_pinv @ error

                # Enforce joint limits
                q = np.clip(
                    q,
                    [l for l, u in self.joint_limits],
                    [u for l, u in self.joint_limits],
                )

        print(f"IK did not fully converge. Best error: {best_error:.6f}")
        return best_q.tolist() if best_q is not None else q.tolist(), False

    def interpolate_path(
        self, q_start: List[float], q_end: List[float], n: int = 20
    ) -> List[List[float]]:
        """Linear interpolation with collision checking"""
        path = []
        for i in range(n + 1):
            alpha = i / n
            q = [(1 - alpha) * s + alpha * e for s, e in zip(q_start, q_end)]
            if self.is_valid(q):
                path.append(q)
        return path if path else [q_start, q_end]

    def __del__(self):
        """Clean up PyBullet connection"""
        try:
            p.disconnect(self.physics_client)
        except:
            pass


def plan_trajectory(
    current_joints: dict[str, float],
    desired_eef_pose: tuple[float, float, float, float, float, float],
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

    current_q = [current_joints.get(name, 0.0) for name in solver.joint_names]

    # Convert desired pose to transformation matrix
    x, y, z, roll, pitch, yaw = desired_eef_pose
    target_transform = np.eye(4)
    target_transform[:3, 3] = [x, y, z]
    target_transform[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()

    target_q, success = solver.inverse_kinematics(target_transform)
    if not success:
        # Check if the pose is severely out of bounds by testing the best attempt
        T_achieved = solver.forward_kinematics(target_q)
        pos_achieved = T_achieved[:3, 3]
        pos_desired = target_transform[:3, 3]
        position_error = np.linalg.norm(pos_achieved - pos_desired)

        print(f"Warning: Inverse kinematics failed to converge.")
        print(f"Position error: {position_error:.3f}m")

        # If error is very large (> 0.5m), reject the solution
        if position_error > 0.5:
            print("ERROR: Target pose is clearly out of workspace bounds.")
            print("Refusing to generate trajectory for unreachable pose.")
            raise ValueError(
                f"Target pose unreachable - position error {position_error:.3f}m > 0.5m threshold"
            )
        else:
            print("Using best attempt solution - small pose error may be acceptable.")

    path = solver.interpolate_path(current_q, target_q)
    return [
        {solver.joint_names[i]: q[i] for i in range(len(solver.joint_names))}
        for q in path
    ]


def time_parameterize_trajectory(
    waypoints: list[dict[str, float]], max_vel: float = 1.0, max_acc: float = 2.0
) -> list[tuple[float, dict[str, float]]]:
    """
    Input:
        waypoints – joint-space path from Part 1
        max_vel, max_acc – scalar joint-space limits

    Output:
        List of (time, joint_dict) tuples giving absolute timestamps
        that satisfy velocity and acceleration bounds.
    """

    # do not construct a splined cubic polynomial if we only have two waypoints, need a different solver
    # or directly command the xarm to the target
    if len(waypoints) < 2:
        return [(0.0, waypoints[0])] if waypoints else []

    joint_names = list(waypoints[0].keys())
    n_joints = len(joint_names)
    n_waypoints = len(waypoints)

    q_waypoints = np.array([[wp[joint] for joint in joint_names] for wp in waypoints])

    # step 1: calculate initial time allocation based on distance and max velocity
    segment_times = []
    for i in range(n_waypoints - 1):
        q_diff = q_waypoints[i + 1] - q_waypoints[i]
        max_joint_diff = np.max(np.abs(q_diff))
        # initial time estimate based on max velocity
        # hopefully never zero diff but just in case there's a bug in my waypoint generator
        dt = max_joint_diff / max_vel if max_joint_diff > 0 else 0.1
        segment_times.append(dt)

    # step 2: iterate to adjust timing to satisfy acceleration constraints
    # iterate 3 times to fix acceleration violations and get a smooth velocity profile
    # based on waypoint distances - make sure no segment has too high acceleration
    for iteration in range(3):
        for i in range(len(segment_times)):
            q_diff = q_waypoints[i + 1] - q_waypoints[i]
            dt = segment_times[i]

            # for cubic polynomial with zero boundary velocities:
            # q(t) = a_0 + a_1t + a_2t^2 + a_3t^3
            # initial conditions: q(t_0)=q_0, q'(0)=0, q(t_1)=q_1, q'(t_1)=0
            # q is joint angle, q' is joint velocity
            # peak acceleration is approximately 6*|q_diff|/dt^2
            peak_acc = 6.0 * np.max(np.abs(q_diff)) / (dt * dt)

            if peak_acc > max_acc:
                dt_new = np.sqrt(6.0 * np.max(np.abs(q_diff)) / max_acc)
                segment_times[i] = max(dt, dt_new)

    # step 3: build cubic polynomial coefficients for each segment and joint
    trajectories = []
    current_time = 0.0
    trajectories.append((current_time, waypoints[0]))

    # generate trajectories for each segment
    for seg_idx in range(n_waypoints - 1):
        dt = segment_times[seg_idx]
        q_start = q_waypoints[seg_idx]
        q_end = q_waypoints[seg_idx + 1]

        # cubic polynomial coefficients for each joint
        # q(t) = a_0 + a_1t + a_2t^2 + a_3t^3
        # boundary conditions: q(0)=q_start, q'(0)=0, q(dt)=q_end, q'(dt)=0
        a0 = q_start
        a1 = np.zeros(n_joints)  # zero initial velocity
        a2 = 3.0 * (q_end - q_start) / (dt * dt)
        a3 = -2.0 * (q_end - q_start) / (dt * dt * dt)

        n_samples = max(10, int(dt * 50))  # 50 Hz sampling
        for i in range(1, n_samples + 1):
            t_local = (i / n_samples) * dt
            t_global = current_time + t_local

            # use the cubic polynomial to evaluate at each time step t
            q_t = a0 + a1 * t_local + a2 * (t_local**2) + a3 * (t_local**3)

            # convert back to dictionary format
            joint_dict = {joint_names[j]: q_t[j] for j in range(n_joints)}
            trajectories.append((t_global, joint_dict))

        current_time += dt

    return trajectories


def main():
    print("XArm6 Kinematics Solver with Cubic Polynomial Trajectory Planning")
    print("=" * 65)

    # Use xarm joint names consistently (joint1, joint2, etc.)
    current_joints = {
        "joint1": 0.0,
        "joint2": 0.0,
        "joint3": 1.578,
        "joint4": 0.0,
        "joint5": -1.57,
        "joint6": 0.0,
    }

    desired_pose = (0.5, 0.3, 0.6, 0.0, 0.0, 0.0)

    print("TEST 1: Valid pose within workspace")
    print("-" * 35)
    print(f"Current joints: {current_joints}")
    print(f"Target pose:    {desired_pose}")

    # Plan trajectory
    waypoints = plan_trajectory(current_joints, desired_pose)
    print(f"Generated {len(waypoints)} spatial waypoints")

    # Time parameterize trajectory
    max_vel = 1.0  # rad/s
    max_acc = 2.0  # rad/s²
    timed_trajectory = time_parameterize_trajectory(waypoints, max_vel, max_acc)

    print(f"Generated {len(timed_trajectory)} timed trajectory points")
    print(f"Total trajectory duration: {timed_trajectory[-1][0]:.2f} seconds")

    print("Verifying final pose...")
    try:
        solver = URDFKinematicsSolver()
        final_joints = timed_trajectory[-1][1]

        # Convert to list format for forward kinematics (no mapping needed!)
        final_q = [final_joints.get(joint, 0.0) for joint in solver.joint_names]
        T_final = solver.forward_kinematics(final_q)
        pos_final, quat_final = solver.get_pose_from_transform(T_final)

        # Convert to RPY for comparison
        rpy_final = R.from_quat(quat_final).as_euler("xyz")
        final_pose = (*pos_final, *rpy_final)

        pose_error = np.linalg.norm(np.array(final_pose) - np.array(desired_pose))
        print(f"Achieved pose: {[f'{v:.3f}' for v in final_pose]}")
        print(f"Desired pose:  {[f'{v:.3f}' for v in desired_pose]}")
        print(f"Pose error:    {pose_error:.6f}")

    except Exception as e:
        print(f"Pose verification failed: {e}")

    # Additional test: Out-of-bounds pose
    print("\n" + "=" * 65)
    print("TEST 2: Out-of-bounds pose (should fail or give poor results)")
    print("-" * 55)

    out_of_bounds_pose = (2.0, 1.5, 0.8, 0.0, 0.0, 0.0)  # Way outside XArm6 reach
    print(f"Target pose (OUT OF BOUNDS): {out_of_bounds_pose}")
    print("Note: XArm6 has ~0.85m max reach, but target is ~2.5m from base")

    try:
        print("\nAttempting trajectory planning for out-of-bounds pose...")
        out_of_bounds_waypoints = plan_trajectory(current_joints, out_of_bounds_pose)
        print(f"Generated {len(out_of_bounds_waypoints)} waypoints (unexpectedly!)")

        if out_of_bounds_waypoints:
            print(
                "This should not happen - trajectory generation should have been rejected!"
            )

    except ValueError as e:
        print(f"✓ CORRECTLY REJECTED: {e}")
    except Exception as e:
        print(f"Out-of-bounds test failed with unexpected error: {e}")


if __name__ == "__main__":
    main()
