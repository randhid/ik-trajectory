import os
import numpy as np
import pybullet as p
from typing import List, Tuple
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R


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

        # Use local xarm folder if no path specified
        if urdf_path is None or "xarm6" in urdf_path:
            # Set search path to current directory to find local xarm folder
            p.setAdditionalSearchPath(".")
            print("Using local XArm6 model from ./xarm/")
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

    Creates smooth continuous motion using cubic splines with intermediate waypoints.
    """

    if len(waypoints) < 2:
        return [(0.0, waypoints[0])] if waypoints else []

    joint_names = list(waypoints[0].keys())
    n_waypoints = len(waypoints)

    q_waypoints = np.array([[wp[joint] for joint in joint_names] for wp in waypoints])

    # step 1: Calculate time allocation based on velocity constraints between waypoints
    segment_times = []
    for i in range(n_waypoints - 1):
        q_diff = q_waypoints[i + 1] - q_waypoints[i]
        max_joint_diff = np.max(np.abs(q_diff))
        # Use more conservative velocity for smoother motion
        dt = max_joint_diff / (max_vel * 0.8) if max_joint_diff > 0 else 0.1
        segment_times.append(dt)

    # step 2: Build cumulative time stamps for original waypoints
    time_stamps = [0.0]
    for dt in segment_times:
        time_stamps.append(time_stamps[-1] + dt)

    total_time = time_stamps[-1]

    # step 3: Create smooth cubic splines directly between original waypoints
    # Use clamped boundary conditions for smooth start/stop motion
    joint_splines = {}
    for j, joint_name in enumerate(joint_names):
        joint_values = q_waypoints[:, j]

        # use clamped boundary conditions (zero velocity at start and end)
        spline = CubicSpline(
            time_stamps, joint_values, bc_type=((1, 0.0), (1, 0.0))
        )  # Zero first derivative at both ends
        joint_splines[joint_name] = spline

    # step 4: Check and adjust for velocity/acceleration constraints
    test_times = np.linspace(0, total_time, max(100, int(total_time * 50)))
    max_vel_violation = 0.0
    max_acc_violation = 0.0

    for t in test_times[1:-1]:  # Skip endpoints
        for joint_name in joint_names:
            spline = joint_splines[joint_name]

            # Check velocity constraint
            velocity = abs(spline(t, 1))  # First derivative
            if velocity > max_vel:
                max_vel_violation = max(max_vel_violation, velocity / max_vel)

            # Check acceleration constraint
            acceleration = abs(spline(t, 2))  # Second derivative
            if acceleration > max_acc:
                max_acc_violation = max(max_acc_violation, acceleration / max_acc)

    # scale time if constraints are violated
    max_violation = max(max_vel_violation, max_acc_violation)
    if max_violation > 1.0:
        scale_factor = max_violation * 1.15  # 15% safety margin
        time_stamps = [t * scale_factor for t in time_stamps]
        total_time *= scale_factor

        # recreate splines with scaled time
        for j, joint_name in enumerate(joint_names):
            joint_values = q_waypoints[:, j]  # Use original waypoints
            spline = CubicSpline(
                time_stamps, joint_values, bc_type=((1, 0.0), (1, 0.0))
            )
            joint_splines[joint_name] = spline

    # step 5: Sample the smooth splines at regular intervals
    dt = 0.05  # 50ms timestep for smooth motion
    trajectory_times = np.arange(0.0, total_time + dt, dt)

    trajectory_points = []
    for t in trajectory_times:
        joint_positions = {}
        for joint_name in joint_names:
            # Evaluate the smooth spline at time t
            joint_positions[joint_name] = float(joint_splines[joint_name](t))

        trajectory_points.append((t, joint_positions))

    return trajectory_points


def simulate_trajectory(
    solver,
    timed_trajectory: list[tuple[float, dict[str, float]]],
    real_time: bool = True,
    speed_factor: float = 1.0,
):
    """Simulate the robot moving through the trajectory in PyBullet's GUI"""
    if not timed_trajectory:
        print("No trajectory data to simulate")
        return

    print(f"\nStarting trajectory simulation...")
    print(f"Duration: {timed_trajectory[-1][0]:.2f} seconds")
    print(f"Trajectory points: {len(timed_trajectory)}")
    print("Close the PyBullet window to end simulation")

    # Create a new connection with GUI for visualization
    sim_client = p.connect(p.GUI)
    if sim_client < 0:
        print("Failed to connect to PyBullet GUI")
        return

    try:
        # Set up the simulation environment
        p.setGravity(0, 0, -9.81, physicsClientId=sim_client)
        p.setRealTimeSimulation(
            0, physicsClientId=sim_client
        )  # Step simulation manually

        # Set search path for local xarm folder and load ground plane
        p.setAdditionalSearchPath(".", physicsClientId=sim_client)
        import pybullet_data

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=sim_client
        )
        plane_id = p.loadURDF("plane.urdf", physicsClientId=sim_client)

        # Load the robot from local xarm folder
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        robot_id = p.loadURDF(
            "xarm/xarm6_robot.urdf",
            start_pos,
            start_orientation,
            physicsClientId=sim_client,
        )

        # Get joint information
        joint_indices = []
        for i in range(p.getNumJoints(robot_id, physicsClientId=sim_client)):
            joint_info = p.getJointInfo(robot_id, i, physicsClientId=sim_client)
            if joint_info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
                joint_indices.append(i)

        print(f"Found {len(joint_indices)} controllable joints")

        # Set camera to get a good view of the robot
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0.3, 0.3, 0.5],
            physicsClientId=sim_client,
        )

        # Add some visual markers for trajectory points
        trajectory_markers = []
        sample_indices = range(
            0, len(timed_trajectory), max(1, len(timed_trajectory) // 10)
        )

        for idx in sample_indices:
            time, joints = timed_trajectory[idx]
            joint_values = [joints.get(joint, 0.0) for joint in solver.joint_names]

            # Set joint positions to get end-effector position
            for i, joint_idx in enumerate(joint_indices):
                if i < len(joint_values):
                    p.resetJointState(
                        robot_id, joint_idx, joint_values[i], physicsClientId=sim_client
                    )

            # Get end-effector position
            if len(joint_indices) > 0:
                link_state = p.getLinkState(
                    robot_id, joint_indices[-1], physicsClientId=sim_client
                )
                ee_pos = link_state[0]

                # Add a small sphere marker at this trajectory point
                marker_color = [
                    1.0 - idx / len(timed_trajectory),
                    0.0,
                    idx / len(timed_trajectory),
                    0.7,
                ]
                marker_id = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=0.02,
                    rgbaColor=marker_color,
                    physicsClientId=sim_client,
                )
                marker_body = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=marker_id,
                    basePosition=ee_pos,
                    physicsClientId=sim_client,
                )
                trajectory_markers.append(marker_body)

        # Simulate the trajectory
        start_time = time.time()

        for trajectory_point in timed_trajectory:
            # Extract time and joint values from trajectory point
            trajectory_time = trajectory_point[0]  # First element is time
            joints = trajectory_point[1]  # Second element is joint dict

            joint_values = [joints.get(joint, 0.0) for joint in solver.joint_names]

            # Set target joint positions
            for i, joint_idx in enumerate(joint_indices):
                if i < len(joint_values):
                    p.setJointMotorControl2(
                        robot_id,
                        joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=joint_values[i],
                        maxVelocity=2.0,  # rad/s
                        force=100,
                        physicsClientId=sim_client,
                    )

            # Step the simulation
            p.stepSimulation(physicsClientId=sim_client)

            # Control timing
            if real_time:
                target_time = trajectory_time / speed_factor
                elapsed = time.time() - start_time
                sleep_time = target_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        print("\nTrajectory simulation completed!")
        print("The robot will remain in the final position.")
        print("Close the PyBullet window when you're done viewing.")

        # Keep the simulation running until window is closed
        try:
            while True:
                p.stepSimulation(physicsClientId=sim_client)
                time.sleep(1 / 240)  # 240 Hz simulation
        except KeyboardInterrupt:
            print("Simulation interrupted by user")

    except Exception as e:
        print(f"Simulation error: {e}")
    finally:
        p.disconnect(physicsClientId=sim_client)
