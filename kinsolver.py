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
        self, target_transform: np.ndarray, max_iter: int = 200, verbose: bool = False
    ) -> Tuple[List[float], bool]:
        """Solve IK using pseudo-inverse Jacobian method with quaternion-based orientation error
        
        Args:
            target_transform: 4x4 target transformation matrix
            max_iter: Maximum number of iterations
            verbose: If True, prints detailed convergence information
        
        Returns:
            Tuple containing:
            - Joint angles (list)
            - Success flag (bool)
        """
        target_pos = target_transform[:3, 3]
        target_quat = R.from_matrix(target_transform[:3, :3]).as_quat()

        if verbose:
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
        convergence_data = []  # Store convergence info for logging
        all_solutions = []  # Store all converged solutions

        if verbose:
            print(f"Trying {len(initial_guesses)} different initial guesses...")

        for guess_idx, initial_guess in enumerate(initial_guesses):
            q = np.array(initial_guess)
            guess_convergence = []  # Track convergence for this guess

            for iter_count in range(max_iter):
                T = self.forward_kinematics(q)
                pos, quat = self.get_pose_from_transform(T)

                # Position error
                pos_err = target_pos - pos
                pos_err_norm = np.linalg.norm(pos_err)

                # Quaternion-based orientation error using scipy
                R_target = R.from_quat(target_quat)
                R_current = R.from_quat(quat)
                R_error = R_target * R_current.inv()
                quat_err = R_error.as_rotvec()
                quat_err_norm = np.linalg.norm(quat_err)

                # Combined error
                error = np.concatenate([pos_err, 0.5 * quat_err])
                error_norm = np.linalg.norm(error)

                # Store convergence data for analysis
                guess_convergence.append({
                    'iteration': iter_count,
                    'total_error': error_norm,
                    'position_error': pos_err_norm,
                    'orientation_error': quat_err_norm
                })

                # Verbose logging every 10 iterations or on convergence
                if verbose and (iter_count % 10 == 0 or error_norm < 1e-3):
                    print(f"  Guess {guess_idx+1}, Iter {iter_count+1}: "
                          f"Total={error_norm:.6f}, Pos={pos_err_norm:.6f}, Ori={quat_err_norm:.6f}")

                # Track best solution across all guesses
                if error_norm < best_error:
                    best_error = error_norm
                    best_q = q.copy()
                    # Store the best convergence data
                    convergence_data = guess_convergence.copy()

                # Check convergence - but don't return, continue with other guesses
                if error_norm < 1e-3:  # Relaxed tolerance
                    if verbose:
                        print(f"  Guess {guess_idx+1} CONVERGED with error: {error_norm:.6f} after {iter_count+1} iterations")
                        # Print convergence summary
                        if len(guess_convergence) > 1:
                            initial_err = guess_convergence[0]['total_error']
                            final_err = guess_convergence[-1]['total_error']
                            improvement = (initial_err - final_err) / initial_err * 100
                            print(f"    Error improved from {initial_err:.6f} to {final_err:.6f} ({improvement:.1f}% reduction)")
                    
                    # Store this solution but continue with other guesses
                    all_solutions.append({
                        'guess_idx': guess_idx,
                        'q': q.copy(),
                        'error': error_norm,
                        'iterations': iter_count + 1,
                        'convergence_data': guess_convergence.copy()
                    })
                    break  # This guess converged, move to next guess

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

        # Analyze all solutions
        if verbose:
            print(f"\nIK Analysis Summary:")
            print(f"  Converged solutions: {len(all_solutions)}/{len(initial_guesses)}")
            if all_solutions:
                print(f"  Best convergence: Guess {all_solutions[0]['guess_idx']+1} with error {all_solutions[0]['error']:.6f}")
        
        if all_solutions:
            # Return the best converged solution
            best_solution = min(all_solutions, key=lambda x: x['error'])
            if verbose:
                print(f"  Selected: Guess {best_solution['guess_idx']+1} (error: {best_solution['error']:.6f})")
            return best_solution['q'].tolist(), True
        else:
            # No solution converged, return best attempt
            if verbose:
                print(f"  No solutions converged. Best error: {best_error:.6f}")
                # Print detailed convergence analysis
                if convergence_data:
                    initial_err = convergence_data[0]['total_error']
                    final_err = convergence_data[-1]['total_error']
                    improvement = (initial_err - final_err) / initial_err * 100
                    print(f"    Error improved from {initial_err:.6f} to {final_err:.6f} ({improvement:.1f}% reduction)")
                    print(f"    Final position error: {convergence_data[-1]['position_error']:.6f}m")
                    print(f"    Final orientation error: {convergence_data[-1]['orientation_error']:.6f}rad")
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

    target_q, success = solver.inverse_kinematics(target_transform, verbose=True)
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

    if len(waypoints) < 2:
        return [(0.0, waypoints[0])] if waypoints else []

    joint_names = list(waypoints[0].keys())
    q_waypoints = np.array([[wp[j] for j in joint_names] for wp in waypoints])

    # initial per-segment time allocation from max joint displacement
    segment_times = []
    for i in range(len(waypoints) - 1):
        dq = q_waypoints[i + 1] - q_waypoints[i]
        max_d = np.max(np.abs(dq))
        dt = max_d / max_vel if (max_d > 0 and max_vel > 0) else 0.1
        segment_times.append(dt)

    time_stamps = [0.0]
    for dt_seg in segment_times:
        time_stamps.append(time_stamps[-1] + dt_seg)
    total_time = max(time_stamps[-1], 1e-6)

    # build cubic splines per joint
    joint_splines = {jn: CubicSpline(time_stamps, q_waypoints[:, j], bc_type=((1, 0.0), (1, 0.0)))
                     for j, jn in enumerate(joint_names)}

    # Find minimal global time scaling k >= 1 by bisection so that
    # velocities (scale 1/k) and accelerations (scale 1/k^2) are within limits.
    def feasible_with_scale(k: float) -> bool:
        T = total_time * k
        n_samples = int(max(200, min(3000, max(1, int(T * 50)))))
        test_times = np.linspace(0.0, T, n_samples)

        # when scaling time by k, new spline derivatives scale: v' = v / k, a' = a / k^2
        for t in test_times:
            # map t back to original spline parameter: t_orig = t / k
            to = t / k
            for jn in joint_names:
                s = joint_splines[jn]
                v = abs(s(to, 1)) / k
                a = abs(s(to, 2)) / (k * k)
                if max_vel > 0 and v > max_vel + 1e-12:
                    return False
                if max_acc > 0 and a > max_acc + 1e-12:
                    return False
        return True

    # quick check: if already feasible with k=1, keep it
    if feasible_with_scale(1.0):
        chosen_k = 1.0
    else:
        # find upper bound by doubling until feasible (cap to avoid runaway)
        k_low = 1.0
        k_high = 2.0
        max_k_cap = 1e3
        while k_high < max_k_cap and not feasible_with_scale(k_high):
            k_high *= 2.0

        if k_high >= max_k_cap:
            # fallback: use conservative scaling to guarantee safety
            chosen_k = k_high
        else:
            # bisection to find minimal feasible k
            for _ in range(40):
                k_mid = 0.5 * (k_low + k_high)
                if feasible_with_scale(k_mid):
                    k_high = k_mid
                else:
                    k_low = k_mid
            chosen_k = k_high

    # apply chosen scaling to timestamps and rebuild splines
    time_stamps = [t * chosen_k for t in time_stamps]
    total_time *= chosen_k
    joint_splines = {jn: CubicSpline(time_stamps, q_waypoints[:, j], bc_type=((1, 0.0), (1, 0.0)))
                     for j, jn in enumerate(joint_names)}

    # Per-segment compression removed: keep global time-scaling result (chosen_k) to ensure prompt runtime.

    # sample at fixed dt
    dt = 0.05
    trajectory_times = np.arange(0.0, total_time + dt, dt)
    trajectory_points = []
    for t in trajectory_times:
        joint_positions = {jn: float(joint_splines[jn](t)) for jn in joint_names}
        trajectory_points.append((t, joint_positions))

    return trajectory_points