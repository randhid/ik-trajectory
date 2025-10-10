# Kinematics TakeHome

### Take-Home Interview Question

Implement two functions for robot motion planning.

### Part 1 – Kinematic Trajectory

```python
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

```

**Task**

Compute inverse kinematics for the target pose, choose the configuration closest to the current one, and interpolate intermediate joint values to produce a minimal-displacement, continuous trajectory.

Assume a 6-DOF serial manipulator. All third party libraries should be standard cross-platform libraries like scikit-learn or open3d and included in a `requirements.txt`.

---

### Part 2 – Time-Parameterized Trajectory

```python
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

```

**Task**

Assign times between waypoints using a trapezoidal or S-curve velocity profile so all joints respect the given limits.

Return absolute times starting at 0 s.