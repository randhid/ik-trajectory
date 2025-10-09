

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
    
