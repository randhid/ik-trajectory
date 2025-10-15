import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import csv


def save_trajectory_csv(timed_trajectory, filename="traj.csv"):
    """Save trajectory to CSV file with time and joint columns"""
    if not timed_trajectory:
        print("No trajectory to save")
        return

    # Get joint names from first trajectory point
    joint_names = list(timed_trajectory[0][1].keys())

    with open(filename, "w", newline="") as csvfile:
        # Create header: time, joint1, joint2, etc.
        fieldnames = ["time"] + joint_names
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write trajectory data
        for time, joints in timed_trajectory:
            row = {"time": time}
            row.update(joints)
            writer.writerow(row)

    print(f"Trajectory saved to {filename}")


def plot_positions_and_velocities(timed_trajectory):
    """Generate position and velocity plots for each joint"""
    if not timed_trajectory:
        print("No trajectory to plot")
        return

    # Extract data
    times = [t for t, _ in timed_trajectory]
    joint_names = list(timed_trajectory[0][1].keys())

    # Create position data matrix
    positions = []
    for _, joints in timed_trajectory:
        positions.append([joints[joint] for joint in joint_names])
    positions = np.array(positions)

    # Calculate velocities using numerical differentiation
    dt = np.diff(times)
    velocities = np.zeros_like(positions)

    # Forward difference for velocities
    for i in range(len(positions) - 1):
        if dt[i] > 0:
            velocities[i] = (positions[i + 1] - positions[i]) / dt[i]
    # Last point uses backward difference
    if len(positions) > 1:
        velocities[-1] = velocities[-2]

    # Create position plots
    fig_pos, axes_pos = plt.subplots(2, 3, figsize=(15, 10))
    fig_pos.suptitle("Joint Positions vs Time", fontsize=16)
    axes_pos = axes_pos.flatten()

    for i, joint_name in enumerate(joint_names):
        if i < len(axes_pos):
            axes_pos[i].plot(times, positions[:, i], "b-", linewidth=2)
            axes_pos[i].set_xlabel("Time (s)")
            axes_pos[i].set_ylabel(f"{joint_name} (rad)")
            axes_pos[i].set_title(f"{joint_name} Position")
            axes_pos[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(joint_names), len(axes_pos)):
        axes_pos[i].axis("off")

    plt.tight_layout()
    plt.savefig("pos.png", dpi=300, bbox_inches="tight")
    print("Position plots saved to pos.png")

    # Create velocity plots
    fig_vel, axes_vel = plt.subplots(2, 3, figsize=(15, 10))
    fig_vel.suptitle("Joint Velocities vs Time", fontsize=16)
    axes_vel = axes_vel.flatten()

    for i, joint_name in enumerate(joint_names):
        if i < len(axes_vel):
            axes_vel[i].plot(times, velocities[:, i], "r-", linewidth=2)
            axes_vel[i].set_xlabel("Time (s)")
            axes_vel[i].set_ylabel(f"{joint_name} (rad/s)")
            axes_vel[i].set_title(f"{joint_name} Velocity")
            axes_vel[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(joint_names), len(axes_vel)):
        axes_vel[i].axis("off")

    plt.tight_layout()
    plt.savefig("vel.png", dpi=300, bbox_inches="tight")
    print("Velocity plots saved to vel.png")


def plot_combined_trajectory(
    solver,
    timed_trajectory: list[tuple[float, dict[str, float]]],
    title: str = "XArm6 Trajectory Analysis",
):
    """Plot end-effector 3D trajectory and joint angles in a single figure"""
    if not timed_trajectory:
        print("No trajectory data to plot")
        return

    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(18, 8))

    # Left subplot: 3D end-effector trajectory
    ax_3d = fig.add_subplot(121, projection="3d")

    # Extract end-effector positions throughout trajectory
    times = [t for t, _ in timed_trajectory]
    ee_positions = []

    for time, joints in timed_trajectory:
        joint_values = [joints.get(joint, 0.0) for joint in solver.joint_names]

        # Set joint positions in PyBullet to get forward kinematics
        for i, joint_idx in enumerate(solver.joint_indices):
            if i < len(joint_values):
                p.resetJointState(solver.robot_id, joint_idx, joint_values[i])

        # Get end-effector position (last link)
        end_effector_idx = len(solver.joint_indices) - 1
        if end_effector_idx < p.getNumJoints(solver.robot_id):
            link_state = p.getLinkState(solver.robot_id, end_effector_idx)
            ee_positions.append(link_state[0])

    ee_positions = np.array(ee_positions)

    # Create color gradient from blue to red based on time progression
    colors_3d = plt.cm.viridis(np.linspace(0, 1, len(timed_trajectory)))

    # Plot 3D trajectory as scatter points with color gradient
    ax_3d.scatter(
        ee_positions[:, 0],
        ee_positions[:, 1],
        ee_positions[:, 2],
        c=colors_3d,
        s=60,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5,
    )

    # Mark start and end points
    ax_3d.scatter(
        ee_positions[0, 0],
        ee_positions[0, 1],
        ee_positions[0, 2],
        c="green",
        s=150,
        marker="o",
        label="Start",
        edgecolors="black",
        linewidth=2,
    )
    ax_3d.scatter(
        ee_positions[-1, 0],
        ee_positions[-1, 1],
        ee_positions[-1, 2],
        c="red",
        s=150,
        marker="s",
        label="End",
        edgecolors="black",
        linewidth=2,
    )

    ax_3d.set_xlabel("X (m)", fontsize=12)
    ax_3d.set_ylabel("Y (m)", fontsize=12)
    ax_3d.set_zlabel("Z (m)", fontsize=12)
    ax_3d.set_title("End-Effector 3D Trajectory", fontsize=14, fontweight="bold")
    ax_3d.legend(fontsize=10)
    ax_3d.grid(True, alpha=0.3)

    # Right subplot: Joint angle time graphs
    ax_joints = fig.add_subplot(122)

    joint_names = list(timed_trajectory[0][1].keys())
    colors_joints = plt.cm.tab10(np.arange(len(joint_names)))

    for i, joint_name in enumerate(joint_names):
        joint_values = [joints[joint_name] for _, joints in timed_trajectory]
        ax_joints.scatter(
            times,
            joint_values,
            label=joint_name,
            color=colors_joints[i],
            s=20,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.3,
        )

    ax_joints.set_xlabel("Time (s)", fontsize=12)
    ax_joints.set_ylabel("Joint Angle (rad)", fontsize=12)
    ax_joints.set_title("Joint Angles vs Time", fontsize=14, fontweight="bold")
    ax_joints.grid(True, alpha=0.3)
    ax_joints.legend(fontsize=10, loc="best")

    # Add minor ticks for better readability
    ax_joints.minorticks_on()
    ax_joints.grid(which="minor", alpha=0.15)

    plt.tight_layout()
    plt.suptitle(title, y=0.98, fontsize=16, fontweight="bold")
    plt.savefig("combined_trajectory.png", dpi=300, bbox_inches="tight")
    print("Combined trajectory plot saved to combined_trajectory.png")
    plt.show()
