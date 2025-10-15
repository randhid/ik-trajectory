import kinsolver
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from plot import (
    plot_positions_and_velocities,
    save_trajectory_csv,
    plot_combined_trajectory,
)


def main():
    print("XArm6 Trajectory Generation and Analysis")
    print("=" * 40)

    # Define start and target configurations
    current_joints = {
        "joint1": 0.0,
        "joint2": 0.0,
        "joint3": 1.578,
        "joint4": 0.0,
        "joint5": -1.57,
        "joint6": 0.0,
    }

    desired_pose = (0.5, 0.3, 0.6, 0.0, 0.0, 0.0)

    print(f"Start joints: {current_joints}")
    print(f"Target pose:  {desired_pose}")
    print("\nStep 1: Planning trajectory...")
    try:
        waypoints = kinsolver.plan_trajectory(current_joints, desired_pose)
        print(f"✓ Generated {len(waypoints)} spatial waypoints")
    except Exception as e:
        print(f"✗ Trajectory planning failed: {e}")
        return

    print("\nStep 2: Time parameterizing trajectory...")
    max_vel = 1.0  # rad/s
    max_acc = 2.0  # rad/s²

    try:
        timed_trajectory = kinsolver.time_parameterize_trajectory(
            waypoints, max_vel, max_acc
        )
        print(f"✓ Generated {len(timed_trajectory)} timed trajectory points")
        print(f"✓ Total duration: {timed_trajectory[-1][0]:.2f} seconds")
    except Exception as e:
        print(f"✗ Time parameterization failed: {e}")
        return

    print("\nStep 3: Saving trajectory to CSV...")
    try:
        save_trajectory_csv(timed_trajectory, "traj.csv")
    except Exception as e:
        print(f"✗ CSV saving failed: {e}")

    print("\nStep 4: Generating position and velocity plots...")
    try:
        plot_positions_and_velocities(timed_trajectory)
    except Exception as e:
        print(f"✗ Plotting failed: {e}")

    print("\nStep 5: Generating combined trajectory plot...")
    try:
        solver = kinsolver.URDFKinematicsSolver()
        plot_combined_trajectory(solver, timed_trajectory)
    except Exception as e:
        print(f"✗ Combined plotting failed: {e}")

    print("\nStep 6: Verifying final pose...")
    try:
        solver = kinsolver.URDFKinematicsSolver()
        final_joints = timed_trajectory[-1][1]
        final_q = [final_joints.get(joint, 0.0) for joint in solver.joint_names]
        T_final = solver.forward_kinematics(final_q)
        pos_final, quat_final = solver.get_pose_from_transform(T_final)

        rpy_final = R.from_quat(quat_final).as_euler("xyz")
        final_pose = (*pos_final, *rpy_final)

        pose_error = np.linalg.norm(np.array(final_pose) - np.array(desired_pose))
        print(f"✓ Achieved pose: {[f'{v:.3f}' for v in final_pose]}")
        print(f"✓ Desired pose:  {[f'{v:.3f}' for v in desired_pose]}")
        print(f"✓ Pose error:    {pose_error:.6f}")

    except Exception as e:
        print(f"✗ Pose verification failed: {e}")

    print(f"\n{'='*40}")
    print("Trajectory generation completed successfully!")
    print("Generated files:")
    print("  - traj.csv (trajectory data)")
    print("  - pos.png (position plots)")
    print("  - vel.png (velocity plots)")
    print("  - combined_trajectory.png (joint angles + 3D trajectory)")


if __name__ == "__main__":
    main()
