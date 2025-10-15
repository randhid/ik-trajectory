import kinsolver


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
    
    # Plot the trajectory
    print("\nGenerating combined trajectory plot...")
    try:
        # Create a new solver for plotting (avoid connection issues)
        plot_solver = URDFKinematicsSolver()
        plot_combined_trajectory(plot_solver, timed_trajectory, "XArm6 Trajectory Analysis")
        
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    # Ask user if they want to run the simulation
    print("\nTrajectory planning and visualization completed!")
    print("\nWould you like to see the robot moving in 3D simulation?")
    user_input = input("Enter 'y' or 'yes' to start simulation (or any other key to skip): ").lower().strip()
    
    if user_input in ['y', 'yes']:
        print("\nStarting 3D simulation...")
        try:
            # Create a new solver for simulation
            sim_solver = URDFKinematicsSolver()
            simulate_trajectory(sim_solver, timed_trajectory, real_time=True, speed_factor=0.5)
        except Exception as e:
            print(f"Simulation failed: {e}")
    else:
        print("Simulation skipped.")


if __name__ == "__main__":
    main()
