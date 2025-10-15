import kinsolver
import pybullet as p
import time

def simulate_trajectory(
    solver: kinsolver.URDFKinematicsSolver,
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
