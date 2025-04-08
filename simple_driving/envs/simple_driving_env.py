import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
import math
import os
import time # Optional: for debugging waits

class SimpleDrivingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'tp_camera', 'fp_camera'], 'render_fps': 30}

    def __init__(self, apply_api_compatibility=False, renders=False, isDiscrete=True, render_mode='tp_camera', max_steps=1000):
        super(SimpleDrivingEnv, self).__init__()

        # Ensure compatibility flag matches if using gymnasium internally
        self.apply_api_compatibility = apply_api_compatibility

        # Simulation parameters
        self.renders = renders
        self.render_mode = render_mode
        self.isDiscrete = isDiscrete
        self.max_steps = max_steps
        self.step_counter = 0

        # Pybullet setup
        # Decide connection mode based on rendering request
        if self.renders or self.render_mode == 'human':
             self._p = p
             self.physicsClient = self._p.connect(self._p.GUI)
             # Add ability to step simulation faster/slower in GUI mode if desired
             # p.setRealTimeSimulation(0) # Option 1: manual stepping control
             p.setTimeStep(1./240.)      # Option 2: let pybullet handle timing (adjust rate here)
        else:
             self._p = p
             self.physicsClient = self._p.connect(self._p.DIRECT) # No GUI

        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Environment parameters
        self.goal_dist_threshold = 1.5  # Target distance to goal

        # Define action space
        if self.isDiscrete:
            # 0: Reverse-Left, 1: Reverse, 2: Reverse-Right
            # 3: Steer-Left (no throttle), 4: No throttle/steering, 5: Steer-Right (no throttle)
            # 6: Forward-right, 7: Forward, 8: Forward-left
            self.action_space = spaces.Discrete(9)
        else:
            # Continuous actions [throttle, steer]
            # Throttle: -1 (reverse) to 1 (forward)
            # Steer: -1 (left) to 1 (right)
            self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)

        # --- PART 4 Modification: Observation Space ---
        # Observation: [relative_goal_x, relative_goal_y, relative_obstacle_x, relative_obstacle_y]
        # Define bounds (e.g., max distance agent might be from goal/obstacle)
        # Adjust these bounds based on expected spawning range if needed
        obs_high = np.array([15, 15, 15, 15], dtype=np.float32) # Increased range slightly
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        # ---------------------------------------------

        # Initialize placeholders
        self.carId = None
        self.goalId = None
        self.obstacleId = None # Part 4: Placeholder for obstacle ID
        self.planeId = None
        self.done = False
        self.car_pos = np.zeros(3)
        self.goal_pos = np.zeros(3)
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)

        # Set gravity
        self._p.setGravity(0, 0, -9.81)

        # Camera parameters for rendering
        self._view_matrix = None
        self._proj_matrix = None
        self._set_camera() # Configure camera initial settings

        # Load plane and car (goal/obstacle loaded in reset)
        self._load_assets()

        # Seed the random number generator
        self.seed()


    def seed(self, seed=None):
        """Seeds the random number generator."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _load_assets(self):
        """Loads the ground plane and the car."""
        self.planeId = self._p.loadURDF("plane.urdf")

        # Load car URDF
        # Assumes 'simplecar.urdf' is accessible (e.g., in pybullet_data or added path)
        # If not, provide the full path or place it in a discoverable location.
        car_start_pos = [0, 0, 0.1]
        car_start_orientation = self._p.getQuaternionFromEuler([0, 0, 0])

        # Attempt to find car URDF, check common locations if necessary
        car_urdf_path = "simplecar.urdf" # Try simple name first
        if not os.path.exists(os.path.join(pybullet_data.getDataPath(), car_urdf_path)):
             # Try relative path within this package (if simplecar.urdf is placed there)
              potential_path = os.path.join(os.path.dirname(__file__), 'resources', car_urdf_path)
              if os.path.exists(potential_path):
                  car_urdf_path = potential_path
                  print(f"Using car URDF found at: {car_urdf_path}")
              else:
                   raise FileNotFoundError(f"Cannot find simplecar.urdf in pybullet_data or {potential_path}")

        self.carId = self._p.loadURDF(car_urdf_path, car_start_pos, car_start_orientation)

        # Load goal marker URDF (only needs loading once, position reset in reset())
        goal_urdf_path = os.path.join(os.path.dirname(__file__), 'resources', 'simplegoal.urdf')
        if not os.path.exists(goal_urdf_path):
             raise FileNotFoundError(f"Cannot find simplegoal.urdf at {goal_urdf_path}. Make sure it's in simple_driving/resources/")
        # Load it initially at origin, will be moved in reset
        self.goalId = self._p.loadURDF(fileName=goal_urdf_path, basePosition=[0, -5, 0.1])


    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        # Seed the RNG if a seed is provided
        if seed is not None:
            super().reset(seed=seed) # Gymnasium standard way

        # Reset step counter and done flag
        self.step_counter = 0
        self.done = False

        # Reset car state
        car_start_pos = [0, 0, 0.1]
        car_start_orientation = self._p.getQuaternionFromEuler([0, 0, 0])
        self._p.resetBasePositionAndOrientation(self.carId, car_start_pos, car_start_orientation)
        self._p.resetBaseVelocity(self.carId, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        self.car_pos = np.array(car_start_pos)

        # --- PART 4 Modification: Remove old obstacle if exists ---
        if self.obstacleId is not None:
            if self._p.isConnected():
                try:
                    # Check if body unique ID is actually still valid before removing
                    body_info = self._p.getBodyInfo(self.obstacleId)
                    if body_info: # Check if body info retrieval was successful
                        self._p.removeBody(self.obstacleId)
                        print(f"Removed previous obstacle (ID: {self.obstacleId}).")
                    else:
                         print(f"Obstacle ID {self.obstacleId} no longer valid (already removed?). Skipping removal.")
                except self._p.error as e:
                    print(f"PyBullet error removing obstacle (ID: {self.obstacleId}): {e}. Might have been removed.")
                finally:
                    self.obstacleId = None # Ensure ID is reset
            else:
                print("Physics client not connected, cannot remove obstacle.")
                self.obstacleId = None # Reset ID holder

        # --- END PART 4 Modification ---


        # Reset goal position randomly within a specified range
        # Ensure goal isn't too close to the start
        while True:
            goal_x = self.np_random.uniform(-7, 7) # Wider range maybe?
            goal_y = self.np_random.uniform(-7, 7)
            # Check distance from origin
            if np.linalg.norm([goal_x, goal_y]) > self.goal_dist_threshold + 0.5: # Ensure not too close to start
                 self.goal_pos = np.array([goal_x, goal_y, 0.1])
                 break
        self._p.resetBasePositionAndOrientation(self.goalId, self.goal_pos, car_start_orientation) # Reuse car's start orientation


        # --- PART 4 Modification: Load new obstacle ---
        # Define path to the obstacle URDF
        obstacle_urdf_path = os.path.join(os.path.dirname(__file__), 'resources', 'simplegoal.urdf') # Using goal marker as obstacle
        if not os.path.exists(obstacle_urdf_path):
             print(f"ERROR: Obstacle URDF not found at {obstacle_urdf_path}")
             # Handle error - perhaps raise it or continue without obstacle (though env expects 4D state)
             # For now, we'll assume it exists if we reach here for the obstacle task.
             self.obstacleId = None # Ensure it's None if path fails
             relative_obstacle_pos = np.array([15.0, 15.0]) # Default to far away if load fails? (risky)

        else:
             # Choose obstacle position randomly, avoiding start and goal area
             while True:
                 obs_x = self.np_random.uniform(-6, 6)
                 obs_y = self.np_random.uniform(-6, 6)
                 obstacle_pos = [obs_x, obs_y, 0.1]

                 # Check distance from car start and goal
                 dist_from_start = np.linalg.norm(np.array(obstacle_pos[:2]) - np.array(car_start_pos[:2]))
                 dist_from_goal = np.linalg.norm(np.array(obstacle_pos[:2]) - self.goal_pos[:2])

                 # Ensure obstacle is not right on top of car start or goal
                 if dist_from_start > 1.5 and dist_from_goal > 1.5:
                      break
             try:
                self.obstacleId = self._p.loadURDF(
                    fileName=obstacle_urdf_path,
                    basePosition=obstacle_pos,
                    useFixedBase=True # Make obstacle static
                )
                print(f"Loaded obstacle (ID: {self.obstacleId}) at position: {obstacle_pos[:2]}")
                # Get its relative position for initial state
                obstacle_actual_pos, _ = self._p.getBasePositionAndOrientation(self.obstacleId)
                relative_obstacle_pos = np.array(obstacle_actual_pos[:2]) - self.car_pos[:2]

             except Exception as e:
                print(f"Error loading obstacle URDF: {e}")
                self.obstacleId = None
                relative_obstacle_pos = np.array([15.0, 15.0]) # Default to far away (still risky if network expects it)

        # --- END PART 4 Modification ---


        # Calculate initial state (observation)
        relative_goal_pos = self.goal_pos[:2] - self.car_pos[:2]
        # Concatenate goal and obstacle relative positions
        self.state = np.concatenate((relative_goal_pos, relative_obstacle_pos)).astype(np.float32)

        # Check if state shape matches observation space AFTER potential concatenation
        if self.state.shape != self.observation_space.shape:
            print(f"\nWARNING: Initial state shape {self.state.shape} doesn't match observation space shape {self.observation_space.shape}. Check reset logic!")
            # Attempt to pad or truncate? Risky. Best to ensure logic always produces correct shape.
            # If obstacle load failed, state might be wrong size. Need robust handling.
            # Hacky fix: Pad with defaults if size is wrong (BAD PRACTICE)
            if self.state.shape[0] == 2 and self.observation_space.shape[0] == 4:
                 print("Padding state with default obstacle values.")
                 self.state = np.concatenate((self.state, [15.0, 15.0])).astype(np.float32)

        # Gymnasium API expects observation, info
        info = {}
        return self.state, info


    def step(self, action):
        """Applies an action and returns the next state, reward, done, info."""
        if self.done:
             # Should not happen if reset called properly, but safeguard
             print("Warning: step called after environment is done. Resetting.")
             observation, info = self.reset()
             # Return state, 0 reward, done=True? or state, 0, done=False from reset? Let's return current state and signal termination.
             # Gymnasium standard seems to favor returning the reset observation and info,
             # along with a reward=0, done=False, truncated=False signal, allowing the training loop to handle the episode boundary.
             # However, this env logic doesn't track truncated separately, so setting done=True might be clearer.
             # Let's stick to old API logic slightly here: return current (invalid) state and signal done=True.
             return self.state, 0.0, True, False, {} # (obs, rew, terminated, truncated, info)

        # Apply action based on whether the space is discrete or continuous
        if self.isDiscrete:
            # Map discrete action index to motor commands
            # Constants for steering and throttle
            max_throttle = 10.0 # Adjust magnitude as needed
            max_steer = 0.5     # Adjust magnitude as needed
            min_throttle = -5.0 # Reverse throttle

            throttle = 0
            steer = 0

            if action == 0: # Reverse-Left
                 throttle = min_throttle
                 steer = max_steer
            elif action == 1: # Reverse
                 throttle = min_throttle
                 steer = 0
            elif action == 2: # Reverse-Right
                 throttle = min_throttle
                 steer = -max_steer
            elif action == 3: # Steer-Left (no throttle)
                 throttle = 0
                 steer = max_steer
            elif action == 4: # No throttle and no steering
                 throttle = 0
                 steer = 0
            elif action == 5: # Steer-Right (no throttle)
                 throttle = 0
                 steer = -max_steer
            elif action == 6: # Forward-Right
                 throttle = max_throttle
                 steer = -max_steer
            elif action == 7: # Forward
                 throttle = max_throttle
                 steer = 0
            elif action == 8: # Forward-Left
                 throttle = max_throttle
                 steer = max_steer
            else:
                 raise ValueError("Invalid discrete action received.")

            # Apply engine force and steering angle to wheels
            # Indices for simplecar.urdf wheels might be 0, 1 (front L/R - steering) and 2, 3 (rear L/R - drive)
            # Adjust indices based on your specific car URDF structure
            steering_joints = [0, 2] # Example: Adjust if your URDF differs
            drive_joints = [1, 3]    # Example: Adjust if your URDF differs

            for joint_index in steering_joints:
                self._p.setJointMotorControl2(self.carId, joint_index, self._p.POSITION_CONTROL, targetPosition=steer)
            for joint_index in drive_joints:
                self._p.setJointMotorControl2(self.carId, joint_index, self._p.VELOCITY_CONTROL, targetVelocity=throttle, force=100) # Added force limit

        else: # Continuous action space
            throttle, steer = action
            # Scale actions to appropriate ranges if necessary (depends on model output vs desired physics values)
            scaled_throttle = throttle * 20 # Example scaling
            scaled_steer = steer * 0.6    # Example scaling

            steering_joints = [0, 2]
            drive_joints = [1, 3]
            for joint_index in steering_joints:
                self._p.setJointMotorControl2(self.carId, joint_index, self._p.POSITION_CONTROL, targetPosition=scaled_steer)
            for joint_index in drive_joints:
                self._p.setJointMotorControl2(self.carId, joint_index, self._p.VELOCITY_CONTROL, targetVelocity=scaled_throttle, force=100)


        # Step the simulation
        self._p.stepSimulation()

        # Get updated car position and orientation
        self.car_pos, car_orn = self._p.getBasePositionAndOrientation(self.carId)
        self.car_pos = np.array(self.car_pos)

        # Calculate distance to goal
        dist_to_goal = np.linalg.norm(self.car_pos[:2] - self.goal_pos[:2]) # Use only x, y distance

        # --- PART 3 Modification: Reward Calculation ---
        # Base reward is negative distance
        reward = -dist_to_goal

        # Check if goal is reached
        goal_reached = (dist_to_goal < self.goal_dist_threshold)
        if goal_reached:
             reward += 50.0 # Add bonus for reaching the goal
             print("Goal reached! Reward bonus applied.")
             self.done = True
        # --- END PART 3 Modification ---


        # --- PART 4 Modification: Collision Check ---
        collision_penalty = 0
        collided_with_obstacle = False
        if self.obstacleId is not None: # Only check if obstacle exists
            # Ensure the obstacle body ID is still valid (optional extra check)
            try:
                 _ = p.getBodyInfo(self.obstacleId) # Quick check if ID is valid
                 contact_points = self._p.getContactPoints(bodyA=self.carId, bodyB=self.obstacleId)
                 if len(contact_points) > 0:
                      collided_with_obstacle = True
                      collision_penalty = -100.0 # Large penalty for collision
                      print("Collision with obstacle detected!")
                      self.done = True # End episode on collision
             except p.error:
                 print(f"Warning: Obstacle ID {self.obstacleId} became invalid during step? Collision check skipped.")

        # Apply collision penalty if occurred
        # Make sure it overrides goal bonus if goal and collision happen ~simultaneously
        if collided_with_obstacle:
             # Override reward completely with penalty? Or add? Let's add (can result in negative reward even if close).
             # Consider: reward = collision_penalty # Or make penalty absolute
             reward += collision_penalty
        # --- END PART 4 Modification ---


        # Increment step counter and check for max steps
        self.step_counter += 1
        truncated = False # Initialize truncated flag
        if self.step_counter >= self.max_steps:
             self.done = True # Terminated due to time limit
             truncated = True # Set truncated flag according to Gymnasium standard
             print("Max steps reached.")

        # Calculate next state (observation)
        relative_goal_pos = self.goal_pos[:2] - self.car_pos[:2]

        # --- PART 4: Get relative obstacle position for state ---
        if self.obstacleId is not None:
             try:
                 obstacle_pos, _ = self._p.getBasePositionAndOrientation(self.obstacleId)
                 relative_obstacle_pos = np.array(obstacle_pos[:2]) - self.car_pos[:2]
             except p.error:
                 print(f"Warning: Failed to get obstacle {self.obstacleId} position in step. Using default.")
                 relative_obstacle_pos = np.array([15.0, 15.0]) # Default far away

             self.state = np.concatenate((relative_goal_pos, relative_obstacle_pos)).astype(np.float32)
        else:
             # Should not happen in Part 4 context, but fallback
             print("Warning: Obstacle ID is None in step. Returning 2D state (this might break learning).")
             # This will likely cause dimension mismatch error if Q-table/network expects 4D
             # Better to ensure obstacle is always present if the env is set up for it.
             # Pad state to match expected shape (Hacky)
             self.state = np.concatenate((relative_goal_pos, [15.0, 15.0])).astype(np.float32) # Pad


        # Ensure state shape matches observation space BEFORE returning
        if self.state.shape != self.observation_space.shape:
             print(f"\nERROR: Final state shape in step {self.state.shape} doesn't match observation space {self.observation_space.shape}!")
             # Try to fix or raise error? Forcing shape might hide bugs.
             # Forcing shape:
             # if self.state.shape[0] < self.observation_space.shape[0]: # Pad if too small
             #      pad_width = self.observation_space.shape[0] - self.state.shape[0]
             #      self.state = np.pad(self.state, (0, pad_width), 'constant', constant_values=15.0)
             # elif self.state.shape[0] > self.observation_space.shape[0]: # Truncate if too large
             #      self.state = self.state[:self.observation_space.shape[0]]
             # Better to raise error:
             raise ValueError(f"State shape mismatch in step(): Got {self.state.shape}, Expected {self.observation_space.shape}")

        info = {}

        # Return values following Gymnasium standard (obs, reward, terminated, truncated, info)
        terminated = self.done and not truncated # terminated is True if done AND not because of truncation
        return self.state, reward, terminated, truncated, info


    def render(self, mode='tp_camera'):
        """Renders the environment.
           'tp_camera' for third-person view.
           'fp_camera' for first-person view.
           'human' to rely on pybullet's GUI window (if connected).
        """
        if mode == 'human':
             # Assumes pybullet was connected in GUI mode.
             # No specific return needed, display is handled by the GUI connection.
              if self.physicsClient >= 0 and p.getConnectionInfo(self.physicsClient)['connectionMethod'] == p.GUI:
                   # Optional: Force a sync point if using RealTimeSimulation(0)
                   # p.syncBodyInfo()
                   time.sleep(1./240.) # Add small delay if GUI seems sluggish without real-time sim
                   return None # Or return an empty np array? Check conventions if needed.
              else:
                   # print("Warning: Render mode 'human' selected but not connected to GUI.")
                   # Fallback to default rendering? Or raise error?
                   # Fallback to 'tp_camera' array rendering for consistency.
                    mode = self.render_mode if self.render_mode in ['tp_camera', 'fp_camera'] else 'tp_camera'


        render_mode_internal = mode if mode in ['tp_camera', 'fp_camera'] else self.render_mode
        # Fallback if self.render_mode was invalid too
        if render_mode_internal not in ['tp_camera', 'fp_camera']:
             render_mode_internal = 'tp_camera'


        if self.physicsClient < 0:
            # print("Warning: Cannot render, physics client not connected.")
            # Return a dummy image maybe? Black screen?
            return np.zeros((300, 400, 3), dtype=np.uint8) # Example dummy size


        # Update camera based on selected mode
        self._set_camera(mode=render_mode_internal)

        # PyBullet rendering call
        img_arr = self._p.getCameraImage(width=400, height=300,
                                         viewMatrix=self._view_matrix,
                                         projectionMatrix=self._proj_matrix,
                                         renderer=p.ER_BULLET_HARDWARE_OPENGL # Use hardware renderer if available
                                         # renderer=p.ER_TINY_RENDERER # Fallback software renderer
                                         )

        if len(img_arr) < 4 or img_arr[2] is None:
             print("Warning: p.getCameraImage returned invalid data.")
             return np.zeros((300, 400, 3), dtype=np.uint8) # Return dummy


        # Process the image data
        rgb_array = np.array(img_arr[2], dtype=np.uint8) # Index 2 is RGB pixels
        if rgb_array is not None and len(rgb_array) > 0:
            w = img_arr[0] # Width
            h = img_arr[1] # Height
            # Reshape and remove alpha channel if present
            if rgb_array.size == w * h * 4:
                 rgb_array = rgb_array.reshape((h, w, 4))
                 rgb_array = rgb_array[:, :, :3] # Slice off alpha channel
            elif rgb_array.size == w * h * 3:
                 rgb_array = rgb_array.reshape((h, w, 3))
            else:
                print(f"Warning: Unexpected image array size {rgb_array.size} for width {w} height {h}.")
                return np.zeros((h, w, 3), dtype=np.uint8) # Return dummy

            return rgb_array
        else:
            print("Warning: RGB array from getCameraImage is None or empty.")
            return np.zeros((300, 400, 3), dtype=np.uint8) # Return dummy


    def _set_camera(self, mode='tp_camera'):
        """Configures the camera view based on the mode."""
        if self.physicsClient < 0 or self.carId is None: # Ensure car exists for fp/tp views
             return # Cannot set camera without simulation and car

        # --- Third-person camera (Default TP) ---
        # Fixed perspective slightly behind and above the scene origin
        cam_dist = 5
        cam_yaw = 50
        cam_pitch = -30
        cam_target_pos = [0, 0, 0] # Look at the center of the arena

        # --- First-person camera ---
        if mode == 'fp_camera':
             # Get car's current position and orientation
             car_pos, car_orn = self._p.getBasePositionAndOrientation(self.carId)
             # Calculate camera position slightly above and behind the car's center
             # Use orientation to calculate forward vector
             rot_matrix = np.array(self._p.getMatrixFromQuaternion(car_orn)).reshape(3, 3)
             forward_vec = rot_matrix[:, 0] # Usually X-axis is forward in URDF
             up_vec = rot_matrix[:, 2]      # Z-axis is up

             cam_offset_forward = -0.5 # Place camera slightly behind car center
             cam_offset_up = 0.5      # Place camera slightly above car center
             cam_pos = np.array(car_pos) + cam_offset_forward * forward_vec + cam_offset_up * up_vec

             # Target position slightly in front of the car
             target_offset_forward = 2.0
             cam_target_pos = np.array(car_pos) + target_offset_forward * forward_vec

             self._view_matrix = self._p.computeViewMatrix(
                 cameraEyePosition=cam_pos,
                 cameraTargetPosition=cam_target_pos,
                 cameraUpVector=up_vec)

        # --- Third-person camera ---
        elif mode == 'tp_camera':
            # Option 1: Fixed View
             cam_target_pos = [0,0,0] # Look at center
             cam_dist=10; cam_yaw=45; cam_pitch=-40 # Adjust as preferred

            # Option 2: Follow Cam (Adjust target based on car)
            # car_pos, _ = self._p.getBasePositionAndOrientation(self.carId)
            # cam_target_pos = car_pos
            # cam_dist=5; cam_yaw=45; cam_pitch=-35 # Closer follow distance


             self._view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
                 cameraTargetPosition=cam_target_pos,
                 distance=cam_dist,
                 yaw=cam_yaw,
                 pitch=cam_pitch,
                 roll=0,
                 upAxisIndex=2) # Z-axis is up
        else:
             # Should not happen if checks above are working, but default just in case
              if self._view_matrix is None: # Set only if not already set
                 self._view_matrix = self._p.computeViewMatrixFromYawPitchRoll([0,0,0], 5, 50, -30, 0, 2)


        # Set projection matrix (usually constant unless FOV changes)
        if self._proj_matrix is None:
            self._proj_matrix = self._p.computeProjectionMatrixFOV(
                fov=60, # Field of view
                aspect=400./300., # Width / Height ratio of camera image
                nearVal=0.1, # Near plane distance
                farVal=100.0) # Far plane distance


    def close(self):
        """Cleans up the environment."""
        if self.physicsClient >= 0:
             if self._p.isConnected():
                  print("Disconnecting PyBullet.")
                  self._p.disconnect()
             self.physicsClient = -1 # Mark as disconnected
        print("Environment closed.")
