import gym
import numpy as np
import math
import pybullet as p
from pybullet_utils import bullet_client as bc
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
from simple_driving.resources.obstacle import Obstacle
import matplotlib.pyplot as plt
import time

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'fp_camera', 'tp_camera']}

    def __init__(self, isDiscrete=True, renders=False):
        if (isDiscrete):
            self.action_space = gym.spaces.Discrete(9)
        else:
            self.action_space = gym.spaces.box.Box(
                low=np.array([-1, -.6], dtype=np.float32),
                high=np.array([1, .6], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-40, -40], dtype=np.float32),
            high=np.array([40, 40], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        if renders:
          self._p = bc.BulletClient(connection_mode=p.GUI)
        else:
          self._p = bc.BulletClient()

        self.reached_goal = False
        self._timeStep = 0.01
        self._actionRepeat = 50
        self._renders = renders
        self._isDiscrete = isDiscrete
        self.car = None
        self.goal_object = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()
        self._envStepCounter = 0

    def step(self, action):
        # Feed action to the car
        if (self._isDiscrete):
            # Example discrete actions -> continuous [throttle, steering]
            fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1] # Example throttle mapping
            steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6] # Example steering mapping
            if action < 0 or action >= len(fwd):
                raise ValueError(f"Invalid discrete action: {action}")
            throttle = fwd[action]
            steering_angle = steerings[action]
            cont_action = [throttle, steering_angle] # Converted action
        else:
            cont_action = action # Assume action is already [throttle, steering]

        # Apply the continuous action to the car model
        if self.car:
            self.car.apply_action(cont_action)
        else:
             raise RuntimeError("Car object not initialized. Call reset() first.")


        # Step simulation multiple times
        car_ob = None # Initialize observation variable
        carpos = None # Initialize car position variable
        for i in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)

            # Get current state inside the loop for termination check and final observation
            # Query PyBullet for the car's state
            car_id = self.car.get_ids()
            current_carpos, carorn = self._p.getBasePositionAndOrientation(car_id)
            carpos = list(current_carpos) # Store the latest position

            # Get the observation state (e.g., pos, vel, orientation)
            car_ob = self.getExtendedObservation() # Based on latest state

            # Check for termination conditions *during* the steps
            if self._termination(carpos): # Pass current pos to termination check
                self.done = True
                print(f"INFO: Termination condition met at step {self._envStepCounter}")
                break # Exit action repeat loop early if terminated

            self._envStepCounter += 1 # Increment counter only if simulation step was taken

        # --- Calculations AFTER the action repeat loop ---

        # Ensure carpos is available (might not be if _actionRepeat is 0 or terminated instantly)
        if carpos is None:
            if self.car:
                carpos, _ = self._p.getBasePositionAndOrientation(self.car.get_ids())
                carpos = list(carpos)
            else:
                 # Fallback if car doesn't exist - should not happen if reset correctly
                 carpos = [0,0,0]
                 print("WARN: Car position unknown after step loop.")


        # Get final car position
        car_x, car_y = carpos[0], carpos[1]

        # Get goal position coordinates from self.goal tuple
        x, y = self.goal[0], self.goal[1] # Use x, y as requested

        # Compute distance to goal based on final car position
        dist_to_goal = math.sqrt(((car_x - x)**2 + (car_y - y)**2))

        # Calculate reward (negative distance to goal)
        reward = -dist_to_goal
        # Optional: Reward based on progress (uncomment if preferred)
        # reward = self.prev_dist_to_goal - dist_to_goal
        # self.prev_dist_to_goal = dist_to_goal

        # --- Obstacle Avoidance Penalty ---
        obstacle_penalty = 0.0
        # Check if obstacle exists (check ID) and get its fixed position
        # Obstacle must have been successfully loaded in reset()
        if (hasattr(self, 'obstacle') and self.obstacle and
            self.obstacle.get_id() is not None and hasattr(self, 'obstacle_position')):

            obstacle_x = self.obstacle_position[0]
            obstacle_y = self.obstacle_position[1]

            # Calculate distance from car center to obstacle center (horizontal plane)
            dist_to_obstacle = math.sqrt(((car_x - obstacle_x)**2 + (car_y - obstacle_y)**2))

            # Apply penalty if too close
            if dist_to_obstacle < self._obstacle_prox_threshold:
                obstacle_penalty = -self._obstacle_penalty_amount
                if self._renders or True: # Print penalty info even if not rendering for debug
                   print(f"Step {self._envStepCounter}: Too close to obstacle! Dist: {dist_to_obstacle:.2f}, Penalty: {obstacle_penalty}")

        # Add obstacle penalty to the reward
        reward += obstacle_penalty
        # --- End Obstacle Penalty ---

        # --- Goal Reached Check ---
        # Done by reaching goal (check uses final dist_to_goal)
        if dist_to_goal < self._goal_reach_threshold and not self.reached_goal:
            print(f"INFO: Reached goal at step {self._envStepCounter}!")
            self.done = True # Set done flag
            self.reached_goal = True
            reward += self._goal_bonus_reward # Add goal bonus reward

        # --- Max Steps Check (Alternative termination) ---
        # Example: Add max steps termination if not done by other means
        max_steps = 1000 # Example max steps per episode
        if not self.done and self._envStepCounter >= max_steps:
             print(f"INFO: Max steps ({max_steps}) reached.")
             self.done = True
             # Optional penalty for running out of time could be added here

        # --- Final Observation ---
        # Use the last observation captured in the loop or get a fresh one if loop didn't run
        if car_ob is None:
             ob = self.getExtendedObservation()
             print("WARN: Using observation fetched after loop.")
        else:
             ob = car_ob

        # Ensure the observation is a numpy float32 array
        if not isinstance(ob, np.ndarray):
            ob = np.array(ob, dtype=np.float32)
        elif ob.dtype != np.float32:
            ob = ob.astype(np.float32)

        # Return standard Gym format (using only `done` flag)
        # info dict is empty as per original structure
        return ob, reward, self.done, {}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)
        # Reload the plane and car
        Plane(self._p)
        self.car = Car(self._p)
        self._envStepCounter = 0

        # Set the goal to a random target
        x = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
             self.np_random.uniform(-9, -5))
        y = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
             self.np_random.uniform(-9, -5))
        self.goal = (x, y)
        self.done = False
        self.reached_goal = False

        # Visual element of the goal
        self.goal_object = Goal(self._p, self.goal)
        
        # Determine a random position for the obstacle
        obstacle_x = t * x
        obstacle_y = t * y
        while True:
            obstacle_x = self.np_random.uniform(-8, 8)
            obstacle_y = self.np_random.uniform(-8, 8)
            dist_from_origin = math.sqrt(obstacle_x**2 + obstacle_y**2)
            dist_from_goal = math.sqrt((obstacle_x - x)**2 + (obstacle_y - y)**2)
            if dist_from_origin > min_dist_from_origin and dist_from_goal > min_dist_from_goal:
                break

        # Set z position based on URDF definition (box size is 0.05)
        obstacle_z = 0
        self.obstacle_position = [obstacle_x, obstacle_y, obstacle_z] # Store position
    
        # Instantiate the Obstacle class, passing the client and position
        # The Obstacle class now handles the actual loadURDF call
        self.obstacle = Obstacle(self._p, self.obstacle_position)
        
        # Get observation to return
        carpos = self.car.get_observation()

        self.prev_dist_to_goal = math.sqrt(((carpos[0] - self.goal[0]) ** 2 +
                                           (carpos[1] - self.goal[1]) ** 2))
        car_ob = self.getExtendedObservation()
        return np.array(car_ob, dtype=np.float32)

    def render(self, mode='human'):
        if mode == "fp_camera":
            # Base information
            car_id = self.car.get_ids()
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                       nearVal=0.01, farVal=100)
            pos, ori = [list(l) for l in
                        self._p.getBasePositionAndOrientation(car_id)]
            pos[2] = 0.2

            # Rotate camera direction
            rot_mat = np.array(self._p.getMatrixFromQuaternion(ori)).reshape(3, 3)
            camera_vec = np.matmul(rot_mat, [1, 0, 0])
            up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
            view_matrix = self._p.computeViewMatrix(pos, pos + camera_vec, up_vec)

            # Display image
            # frame = self._p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
            # frame = np.reshape(frame, (100, 100, 4))
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
            # self.rendered_img.set_data(frame)
            # plt.draw()
            # plt.pause(.00001)

        elif mode == "tp_camera":
            car_id = self.car.get_ids()
            base_pos, orn = self._p.getBasePositionAndOrientation(car_id)
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                    distance=20.0,
                                                                    yaw=40.0,
                                                                    pitch=-35,
                                                                    roll=0,
                                                                    upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                             aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                             nearVal=0.1,
                                                             farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
        else:
            return np.array([])

    def getExtendedObservation(self):
        # self._observation = []  #self._racecar.getObservation()
        carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
        goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
        invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
        goalPosInCar, goalOrnInCar = self._p.multiplyTransforms(invCarPos, invCarOrn, goalpos, goalorn)

        observation = [goalPosInCar[0], goalPosInCar[1]]
        return observation

    def _termination(self):
        return self._envStepCounter > 2000

    def close(self):
        self._p.disconnect()
