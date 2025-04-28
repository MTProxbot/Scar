import pybullet as p
import os

class Obstacle:

    def __init__(self, client, base):
        f_name = os.path.join(os.path.dirname(__file__), 'cube.urdf')
        self.obstacle_id = client.loadURDF(fileName=f_name,
                                            basePosition=base_pos,
                                            basePosition=[0, 0, 0.1])
