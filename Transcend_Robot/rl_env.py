import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import math
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
import torch


class HumanoidEnv(gym.Env):
    
    '''The gym RL Env class for the Transcend Robot'''
    
    def __init__(self):
        
        print("------------------------------------------------------------------------------------------")
        self.timestep = 0
        self.terminated = False
        self.truncated = False
        self.foot_contact_reward = 0
        self.info = {}
        self.seed = None
        physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Spawning the Plane in pybullet
        print("LOADING Terrain!")
        z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
        self.planeId = p.loadURDF("plane_implicit.urdf", [0, 0, 0], z2y, useMaximalCoordinates=True)
        p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
        p.setGravity(0, -9.8, 0)
        p.setPhysicsEngineParameter(numSolverIterations=10)
        p.changeDynamics(self.planeId, linkIndex=-1, lateralFriction=0.9)
        p.setRealTimeSimulation(1)
        
        #Spawning the robot in pybullet
        print("LOADING Humanoid!")
        time.sleep(2)
        flags=p.URDF_MAINTAIN_LINK_ORDER+p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        useFixedBase=False
        self.robotID = p.loadURDF("Transcend.urdf", [0, 1.1, 0], globalScaling=1, useFixedBase=useFixedBase, flags=flags)
        p.setJointMotorControlArray(self.robotID, range(38), p.POSITION_CONTROL, [0]*38)
        #Setting up the camera
        p.resetDebugVisualizerCamera(cameraDistance=1.96, cameraYaw=-90, cameraPitch=-13,cameraTargetPosition=[-0.74, 0.61, -0.01])
        
        #enabling simulation
        
        #TODO : set friction coefficients between the hands, legs and the surface 
        p.changeDynamics(self.robotID, 11, lateralFriction=0.3)
        p.changeDynamics(self.robotID, 17, lateralFriction=0.3)
        #wait for 2 seconds to set everything up
        present_time = time.time()
        while(time.time() <= present_time + 2):
            pass
        #Setting up my state : essentially all the corresponding joint angles of each joint ID
        self.joint_num = p.getNumJoints(self.robotID)
        self.jointNames = [0]*self.joint_num
        self.joints_lower_limit = [0]*(self.joint_num)
        self.joints_upper_limit = [0]*(self.joint_num)
        self.joints_present_pos = [0]*(self.joint_num)
        
        for j in range(p.getNumJoints(self.robotID)):
            joint_info = p.getJointInfo(self.robotID, j)
            self.jointNames[j] = joint_info[1]
            self.joints_lower_limit[j] = joint_info[8]
            self.joints_upper_limit[j] = joint_info[9]
        self.joints_lower_limit = [-3]*38
        self.joints_upper_limit = [3]*38
            
        #defining my action space
        #defining a normaised action space where all the positions of the joints are mapped from 0 to 1
        self.action_space = spaces.Box(np.array([0]*len(self.jointNames)), np.array([1]*len(self.jointNames)))
        
        #defining my observation space
        #defining a normalised observation space where all the positions of the joints are mapped from 0 to 1
        self.observation_space = spaces.Box(np.array([0]*len(self.jointNames)), np.array([1]*len(self.jointNames)))
        
        
    def step(self, actions):
        #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        #print("step is running")
        #print(actions)
        #print(self.truncated, self.terminated)
        print(f"I am inside step {self.timestep}")
        self.timestep = self.timestep + 1
        rn_actions = [0]*(self.joint_num)
        
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()

        #print(actions)
        #actions = [0]*38
        #Renormalising the action list to within the actual upper and lower limits of the joint
        for i in range(len(actions)) :
            rn_actions[i] = ((self.joints_upper_limit[i] - self.joints_lower_limit[i])*(actions[i] - 1)) + self.joints_upper_limit[i]
        #rn_actions = [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        #print(rn_actions)
        #print("*************************************")
            
        #Setting joint angles using position control
        p.setJointMotorControlArray(self.robotID, range(38), p.TORQUE_CONTROL, rn_actions)
        present_time = time.time()
        while(time.time() <= present_time + 0.3):
            pass
        #p.setJointMotorControlArray(self.robotID, range(38), p.POSITION_CONTROL, [1]*38)
        updated_angles = [0]*(p.getNumJoints(self.robotID))
        updated_obs = [0]*(p.getNumJoints(self.robotID))
        
        #Getting present angles after change
        for j in range(self.joint_num):
            present_info = p.getJointState(self.robotID, j)
            updated_angles[j] = present_info[0]
            updated_obs[j] = (present_info[0] - self.joints_upper_limit[i])/(self.joints_upper_limit[i] - self.joints_lower_limit[i]) + 1
            
        #print(updated_angles)
        reward_var = 0
        done = False
        # TODO : Design reward function
        # Dependent on : distance between both the hands and the surface
        r_wrist_pos_1 = p.getLinkState(self.robotID,22)
        r_wrist_pos_2 = p.getLinkState(self.robotID,23)
        l_wrist_pos_1 = p.getLinkState(self.robotID,32)
        l_wrist_pos_2 = p.getLinkState(self.robotID,33)
        r_hand_pos = (r_wrist_pos_1[0][1] + r_wrist_pos_2[0][1])/2
        l_hand_pos = (l_wrist_pos_1[0][1] + l_wrist_pos_2[0][1])/2
        reward_var = 50*(1-r_hand_pos) + 50*(1-l_hand_pos)

        # penatly for collision
        check_collision = 0
        safe_links_IDs = [11,17,22,23,24,25,26,27,32,33,34,35,36,37]
        contact_info = p.getContactPoints(self.robotID, self.planeId)
        for i in range(len(contact_info)):
            if contact_info[i][3] not in safe_links_IDs:
                check_collision = 1
                reward_var = reward_var - 10
            else:
                reward_var = reward_var + 5
        if(check_collision == 1):
            
            print("thre is collision")
        else:
            
            print("there is no collision")
            
        # always in contact with the ground in both foot
        foot_contact_points = 0
        for i in range(len(contact_info)):
            if (((contact_info[i][3]) == 11) or ((contact_info[i][3]) == 17)):
                foot_contact_points += 1
            
        #print(f"number of total contact points in feet is {foot_contact_points}")
        if(foot_contact_points < 6):
            reward_var = reward_var - 15
        else :
            self.foot_contact_reward += 2
            reward_var = reward_var + self.foot_contact_reward
            
        if reward_var > 300:
            self.terminated = True
            print(f"reward after termination = {reward_var}")
        if self.timestep > 50:
            self.truncated = True
            print(f"reward after truncation = {reward_var}")

        self.info = {}
        updated_obs = [i if math.isnan(i) == False and math.isinf(i) == False else 0 for i in updated_obs]
        
        return updated_obs, reward_var, self.terminated, self.truncated, self.info
    
    def reset(self, *, seed=None, **kwargs):
        print("I am reseting")
        self.timestep = 0
        self.terminated = False
        self.truncated = False
        #super().reset(seed=None)
        #Reseting the simulation
        p.setJointMotorControlArray(self.robotID, range(38), p.POSITION_CONTROL, [0]*38)
        p.resetBasePositionAndOrientation(self.robotID,[0,1.2,0],[0,0,0,1])
        
        print("resetted")+
        #change all joint angles to zero
        #give 2 seconds delay using while pass
        present_time = time.time()
        while(time.time() <= present_time + 5):
            pass
        initial_state = [0]*38
        return initial_state, self.info

    def close(self):
        #disconnect the simulation
        p.disconnect()
    
env = DummyVecEnv([lambda: HumanoidEnv()])
env = VecCheckNan(env, raise_exception=True)
#observation, info = env.reset()
#print (env)
#print(env.joints_present_pos)

ppo_agent = PPO('MlpPolicy', env, verbose=2, learning_rate=0.0003)
ppo_agent.learn(total_timesteps=10000)
ppo_agent.save('transcend_robot')
mean_reward, std_reward = evaluate_policy(ppo_agent, env, n_eval_episodes=10, deterministic=True)
print(f'Mean reward: {mean_reward}, Std reward: {std_reward}')


# while True:
#     print("in loop")
# '''while True:
#     actions_test = env.action_space.sample()
#     #print(actions_test)
#     observation, reward, done, info = env.step(actions_test)
#     present_time = time.time()
#     print(observation)
#     while(time.time() <= present_time + 0.2):
#        pass
#print("over")
    #print("GOing back to a new action")'''

