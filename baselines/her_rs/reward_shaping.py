import numpy as np
import pandas as pd
import os
import gym
import tqdm
import time

from baselines import logger
from baselines.her_rs.util import store_args


def generate_subgoals(n_obs):
    # Subgoal1: Objectの絶対座標[x,y,z] = achieved_goal
    # Subgoal2: Objectの絶対座標とArmの位置が同じでアームを閉じている状態。
    subgoal1 = [None for _ in range(n_obs)]
    subgoal1[6:9] = [0, 0, 0]
    subgoal2 = [None for _ in range(n_obs)]
    subgoal2[6:11] = [0, 0, 0, 0.02, 0.02]
    return [subgoal1, subgoal2]


class Subgoal:
    @store_args
    def __init__(self, obs, range=0.008):
        # self.grip_pos​ = obs[0:3]
        # self.object_pos​ = obs[3:6]
        self.object_rel_pos = obs[6:9]
        self.gripper_state = obs[9:11]
        # self.object_rot​ = obs[11:14]
        # self.object_velp​ = obs[14:17]
        # self.object_velr​ = obs[17:20]
        # self.grip_velp​ = obs[20:23]
        # self.gripper_vel​ = obs[23:25]
        self.next_subgoal = None

    def set_next(self, subgoal):
        self.next_subgoal = subgoal

    def get_next(self):
        return self.next_subgoal

    def equal(self, obs):
        object_rel_pos = obs[0][6:11]
        assert len(object_rel_pos) == 5
        for li, mi in zip(self.object_rel_pos + self.gripper_state, object_rel_pos):
            if li is None:
                continue
            if mi < li - self.range or li + self.range < mi:
                return False
        return True


class SubgoalPotentialRewardShaping:
    @store_args
    def __init__(self, gamma, n_obs=25, subgoals=None):
        self.last_potential = 0
        if subgoals is None:
            subgoals = generate_subgoals(n_obs)
        self.subgoals = self.arange_subgoals(subgoals)
        self.reset_next_subgoal()
        self.subg_state = 0
        self.prev_state = 0
        self.dur_subg_state = 0

    def value(self, last_obs, action, reward, obs, done):
        self.prev_state = self.subg_state
        if self.target_subgoal is not None and self.target_subgoal.equal(obs):
            logger.info("Reach Subgoal{} at {}, state: {} in subg: {}!".format(self.subg_state, self.dur_subg_state, obs[0][6:11], self.target_subgoal.object_rel_pos+self.target_subgoal.gripper_state))
            self.target_subgoal = self.target_subgoal.get_next()
            self.subg_state += 1
            # dur_subg_stateのリセットタイミングはアルゴリズムに依るので、子クラスで実装。
        self.dur_subg_state += 1
        self.train(last_obs, action, reward, obs, done)
        current_potential = self.potential()
        v = self.gamma * current_potential - self.last_potential
        self.last_potential = current_potential
        if done:
            self.reset()
        return v

    def potential(self):
        raise NotImplementedError

    def train(self, last_obs, action, reward, obs, done):
        pass

    def reset_next_subgoal(self):
        self.target_subgoal = self.subgoals[0]

    def arange_subgoals(self, subgoals):
        series = [Subgoal(s) for s in subgoals]
        for i in range(len(subgoals)-1):
            series[i].set_next(series[i+1])
        return series

    def reset(self):
        self.reset_next_subgoal()
        self.subg_state = 0
        self.dur_subg_state = 0
        self.last_potential = 0


class NaiveSubgoalPotential(SubgoalPotentialRewardShaping):
    def __init__(self, gamma, eta, n_obs, subgoals=None):
        logger.info("Naive Subgoal Potential class.")
        super().__init__(gamma, n_obs, subgoals)
        self.eta = eta

    def potential(self):
        if self.prev_state != self.subg_state:
            return self.subg_state * self.eta
        else:
            return 0


class FixedSubgoalPotential(SubgoalPotentialRewardShaping):
    def __init__(self, gamma, eta, rho, n_obs=25, subgoals=None):
        super().__init__(gamma, n_obs, subgoals)
        self.eta = eta
        self.rho = rho

    def potential(self):
        if self.prev_state != self.subg_state:
            self.dur_subg_state = 0
        if self.subg_state > 0:
            logger.info("prev: {}, current: {}".format(self.prev_state, self.subg_state))
            logger.info("eta: {}, rho: {}".format(self.eta, self.rho))
            logger.info(self.subg_state * self.eta, self.dur_subg_state * self.rho)
        return max(self.subg_state * self.eta - self.dur_subg_state * self.rho, 0)


class OnlineLearningRewardShaping(SubgoalPotentialRewardShaping):
    def __init__(self, gamma, lr, n_obs=25, subgoals=None):
        super().__init__(gamma, n_obs, subgoals)
        self.lr = lr
        self.reset_value()
        # TODO  desired goalの位置情報をサブゴール状態に含める必要があるか？

    def reset_value(self):
        self.v = np.zeros(len(self.subgoals) + 1)

    def train(self, last_obs, action, reward, obs, done):
        if reward != 0 or self.subg_state != self.prev_state:
            tderror = reward + self.gamma ** self.dur_subg_state * self.v[self.subg_state] - self.v[self.prev_state]
            self.v[self.prev_state] = self.v[self.prev_state] + self.lr * tderror
            # logger.info(self.v)
            self.dur_subg_state = 0

    def potential(self):
        return self.v[self.subg_state]



def main():
    gamma = 0.99
    eta = 1
    rho = 0.1
    n_obs = 25
    alpha = 0.001
    rs = OnlineLearningRewardShaping(gamma, alpha, n_obs)
    env = gym.make("FetchPickAndPlace-v1")
    pre_obs = env.reset()["observation"]
    done = False
    while(not done):  # for _ in tqdm.tqdm(range(10000)):
        action = 2 * np.random.rand(4) - 1
        obs, r, done, _ = env.step(action)
        logger.info(r)
        obs = obs["observation"]
        v = rs.value(pre_obs, action, r, obs, done)
        env.render()
        # if v != 0:
        logger.info(v)
        #     time.sleep(5)
        #     break
        pre_obs = obs


if __name__ == "__main__":
    main()
