import numpy as np
import pandas as pd
import os
import gym
import tqdm
import time
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea
from collections import OrderedDict

from baselines import logger
from baselines.her_rs.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch, convert_episode_to_batch_major, nn)
from baselines.her_rs.normalizer import Normalizer
from baselines.common.mpi_adam import MpiAdam
from baselines.common import tf_util
from baselines.her_rs.ddpg import DDPG
from baselines.her_rs.replay_buffer import ValueReplayBuffer


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}

def transitions_in_episode_batch2(episode_batch):
    """Number of transitions in a given episode batch.
    """
    shape = episode_batch['o'].shape
    return shape[0] * shape[1]


def generate_subgoals(n_obs):
    # Subgoal1: Objectの絶対座標[x,y,z] = achieved_goal
    # Subgoal2: Objectの絶対座標とArmの位置が同じでアームを閉じている状態。
    subgoal1 = [None for _ in range(n_obs)]
    # subgoal1[6:8] = [0, 0]
    subgoal1[6:9] = [0, 0, 0]
    subgoal2 = [None for _ in range(n_obs)]
    # subgoal2[6:9] = [0, 0, 0]
    subgoal2[6:11] = [0, 0, 0, 0.02, 0.02]
    return [subgoal1, subgoal2]

class Subgoal:
    @store_args
    def __init__(self, id, obs, range=0.01):
        # self.grip_pos​ = obs[0:3]
        # self.object_pos​ = obs[3:6]
        self.object_rel_pos = obs[6:9]
        self.gripper_state = obs[9:11]
        # self.gripper_state​ = obs[9:11]
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
        object_rel_pos = obs[0][6:9]
        for li, mi in zip(self.object_rel_pos + self.gripper_state, object_rel_pos):
            if li is None:
                continue
            if mi < li - self.range or li + self.range < mi:
                return False
        logger.info("Achieve the subgoal{}".format(self.id))
        return True


class OnlineLearningRewardShaping:
    @store_args
    def __init__(self, gamma, lr, policy, n_obs=25, subgoals=None):
        self.last_potential = 0
        if subgoals is None:
            subgoals = generate_subgoals(n_obs)
        self.subgoals = self.arange_subgoals(subgoals)
        self.reset_next_subgoal()
        self.dur_subg_state = 0
        self.r_v = np.array([0])
        self.subg_obs = None
        self.policy = policy

    def value(self, reward, obs, done):
        # if self.target_subgoal is not None and self.target_subgoal.equal(obs):
        #     # logger.info("Reach Subgoal{}!".format(self.subg_state))
        #     self.target_subgoal = self.target_subgoal.get_next()
        #     # dur_subg_stateのリセットタイミングはアルゴリズムに依るので、小クラスで実装。
        # self.dur_subg_state += 1
        current_potential = self.potential(obs)
        v = self.gamma * current_potential - self.last_potential
        self.last_potential = current_potential
        self.dur_subg_state += 1
        # if done:
        #     self.reset()
        return v

    def potential(self, obs):
        # TODO 価値関数の取得
        # 価値関数を返す。
        return self.policy.get_value(obs['observation'], obs['achieved_goal'], obs['desired_goal'])

    def train(self, last_obs, action, reward, obs, done):
        # TODO 価値関数の学習
        pass

    def reset_next_subgoal(self):
        self.target_subgoal = self.subgoals[0]

    def arange_subgoals(self, subgoals):
        series = [Subgoal(i+1, s) for i, s in enumerate(subgoals)]
        for i in range(len(subgoals)-1):
            series[i].set_next(series[i+1])
        return series

    def is_achieve(self, obs):
        return self.target_subgoal is not None and self.target_subgoal.equal(obs)

    def achieve(self, obs):
        # サブゴールに到達したら実行する関数
        self.target_subgoal = self.target_subgoal.get_next()
        self.dur_subg_state = 0
        self.subg_obs = obs

    def get_t(self):
        return self.dur_subg_state

    def get_subg_obs(self):
        return self.subg_obs

    def high_reward(self, r, o, done):
        # self.r_v = self.gamma * self.r_v + sr
        if self.is_achieve(o) or done:
            reward = self.r_v.copy()
            self.r_v = np.array([r], dtype=np.float32)
            return reward
        else:
            self.r_v = self.r_v + self.gamma ** self.dur_subg_state * r
            return np.array([0], dtype=np.float32)

    def start(self, obs):
        self.subg_obs = obs.copy()

    def reset(self):
        # logger.info("Reset Subgoal Series")
        self.reset_next_subgoal()
        self.subg_obs = None
        self.dur_subg_state = 0
        self.last_potential = 0


class DeepValueGradient:
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 bc_loss, q_filter, num_demo, demo_batch_size, prm_loss_weight, aux_loss_weight,
                 sample_transitions, gamma, reuse=False, n_subg=2, **kwargs):
        if self.clip_return is None:
            self.clip_return = np.inf

        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        # Prepare staging area for feeding data to the model.
        self.init_stage_shapes(input_shapes)

        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(reuse=reuse)

        # Configure the replay buffer.
        buffer_shapes = {
            'o': input_shapes['o'],
            'no': input_shapes['o'],
            'g': (self.dimg,),
            'ag':(self.dimg,),
            'r': (1,),
            't': (1,),
            'nag':(self.dimg,)
        }

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ValueReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

    def init_stage_shapes(self, input_shapes):
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        stage_shapes['t'] = (None,)
        stage_shapes['no'] = (None, *input_shapes['o'])
        self.stage_shapes = stage_shapes

    def _create_network(self, reuse=False):
        logger.info("Creating a Deep Value agent with action space %d x %s..." % (self.dimu, self.max_u))
        self.sess = tf_util.get_session()

        # running averages
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])

        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = DVN(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['no']
            target_batch_tf['g'] = batch_tf['g']
            self.target = DVN(
                target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        self.Q_loss_tf = self._create_loss(batch_tf)

        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))

        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)

        # polyak averaging
        self.main_vars = self._vars('main/Q')
        self.target_vars = self._vars('target/Q')
        self.stats_vars = self._global_vars('o_stats')
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def _create_loss(self, batch_tf):
        target_Q_tf = self.target.Q_tf
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma ** batch_tf['t'] * target_Q_tf, *clip_range)
        return tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))

    def store_episode(self, episode_batch, update_stats=True):
        self.buffer.store_episode(episode_batch)

        if update_stats:
            # add transitions to normalizer
            num_normalizing_transitions = transitions_in_episode_batch2(episode_batch)
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

            o, g, ag = transitions['o'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stats

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

    def get_value(self, o, ag, g):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target
        target_Q_tf = policy.Q_tf
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg)
        }
        return self.sess.run(target_Q_tf, feed_dict=feed)

    def sample_batch(self):
        transitions = self.buffer.sample(self.batch_size) #otherwise only sample from primary buffer
        o, o_2, g = transitions['o'], transitions['no'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['nag']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        return transitions_batch

    def _preprocess_og(self, o, ag, g):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def _sync_optimizers(self):
        self.Q_adam.sync()

    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run([
            self.Q_loss_tf,
            self.main.Q_pi_tf,
            self.Q_grad_tf,
        ])
        return critic_loss, actor_loss, Q_grad

    def _update(self, Q_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def train(self, stage=True):
        if stage:
            self.stage_batch()
        critic_loss, actor_loss, Q_grad= self._grads()
        self._update(Q_grad)
        return critic_loss, actor_loss

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res




class DVN:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her_rs.Normalizer): normalizer for observations
            g_stats (baselines.her_rs.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)

        # Networks.
        with tf.variable_scope('Q'):
            # for critic training
            input_Q = tf.concat(axis=1, values=[o, g])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1])



# class SubgDDPG(DDPG):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def init_stage_shapes(self, input_shapes):
#         stage_shapes = OrderedDict()
#         for key in sorted(self.input_dims.keys()):
#             if key.startswith('info_'):
#                 continue
#             stage_shapes[key] = (None, *input_shapes[key])
#         for key in ['o', 'g']:
#             stage_shapes[key + '_2'] = stage_shapes[key]
#         stage_shapes['r'] = (None,)
#         stage_shapes['rs'] = (None,)
#         stage_shapes['t'] = (None,)
#         self.stage_shapes = stage_shapes

#     def store_episode(self, episode_batch, update_stats=True):
#         """
#         episode_batch: array of batch_size x (T or T+1) x dim_key
#                        'o' is of size T+1, others are of size T
#         """
#         #  TODO
#         self.buffer.store_episode(episode_batch)
#         if update_stats:
#             # add transitions to normalizer
#             episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
#             episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
#             num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
#             transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

#             o, g, ag = transitions['o'], transitions['g'], transitions['ag']
#             transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
#             # No need to preprocess the o_2 and g_2 since this is only used for stats

#             self.o_stats.update(transitions['o'])
#             self.g_stats.update(transitions['g'])

#             self.o_stats.recompute_stats()
#             self.g_stats.recompute_stats()


#     def get_value(self, o, ag, g):
#         o, g = self._preprocess_og(o, ag, g)
#         policy = self.target
#         target_Q_pi_tf = policy.Q_pi_tf
#         feed = {
#             policy.o_tf: o.reshape(-1, self.dimo),
#             policy.g_tf: g.reshape(-1, self.dimg),
#             policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
#         }
#         return self.sess.run(target_Q_pi_tf, feed_dict=feed)


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
        # import pdb; pdb.set_trace()
        v = rs.value(pre_obs, action, r, obs, done)
        env.render()
        # if v != 0:
        logger.info(v)
        #     time.sleep(5)
        #     break
        pre_obs = obs


if __name__ == "__main__":
    main()
