from baselines.deepq.experiments.training_wrapper import make_dqn
from gym.spaces import Box
import numpy as np

class GoalEnv:

    def __init__(self, env):
        orig_obs = env.observation_space
        self.action_space = env.action_space
        [h, w, c] = orig_obs.shape
        self.observation_space = Box(0, 255, shape=[h,w,2*c], dtype=orig_obs.dtype)



class GoalQWrapper:

    def __init__(self, env, name, gpu_num):
        goal_env = GoalEnv(env)
        self.dqn = make_dqn(goal_env, name, gpu_num)


    def train_batch_goals(self, time, s, a, sp, goals):
        eps = 0.1
        goals = [sp[i] if np.random.uniform(0, 1) < eps else goals[i] for i in range(len(goals))]
        s = [np.concatenate([s[i], goals[i]], axis=2) for i in range(len(s))]
        sp = [np.concatenate([sp[i], goals[i]], axis=2) for i in range(len(sp))]
        r = [1.0 if np.array_equal(sp[i], goals[i]) else -0.01 for i in range(len(sp))]
        t = [np.array_equal(sp[i], goals[i]) for i in range(len(sp))]
        loss = self.dqn.train_batch(time, s, a, r, sp, t, np.ones_like(t), None)
        return loss

    def get_values(self, s, goals):
        s = [np.concatenate([s[i], goals[i]], axis=2) for i in range(len(s))]
        return np.max(self.dqn.get_Q(s), axis=1)




