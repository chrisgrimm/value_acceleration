from gridworld import Gridworld
from dqn_helpers import GoalQWrapper
from replay_buffer import ReplayBuffer
import numpy as np
import itertools
from visualization import visualize_all_values

env = Gridworld(10)
gpu_num = 0

dqn = GoalQWrapper(env, 'dqn', 0)
buffer = ReplayBuffer(100000)

steps_before_train = 1000
viz_freq = 10000
batch_size = 32

s = env.reset()
for time in itertools.count():

    a = np.random.randint(0, 4)
    sp, r, t, info = env.step(a)
    buffer.append(s, a, r, sp, t)
    s = sp
    if time < steps_before_train:
        continue

    s_batch, a_batch, r_batch, sp_batch, t_batch = buffer.sample(batch_size)
    g_batch, _, _, _, _ = buffer.sample(batch_size)
    loss = dqn.train_batch_goals(time, s_batch, a_batch, sp_batch, g_batch)
    print(time, loss)

    if time % viz_freq == 0:
        visualize_all_values(dqn, env.get_all_states())










