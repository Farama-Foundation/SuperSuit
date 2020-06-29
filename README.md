# SuperSuit

[![Build Status](https://travis-ci.com/PettingZoo-Team/SuperSuit.svg?branch=master)](https://travis-ci.com/PettingZoo-Team/SuperSuit)

SuperSuit introduces a collection of small functions which can wrap reinforcement learning environments to do preprocessing ('microwrappers').
We support Gym for single agent environments and PettingZoo for multi-agent environments. Using it to convert space invaders to have a grey scale observation space and stack the last 4 frames looks like:

```
import gym
from supersuit.gym_wrappers import color_reduction, frame_stack

env = gym.make('SpaceInvaders-v0')

env = frame_stack(color_reduction(env, 'full'), 4)
```

You can install it via `pip install supersuit`

Similarly, for pettingzoo,

```
import pettingzoo.gamma
from supersuit.aec_wrappers import color_reduction, frame_stack

env = pettingzoo.gamma.pistonball_v0.env()

env = frame_stack(color_reduction(env, 'full'), 4)
```

## Built in Functions

`color_reduction(env, mode='full')` simplifies color information in graphical ((x,y,3) shaped) environments. `mode='full'` fully greyscales of the observation. This can be computationally intensive. Arguments of 'R', 'G' or 'B' just take the corresponding R, G or B color channel from observation. This is much faster and is generally sufficient.

`continuous_actions(env, bounds=(-10,10))` discrete action spaces are converted to a 1d Box action space of size *n*. This space is treated as a vector of logits, and the multinomial distribution defined by those input logits is sampled to get a discrete value. The lower and upper bounds of this Box space can be specified by the `bounds` parameter. Currently supports Discrete action spaces. It passes Box action spaces through without any alteration.

`down_scale(env, x_scale=1, y_scale=1)` uses mean pooling to reduce the observations output by each game by the given x and y scales. The dimension of an environment must be an integer multiple of it's scale. This is only available for 2D or 3D observations.

`dtype(env, dtype)` recasts your observation as a certain dtype. Many graphical games return `uint8` observations, while neural networks generally want `float16` or `float32`.

`flatten(env)` flattens observations into a 1D array.

`frame_stack(env, num_frames=4)` stacks the most recent frames. For vector games observed via plain vectors (1D arrays), the output is just concatenated to a longer 1D array. For games via observed via graphical outputs (a 2D or 3D array), the arrays are stacked to be taller 3D arrays. At the start of the game, frames that don't yet exist are filled with 0s. `num_frames=1` is analogous to not using this function.

`normalize_obs(env, env_min=0, env_max=1)` linearly scales observations to be 0 to 1, given known minimum and maximum observation values. Only works on Box observations with finite bounds.

`reshape(env, shape)` reshapes observations into given shape.


## Built in Multi-Agent Only Functions

`agent_indicator(env, type_only=False)` Adds an indicator of the agent ID to the observation, only supports discrete and 1D, 2D, and 3D box. This allows MADRL methods like parameter sharing to learn policies for heterogeneous agents since the policy can tell what agent it's acting on. The `type_only` parameter means that only the type of the agent defined in the `<type>_<n>` name specification is added to the observation. This is useful for environments with a large number of mostly homogeneous agents.

`pad_action_space(env)` actions spaces of all players will all be padded to be be the same as the biggest, per the algorithm posed in *Parameter Sharing is Surprisingly Useful for Deep Reinforcement Learning*.  This enables MARL methods that require the homogeneous action spaces for all agents to work in environments with heterogeneous action spaces. Discrete actions inside padded region will be set to zero, and Box actions will be cropped down to the original space.

`pad_observations(env)` pads observations to be of the shape of the largest observation of any agent, per the algorithm posed in *Parameter Sharing is Surprisingly Useful for Deep Reinforcement Learning*. This enables MARL methods that require homogeneous observations from all agents to work in environments with heterogeneous observations. This currently supports Discrete and Box observation spaces.

`clip_reward(env, lower_bound=-1, upper_bound=1)` clips rewards to between lower_bound and upper_bound. This is a popular way of handling rewards with significant variance of magnitude, especially in Atari environments.

## Lambda Functions

If none of the build in micro-wrappers are suitable for your needs, you can use a lambda function (or if your needs are still not met, submit a PR).

`action_lambda(env, change_action_fn, change_space_fn)` allows you to define arbitrary changes to the actions via `change_action_fn(action, space) : action` and to the action spaces with `change_space_fn(action_space) : action_space`. Remember that you are transforming the actions received by the wrapper to the actions expected by the base environment.

`observation_lambda(env, observation_fn, observation_space_fn=None)` allows you to define arbitrary changes to the via `observation_fn(observation) : observation`, and `observation_space_fn(obs_space) : obs_space`. For Box-Box transformations the space transformation will be inferred from `change_observation_fn` if `change_obs_space_fn=None` by passing the `high` and `low` bounds through the `observation_space_fn`.

`reward_lambda(env, change_reward_fn)` allows you to make arbitrary changes to rewards by passing in a `change_reward_fn(reward) : reward` function. For gym environments this is called every step to transform the returned reward. For AECEnv, this function is used to change each element in the rewards dictionary every step, taking NxM time.

### Lambda Function Examples

Adding noise to a Box observation looks like:

```
env = observation_lambda(env, lambda x : x + np.random.normal(size=x.shape))
```

Adding noise to a box observation and increasing the high and low bounds to accommodate this extra noise looks like:

```
env = observation_lambda(env,
    lambda x : x + np.random.normal(size=x.shape),
    lambda obs_space : gym.spaces.Box(obs_space.low-5,obs_space.high+5))
```

If you know the inner details of the environment, you can hardcode the appropriate values. For example, if you know you have a Box space of 20x20, you can just do

```
env = observation_lambda(env,
    lambda x : np.pad(x,pad_width=4)
    lambda _ : gym.spaces.Box(low=0,high=1,shape=(28,28)))
```

Changing 1d box action space to a Discrete space by mapping the discrete actions to one-hot vectors.

```
def one_hot(x,n):
    v = np.zeros(n)
    v[x] = 1
    return v

env = action_lambda(env,
    lambda action, act_space : one_hot(action, act_space.shape[0]),
    lambda act_space : gym.spaces.Discrete(act_space.shape[0]))
```
