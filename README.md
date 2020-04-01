# SuperSuit

SuperSuit introduces a collection of small functions which can wrap reinforcement learning environments to do preprocessing ('microwrappers').
We support Gym for single agent environments and PettingZoo for multi-agent environments. Using it to convert space invaders to have a grey scale observation space and stack the last 4 frames looks like:

```
import gym
from supersuit import color_reduction, frame_stacking

env = gym.make('SpaceInvaders-v0')

env = frame_stacking(color_reduction(env, 'full'))
```


## Full list of functions:

`color_reduction(env, mode='full')` simplifies color information in graphical ((x,y,3) shaped) environments. `mode='full'` fully greyscales of the observation. This can be computationally intensive. Arguments of 'R', 'G' or 'B' just take the corresponding R, G or B color channel from observation. This is much faster and is generally sufficient.

`continuous_actions(env)` discrete action spaces are converted to a 1d Box action space of size *n*. This space is treated as a vector of logits, and the softmax distribution of the inputs is sampled to get a discrete value. Currently supports both Discrete and MultiDiscrete action spaces. <Ben fix this one please>

`down_scale(env, x_scale=1, y_scale=1)` uses mean pooling to reduce the observations output by each game by the given x and y scales. The dimension of an environment must be an integer multiple of it's scale. This is only available for 2D or 3D observations.

`dtype(env, dtype)` recasts your observation as a certain dtype. Many graphical games return `uint8` observations, while neural networks generally want `float16` or `float32`.

`flatten(env)` flattens observations into a 1D array. <Ben this used to be part of reshape>

`frame_stack(env, num_frames=4)` stacks the most recent frames. For vector games observed via plain vectors (1D arrays), the output is just concatenated to a longer 1D array. For games via observed via graphical outputs (a 2D or 3D array), the arrays are stacked to be taller 3D arrays. At the start of the game, frames that don't yet exist are filled with 0s. `num_frames=1` is analogous to not using this function.

`normalize_obs(env, env_min=0, env_max=1)` linearly scales observations to be 0 to 1, given known minimum and maximum observation values. <Ben this used to be called range_scale.>

`reshape(env, shape)` reshapes observations into given shape. <Ben I slightly changed what this function should be.>
