<p align="center">
    <img src="https://raw.githubusercontent.com/Farama-Foundation/SuperSuit/master/supersuit-text.png" width="500px"/>
</p>


Once the planned wrapper rewrite of Gymnasium is complete and the vector API is stabilized, this project will be deprecated and rewritten as part of a new wrappers package in pettingzoo and the vectorized API will be redone, taking inspiration from the functionality currently in gymnasium.


SuperSuit introduces a collection of small functions which can wrap reinforcement learning environments to do preprocessing ('microwrappers').
We support Gymnasium for single agent environments and PettingZoo for multi-agent environments (both AECEnv and ParallelEnv environments). Using it to convert space invaders to have a grey scale observation space and stack the last 4 frames looks like:

```
import gymnasium
from supersuit import color_reduction_v0, frame_stack_v1

env = gymnasium.make('SpaceInvaders-v0')

env = frame_stack_v1(color_reduction_v0(env, 'full'), 4)
```

Similarly, using SuperSuit with PettingZoo environments looks like

```
from pettingzoo.butterfly import pistonball_v0
env = pistonball_v0.env()

env = frame_stack_v1(color_reduction_v0(env, 'full'), 4)
```

You can install SuperSuit via `pip install supersuit`


## Citation

If you use this in your research, please cite:

```
@article{SuperSuit,
  Title = {SuperSuit: Simple Microwrappers for Reinforcement Learning Environments},
  Author = {Terry, J. K and Black, Benjamin and Hari, Ananth},
  journal={arXiv preprint arXiv:2008.08932},
  year={2020}
}
```
