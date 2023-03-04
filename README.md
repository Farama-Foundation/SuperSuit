<p align="center">
    <img src="https://raw.githubusercontent.com/Farama-Foundation/SuperSuit/master/supersuit-text.png" width="500px"/>
</p>


SuperSuit introduces a collection of small functions which can wrap reinforcement learning environments to do preprocessing ('microwrappers').
We support Gymnasium for single agent environments and PettingZoo for multi-agent environments (both AECEnv and ParallelEnv environments).


Using it with Gymnasium to convert space invaders to have a grey scale observation space and stack the last 4 frames looks like:

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


**Please note**: Once the planned wrapper rewrite of Gymnasium is complete and the vector API is stabilized, this project will be deprecated and rewritten as part of a new wrappers package in PettingZoo and the vectorized API will be redone, taking inspiration from the functionality currently in Gymnasium.

## Installing SuperSuit
To install SuperSuit from pypi:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install supersuit
```

Alternatively, to install SuperSuit from source, clone this repo, `cd` to it, and then:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -e .
```

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
