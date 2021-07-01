from supersuit.utils.wrapper_chooser import WrapperChooser
from .frame_skip_aec import frame_skip_aec
from .frame_skip_gym import frame_skip_gym


frame_skip_v0 = WrapperChooser(aec_wrapper=frame_skip_aec, gym_wrapper=frame_skip_gym)
