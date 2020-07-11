
def check_transform_frameskip(frame_skip):
    if isinstance(frame_skip, tuple) and len(frame_skip) == 2 and isinstance(frame_skip[0], int) and isinstance(frame_skip[1], int) and 1 <= frame_skip[0] <= frame_skip[1]:
        return frame_skip
    elif isinstance(frame_skip, int):
        return (frame_skip, frame_skip)
    else:
        assert False, "frame_skip must be an int or a tuple of two ints, where the first values is at least one and not greater than the second"
