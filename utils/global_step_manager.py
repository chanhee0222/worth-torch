
_global_step_val = 1


def get_global_step():
    return _global_step_val


def set_global_step(step):
    global _global_step_val
    _global_step_val = step
