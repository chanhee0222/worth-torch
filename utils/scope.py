from contextlib import contextmanager


class NameScope:
    def __init__(self):
        # Initialize a new, empty name scope.
        self._name_stack = []

    def sub_scope(self, name):
        self._name_stack.append(name)
        return self

    def parent(self):
        self._name_stack.pop()
        return self

    def __str__(self):
        if len(self._name_stack) > 0:
            return "/".join(self._name_stack) + "/"
        else:
            return ""


_global_default_name_scope = NameScope()


@contextmanager
def name_scope(name, scope=None):
    if scope is None:
        scope = _global_default_name_scope

    try:
        yield scope.sub_scope(name)

    finally:
        scope.parent()


def get_scope_string(scope=None):
    if scope is None:
        scope = _global_default_name_scope

    return str(scope)
