class SklearnTrainingSession(object):
    _session_stack = []

    def __init__(self, clazz, allow_children=True):
        self.allow_children = allow_children
        self.clazz = clazz
        self._parent = None

    def __enter__(self):
        if len(SklearnTrainingSession._session_stack) > 0:
            self._parent = SklearnTrainingSession._session_stack[-1]
            self.allow_children = (
                SklearnTrainingSession._session_stack[-1].allow_children and self.allow_children
            )
        SklearnTrainingSession._session_stack.append(self)
        print(self._session_stack)
        return self

    def __exit__(self, tp, val, traceback):
        SklearnTrainingSession._session_stack.pop()

    def should_log(self):
        return (self._parent is None) or (
            self._parent.allow_children and self._parent.clazz != self.clazz
        )


class Parent:
    pass


class Children:
    pass


with SklearnTrainingSession(Parent) as p:
    with SklearnTrainingSession(Children) as c:
        pass
