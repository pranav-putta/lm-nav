from typing import Any


class BaseDataTransform:
    def __call__(self, x) -> Any:
        return x


class ReverseTurnsTransform(BaseDataTransform):
    def __call__(self, x) -> Any:
        import pdb

        pdb.set_trace()
        return x
