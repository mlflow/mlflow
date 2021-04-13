from typing import TypeVar, List

T = TypeVar("T")


class PagedList(List[T]):
    def __init__(self, items: List[T], token):
        super().__init__(items)
        self.token = token
