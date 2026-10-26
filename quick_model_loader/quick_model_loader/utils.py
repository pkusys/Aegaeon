from collections import OrderedDict
from collections.abc import MutableMapping

from typing import (
    Optional,
    Tuple,
    TypeVar,
    Generic,
    Iterator,
    Any,
)

K = TypeVar("K")
V = TypeVar("V")


class LRUDict(MutableMapping, Generic[K, V]):
    def __init__(self, *args: Tuple[Any, Any], **kwargs: Any):
        self.store = OrderedDict()
        self.update(
            dict(*args, **kwargs)
        )  # Use the update method to initialize the dictionary

    def __getitem__(self, key: K) -> V:
        if key in self.store:
            self.store.move_to_end(key)
        return self.store[key]

    def __setitem__(self, key: K, value: V) -> None:
        self.store[key] = value
        self.store.move_to_end(key)

    def __delitem__(self, key: K) -> None:
        del self.store[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.store})"

    def lru(self) -> Tuple[Optional[K], Optional[V]]:
        try:
            key, value = next(iter(self.store.items()))
            return key, value
        except:
            return None, None
