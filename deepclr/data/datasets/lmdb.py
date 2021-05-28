from typing import List, NoReturn, Optional

import math

from dataflow import DataFlow, MapData, LMDBData, LMDBSerializer

from .utils import MapDataList


class LMDBSequenceData(LMDBData):
    """Read consecutive sequences from LMDB file."""
    def __init__(self, lmdb_path: str, seq_length: int, seq_step: int = 1, sort_key: str = None,
                 reverse: bool = False, shuffle: bool = False, keys: Optional[List[str]] = None):
        super().__init__(lmdb_path, shuffle=True, keys=keys)
        self._shuffle = shuffle

        # sort keys
        if sort_key:
            if isinstance(sort_key, bool):
                self.keys.sort(reverse=reverse)
            else:
                self.keys.sort(key=sort_key, reverse=reverse)

        self.seq_length = int(seq_length)
        assert self.seq_length > 0

        self.seq_step = int(seq_step)
        assert self.seq_step > 0

    def __len__(self):
        return math.ceil((super().__len__() - self.seq_length + 1) / self.seq_step)

    def __iter__(self):
        with self._guard:
            key_indices = list(range(0, len(self.keys) - self.seq_length + 1, self.seq_step))
            if self._shuffle:
                self.rng.shuffle(key_indices)

            for start_idx in key_indices:
                keys = [self.keys[i] for i in range(start_idx, start_idx + self.seq_length)]
                yield [[k, self._txn.get(k)] for k in keys]


class LMDBSortedData(LMDBData):
    """Read sorted data from LMDB file."""
    def __init__(self, lmdb_path: str, sort_key: Optional[str] = None, reverse: bool = False,
                 keys: Optional[List[str]] = None):
        # initialize LMDBData with shuffle to force creating keys
        super().__init__(lmdb_path, shuffle=True, keys=keys)
        self._shuffle = False

        # sort keys
        if sort_key:
            self.keys.sort(key=sort_key, reverse=reverse)
        else:
            self.keys.sort(reverse=reverse)

    def __iter__(self):
        with self._guard:
            for k in self.keys:
                v = self._txn.get(k)
                yield [k, v]


class LMDBSequenceSerializer(LMDBSerializer):
    """Read consecutive sequences from LMDB file."""
    @staticmethod
    def load(path: str, shuffle: bool = True) -> NoReturn:
        raise NotImplementedError("Use LMDBSequenceSerializer.load_sequence()")

    @staticmethod
    def load_sequence(path: str, seq_length: int, seq_step: int = 1, reverse: bool = False, shuffle: bool = True) \
            -> DataFlow:
        df = LMDBSequenceData(path, seq_length, seq_step=seq_step, reverse=reverse, shuffle=shuffle)
        return MapDataList(df, LMDBSequenceSerializer._deserialize_lmdb)


class LMDBSortedSerializer(LMDBSerializer):
    """Read sorted data from LMDB file."""
    @staticmethod
    def load(path: str, shuffle: bool = True) -> NoReturn:
        raise NotImplementedError("Use LMDBSortedSerializer.load_sorted()")

    @staticmethod
    def load_sorted(path: str, reverse: bool = False) -> DataFlow:
        df = LMDBSortedData(path, reverse=reverse)
        return MapData(df, LMDBSortedSerializer._deserialize_lmdb)
