import typing as _typing

_D = _typing.TypeVar('_D')


class Dataset(_typing.Iterable[_D], _typing.Sized):
    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> _typing.Iterator[_D]:
        raise NotImplementedError

    def __getitem__(self, index: int) -> _D:
        raise NotImplementedError

    def __setitem__(self, index: int, data: _D):
        raise NotImplementedError

    @property
    def train_split(self) -> _typing.Optional[_typing.Iterable[_D]]:
        raise NotImplementedError

    @property
    def val_split(self) -> _typing.Optional[_typing.Iterable[_D]]:
        raise NotImplementedError

    @property
    def test_split(self) -> _typing.Optional[_typing.Iterable[_D]]:
        raise NotImplementedError

    @property
    def train_index(self) -> _typing.Optional[_typing.AbstractSet[int]]:
        raise NotImplementedError

    @property
    def val_index(self) -> _typing.Optional[_typing.AbstractSet[int]]:
        raise NotImplementedError

    @property
    def test_index(self) -> _typing.Optional[_typing.AbstractSet[int]]:
        raise NotImplementedError

    @train_index.setter
    def train_index(self, train_index: _typing.Optional[_typing.Iterable[int]]):
        raise NotImplementedError

    @val_index.setter
    def val_index(self, val_index: _typing.Optional[_typing.Iterable[int]]):
        raise NotImplementedError

    @test_index.setter
    def test_index(self, test_index: _typing.Optional[_typing.Iterable[int]]):
        raise NotImplementedError


class _FoldsContainer:
    def __init__(
            self,
            folds: _typing.Optional[_typing.Iterable[_typing.Tuple[_typing.Sequence[int], _typing.Sequence[int]]]] = ...
    ):
        self._folds: _typing.Optional[_typing.List[_typing.Tuple[_typing.Sequence[int], _typing.Sequence[int]]]] = (
            list(folds) if isinstance(folds, _typing.Iterable) else None
        )
        if self._folds is not None and len(self._folds) == 0:
            self._folds = None

    @property
    def folds(self) -> _typing.Optional[_typing.Sequence[_typing.Tuple[_typing.Sequence[int], _typing.Sequence[int]]]]:
        if self._folds is not None and len(self._folds) == 0:
            self._folds = None
        return self._folds

    @folds.setter
    def folds(self, folds: _typing.Optional[_typing.Iterable[_typing.Tuple[_typing.Sequence[int], _typing.Sequence[int]]]]):
        self._folds: _typing.Optional[_typing.List[_typing.Tuple[_typing.Sequence[int], _typing.Sequence[int]]]] = (
            list(folds) if isinstance(folds, _typing.Iterable) else None
        )
        if self._folds is not None and len(self._folds) == 0:
            self._folds = None


class _FoldView:
    def __init__(self, folds_container: _FoldsContainer, fold_index: int):
        self._folds_container: _FoldsContainer = folds_container
        self._fold_index: int = fold_index

    @property
    def train_index(self) -> _typing.Sequence[int]:
        return self._folds_container.folds[self._fold_index][0]

    @property
    def val_index(self) -> _typing.Sequence[int]:
        return self._folds_container.folds[self._fold_index][1]


class _FoldsView(_typing.Sequence[_FoldView]):
    def __init__(self, folds_container: _FoldsContainer):
        self._folds_container = folds_container

    def __len__(self) -> int:
        return (
            len(self._folds_container.folds)
            if self._folds_container.folds is not None
            else 0
        )

    def __getitem__(self, fold_index: int) -> _FoldView:
        return _FoldView(self._folds_container, fold_index)


class InMemoryDataset(Dataset[_D]):
    def __init__(
            self, data: _typing.Iterable[_D],
            train_index: _typing.Optional[_typing.Iterable[int]] = ...,
            val_index: _typing.Optional[_typing.Iterable[int]] = ...,
            test_index: _typing.Optional[_typing.Iterable[int]] = ...
    ):
        self.__data: _typing.MutableSequence[_D] = list(data)
        self.__train_index: _typing.Optional[_typing.Iterable[int]] = (
            train_index if isinstance(train_index, _typing.Iterable) else None
        )
        self.__val_index: _typing.Optional[_typing.Iterable[int]] = (
            val_index if isinstance(val_index, _typing.Iterable) else None
        )
        self.__test_index: _typing.Optional[_typing.Iterable[int]] = (
            test_index if isinstance(test_index, _typing.Iterable) else None
        )
        self.__folds_container: _FoldsContainer = _FoldsContainer()

    @property
    def folds(self) -> _typing.Optional[_FoldsView]:
        return (
            _FoldsView(self.__folds_container)
            if (
                    self.__folds_container.folds is not None and
                    len(self.__folds_container.folds) > 0
            )
            else None
        )

    @folds.setter
    def folds(
            self,
            folds: _typing.Optional[
                _typing.Iterable[
                    _typing.Tuple[_typing.Sequence[int], _typing.Sequence[int]]
                ]
            ] = ...
    ):
        self.__folds_container.folds = folds

    def __len__(self) -> int:
        return len(self.__data)

    def __iter__(self) -> _typing.Iterator[_D]:
        return iter(self.__data)

    def __getitem__(self, index: int) -> _D:
        return self.__data[index]

    def __setitem__(self, index: int, data: _D):
        self.__data[index] = data

    def reset_dataset(self, data: _typing.Iterable[_D]):
        if not isinstance(data, _typing.Iterable):
            raise TypeError
        __data: _typing.MutableSequence[_D] = list(data)
        __preserve_info: bool = __data == len(self)
        self.__data: _typing.MutableSequence[_D] = __data
        if not __preserve_info:
            self.train_index = self.val_index = self.test_index = None

    @property
    def train_split(self) -> _typing.Optional[_typing.Iterable[_D]]:
        return (
            [self.__data[i] for i in self.__train_index]
            if isinstance(self.__train_index, _typing.Iterable) else None
        )

    @property
    def val_split(self) -> _typing.Optional[_typing.Iterable[_D]]:
        return (
            [self.__data[i] for i in self.__val_index]
            if isinstance(self.__val_index, _typing.Iterable) else None
        )

    @property
    def test_split(self) -> _typing.Optional[_typing.Iterable[_D]]:
        return (
            [self.__data[i] for i in self.__test_index]
            if isinstance(self.__test_index, _typing.Iterable) else None
        )

    @property
    def train_index(self) -> _typing.Optional[_typing.AbstractSet[int]]:
        return self.__train_index

    @property
    def val_index(self) -> _typing.Optional[_typing.AbstractSet[int]]:
        return self.__val_index

    @property
    def test_index(self) -> _typing.Optional[_typing.AbstractSet[int]]:
        return self.__test_index

    @train_index.setter
    def train_index(self, train_index: _typing.Optional[_typing.Iterable[int]]):
        if not (train_index is None or isinstance(train_index, _typing.Iterable)):
            raise TypeError
        elif train_index is None:
            self.__train_index: _typing.Optional[_typing.Iterable[int]] = None
        elif isinstance(train_index, _typing.Iterable):
            if not all([isinstance(i, int) for i in train_index]):
                raise TypeError
            if not (0 <= min(train_index) <= max(train_index) < len(self)):
                raise ValueError
            self.__train_index: _typing.Optional[_typing.Iterable[int]] = train_index

    @val_index.setter
    def val_index(self, val_index: _typing.Optional[_typing.Iterable[int]]):
        if not (val_index is None or isinstance(val_index, _typing.Iterable)):
            raise TypeError
        elif val_index is None:
            self.__val_index: _typing.Optional[_typing.Iterable[int]] = None
        elif isinstance(val_index, _typing.Iterable):
            if not all([isinstance(i, int) for i in val_index]):
                raise TypeError
            if not (0 <= min(val_index) <= max(val_index) < len(self)):
                raise ValueError
            self.__val_index: _typing.Optional[_typing.Iterable[int]] = val_index

    @test_index.setter
    def test_index(self, test_index: _typing.Optional[_typing.Iterable[int]]):
        if not (test_index is None or isinstance(test_index, _typing.Iterable)):
            raise TypeError
        elif test_index is None:
            self.__test_index: _typing.Optional[_typing.Set[int]] = None
        elif isinstance(test_index, _typing.Iterable):
            if not all([isinstance(i, int) for i in test_index]):
                raise TypeError
            if not (0 <= min(test_index) <= max(test_index) < len(self)):
                raise ValueError
            self.__test_index: _typing.Optional[_typing.Iterable[int]] = test_index
