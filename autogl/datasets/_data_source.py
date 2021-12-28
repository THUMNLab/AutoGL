import os
import typing as _typing


class OnlineDataSource:
    @property
    def _raw_directory(self) -> str:
        return os.path.join(self.__path, "raw")

    @property
    def _processed_directory(self) -> str:
        return os.path.join(self.__path, "processed")

    @property
    def _raw_filenames(self) -> _typing.Iterable[str]:
        raise NotImplementedError

    @property
    def _processed_filenames(self) -> _typing.Iterable[str]:
        raise NotImplementedError

    @property
    def _raw_file_paths(self) -> _typing.Iterable[str]:
        return [
            os.path.join(self._raw_directory, raw_filename)
            for raw_filename in self._raw_filenames
        ]

    @property
    def _processed_file_paths(self) -> _typing.Iterable[str]:
        return [
            os.path.join(self._processed_directory, processed_filename)
            for processed_filename in self._processed_filenames
        ]

    @classmethod
    def __files_exist(cls, files: _typing.Iterable[str]) -> bool:
        return all([os.path.exists(file) for file in files])

    @classmethod
    def __make_directory(cls, path):
        os.makedirs(os.path.expanduser(os.path.normpath(path)), exist_ok=True)

    def _fetch(self):
        raise NotImplementedError

    def __fetch(self):
        if not self.__files_exist(self._raw_file_paths):
            self.__make_directory(self._raw_directory)
            self._fetch()

    def _process(self):
        raise NotImplementedError

    def __preprocess(self):
        if not self.__files_exist(self._processed_file_paths):
            self.__make_directory(self._processed_directory)
            self._process()

    def __getitem__(self, index: int) -> _typing.Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __init__(self, path: str):
        self.__path: str = os.path.expanduser(os.path.normpath(path))
        self.__fetch()
        self.__preprocess()
