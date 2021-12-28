import typing as _typing


class CanonicalEdgeType(_typing.Sequence[str]):
    def __init__(self, source_node_type: str, relation_type: str, target_node_type: str):
        if not isinstance(source_node_type, str):
            raise TypeError
        elif ' ' in source_node_type:
            raise ValueError
        if not isinstance(relation_type, str):
            raise TypeError
        elif ' ' in relation_type:
            raise ValueError
        if not isinstance(target_node_type, str):
            raise TypeError
        elif ' ' in target_node_type:
            raise ValueError
        self.__source_node_type: str = source_node_type
        self.__relation_type: str = relation_type
        self.__destination_node_type: str = target_node_type

    @property
    def source_node_type(self) -> str:
        return self.__source_node_type

    @property
    def relation_type(self) -> str:
        return self.__relation_type

    @property
    def target_node_type(self) -> str:
        return self.__destination_node_type

    def __eq__(self, other):
        if not (isinstance(other, CanonicalEdgeType) or isinstance(other, _typing.Sequence)):
            return False
        elif isinstance(other, _typing.Sequence):
            if not (len(other) == 3 and all([(isinstance(t, str) and ' ' not in t) for t in other])):
                raise TypeError
            return (
                    other[0] == self.source_node_type and
                    other[1] == self.relation_type and
                    other[2] == self.target_node_type
            )
        elif isinstance(other, CanonicalEdgeType):
            return (
                    other.source_node_type == self.source_node_type and
                    other.relation_type == self.relation_type and
                    other.target_node_type == self.target_node_type
            )

    def __getitem__(self, index: int):
        return (self.source_node_type, self.relation_type, self.target_node_type)[index]

    def __len__(self) -> int:
        return 3
