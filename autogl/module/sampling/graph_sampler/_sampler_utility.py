import typing


class ConfigurationsFilter:
    def __init__(
            self,
            rules: typing.Iterable[
                typing.Tuple[
                    typing.Sequence[str],
                    typing.Optional[typing.Callable[[typing.Any], bool]],
                    typing.Optional[typing.Callable[[typing.Any], typing.Any]],
                    typing.Any, typing.Optional[str]
                ]
            ]
    ):
        if not isinstance(rules, typing.Iterable):
            raise TypeError
        for rule in rules:
            if not (isinstance(rule, typing.Sequence) and len(rule) == 5):
                raise TypeError
        self.__filter_rules: typing.Iterable[
            typing.Tuple[
                typing.Sequence[str],
                typing.Optional[typing.Callable[[typing.Any], bool]],
                typing.Optional[typing.Callable[[typing.Any], typing.Any]],
                typing.Any, typing.Optional[str]
            ]
        ] = rules

    def filter(
            self, configurations: typing.Mapping[str, typing.Any]
    ) -> typing.Tuple[typing.Mapping[str, typing.Any], typing.Mapping[str, typing.Any]]:
        remaining_configurations: typing.MutableMapping[str, typing.Any] = dict(configurations)
        filtered_configurations: typing.MutableMapping[str, typing.Any] = dict()
        for rule in self.__filter_rules:
            if len(rule[0]) == 0:
                continue
            _matched: bool = False
            for matching_key in rule[0][::-1]:
                if matching_key in remaining_configurations:
                    _matched = True
                    __configuration_item = remaining_configurations.pop(matching_key)
                    if rule[1] not in (Ellipsis, None) and callable(rule[1]):
                        if rule[1](__configuration_item):
                            filtered_configurations[rule[0][0]] = (
                                rule[2](__configuration_item) if rule[2] is not None and callable(rule[2])
                                else __configuration_item
                            )
                    else:
                        filtered_configurations[rule[0][0]] = (
                            rule[2](__configuration_item) if rule[2] is not None and callable(rule[2])
                            else __configuration_item
                        )
            if _matched:
                if rule[0][0] not in filtered_configurations:
                    if rule[4] is Ellipsis:
                        continue
                    if rule[4] is None:
                        raise ValueError(
                            f"One of the following keys {rule[0]} exists in provided configurations "
                            f"but none of the matched values satisfies certain requirement"
                        )
                    if isinstance(rule[4], str) and len(rule[4].strip()) > 0:
                        raise ValueError(
                            f"One of the following keys {rule[0]} exists in provided configurations "
                            f"but none of the matched values satisfies certain requirement, "
                            f"the auxiliary information: {rule[4].strip()}"
                        )
            else:
                if rule[3] not in (Ellipsis, None):
                    filtered_configurations[rule[0][0]] = (
                        rule[2](rule[3]) if rule[2] is not None and callable(rule[2]) else rule[3]
                    )
                if rule[3] is Ellipsis:
                    continue
                if rule[3] is None:
                    raise KeyError
        return filtered_configurations, remaining_configurations
