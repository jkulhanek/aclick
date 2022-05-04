import typing as t

import click as _click
from click.types import ParamType

from .utils import _ClassArgument, _is_class, get_class_name, parse_class_structure


class Tuple(_click.Tuple):
    """The default behavior of Click is to apply a type on a value directly.
    This works well in most cases, except for when `nargs` is set to a fixed
    count and different types should be used for different items.  In this
    case the :class:`Tuple` type can be used.  This type can only be used
    if `nargs` is set to a fixed number.
    This can be selected by using a Python tuple literal as a type.

    As opposed to Click's original Tuple type, this type supports inline
    class parsing. A complicated structure of classes can be parsed
    using this type. See :func:`parse_class_structure`.

    :param types: a list of types that should be used for the tuple items.
    """

    def __init__(self, types: t.Sequence[t.Union[t.Type, ParamType]]) -> None:
        super().__init__([convert_type(ty) for ty in types])


class List(_click.ParamType):
    """This type parses a list of values separated by commas as individual
    instances.
    This type supports inline
    class parsing. A complicated structure of classes can be parsed
    using this type. See :func:`parse_class_structure`.


    :param inner_type: Type of the objects in the list.
    """

    name = "list"

    def __init__(self, inner_type: t.Type):
        self.inner_type = convert_type(inner_type)
        self.name = f"list of {self.inner_type.name}s"

    def __repr__(self) -> str:
        return f"LIST[{repr(self.inner_type)}]"

    def convert(
        self,
        value: t.Any,
        param: t.Optional[_click.Parameter],
        ctx: t.Optional[_click.Context],
    ) -> t.Any:
        assert isinstance(value, str)
        values = [str(x) for x in _ClassArgument.from_str(f"list({value})").args]
        return list(map(self.inner_type, values))


class ClassUnion(_click.ParamType):
    """This type parses the string value as the instance of
    one of specified classes. A complicated structure of
    classes can be parsed using this type.
    See :func:`parse_class_structure`.

    :param classes: List of classes to parse into.
    :param known_classes: List of classes supported at deeper levels (e.g. as property values).
    """

    name = "class union"

    def __init__(
        self, classes: t.List[t.Type], known_classes: t.Optional[t.List[t.Type]] = None
    ):
        self.classes = classes
        self.known_classes = known_classes

    def __repr__(self) -> str:
        return "|".join(sorted(get_class_name(x) for x in self.classes))

    def convert(
        self,
        value: t.Any,
        param: t.Optional[_click.Parameter],
        ctx: t.Optional[_click.Context],
    ) -> t.Any:

        assert isinstance(value, str)
        value = parse_class_structure(value, self.classes, self.known_classes)
        return value


def convert_type(ty: t.Optional[t.Any], default: t.Optional[t.Any] = None) -> ParamType:
    if ty is not None and _is_class(ty):
        return ClassUnion([ty])
    if ty is not None and getattr(ty, "__origin__", None) is t.Union:
        classes = [c for c in ty.__args__ if _is_class(c)]
        return ClassUnion(classes)
    if ty is not None and getattr(ty, "__origin__", None) is list:
        return List(ty.__args__[0])
    return _click.types.convert_type(ty, default)
