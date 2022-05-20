import inspect
import typing as t
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import click as _click
from click.types import ParamType

from .utils import (
    _ClassArgument,
    _full_signature,
    _is_class,
    _wrap_fn_to_allow_kwargs_instead_of_args,
    build_examples,
    get_class_name,
    get_class_name as _get_class_name,
    parse_class_structure,
    ParseClassStructureError,
)

if TYPE_CHECKING:
    from .core import Command


class ParameterGroup:
    def __init__(self, full_name: t.Optional[str]):
        self.full_name = full_name
        self.name = full_name.replace(".", "_") if full_name is not None else None

    def assert_is_supported(self, command: "Command") -> "ParameterGroup":
        return self  # pragma: no cover

    def get_params(
        self, ctx: _click.Context
    ) -> t.Iterable[t.Union["ParameterGroup", _click.Parameter]]:
        return []  # pragma: no cover

    def handle_parse_group_result(self, ctx: _click.Context):
        pass  # pragma: no cover

    def _handle_parse_group_result_for_class(self, cls: t.Type, ctx: _click.Context):
        assert hasattr(ctx.command, "build_click_parameter")
        local_params = dict()
        for param in _full_signature(cls).parameters.values():
            param_name = f"{self.full_name}.{param.name}"
            click_param = getattr(ctx.command, "build_click_parameter")(
                param_name, param
            )
            if click_param is not None and click_param.name is not None:
                local_params[param.name] = ctx.params.pop(click_param.name)
        return _wrap_fn_to_allow_kwargs_instead_of_args(cls)(**local_params)

    def _get_params_for_class(self, cls: t.Type, ctx: _click.Context):
        assert hasattr(ctx.command, "build_click_parameter")
        for param in _full_signature(cls).parameters.values():
            param_name = f"{self.full_name}.{param.name}"
            click_param = getattr(ctx.command, "build_click_parameter")(
                param_name, param
            )
            if click_param is not None:
                yield click_param

    def _get_or_store_first_value(
        self, ctx: _click.Context, value: t.Any = inspect._empty
    ):
        assert self.name is not None
        param_groups = ctx.ensure_object(_AClickContext).param_group_contexts
        if self.name in param_groups:
            return param_groups[self.name]
        elif value is not inspect._empty:
            param_groups[self.name] = value
            return value

    def _get_first_value(self, ctx: _click.Context):
        assert self.name is not None
        param_groups = ctx.ensure_object(_AClickContext).param_group_contexts
        return param_groups[self.name]


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
        if isinstance(value, list):
            return value
        assert isinstance(value, str)
        values = [str(x) for x in _ClassArgument.from_str(f"list({value})").args]
        return [self.inner_type.convert(x, param, ctx) for x in values]


class ClassUnion(_click.ParamType):
    """This type parses the string value as the instance of
    one of specified classes. A complicated structure of
    classes can be parsed using this type.
    See :func:`parse_class_structure`.

    :param classes: List of classes to parse into.
    :param known_classes: List of classes supported at deeper levels (e.g. as property values).

    :param num_examples_on_error: Number of examples to show if there was an error during parsing.
                                  If the value is -1, all possible examples are shown. Default is 3.
    """

    name = "class union"

    def __init__(
        self,
        classes: t.List[t.Type],
        known_classes: t.Optional[t.List[t.Type]] = None,
        num_examples_on_error: int = 3,
    ):
        self.classes = classes
        self.known_classes = known_classes
        self.num_examples_on_error = num_examples_on_error

    def __repr__(self) -> str:
        return "|".join(sorted(get_class_name(x) for x in self.classes))

    def convert(
        self,
        value: t.Any,
        param: t.Optional[_click.Parameter],
        ctx: t.Optional[_click.Context],
    ) -> t.Any:

        if any(isinstance(value, x) for x in self.classes):
            return value

        assert isinstance(value, str)
        aclick_ctx = (
            _AClickContext() if ctx is None else ctx.ensure_object(_AClickContext)
        )
        try:
            value = parse_class_structure(value, self.classes, self.known_classes)
        except ParseClassStructureError as err:
            msg = str(err)
            if self.num_examples_on_error != 0:
                msg += "\n\nExamples of allowed values:\n"
                union_type: t.Type = t.cast(t.Type, t.Union[tuple(self.classes)])
                num_examples_on_error = (
                    self.num_examples_on_error
                    if self.num_examples_on_error >= 0
                    else None
                )
                msg += "\n\n".join(
                    build_examples(
                        union_type,
                        use_dashes=aclick_ctx.use_dashes,
                        limit=num_examples_on_error,
                    )
                )

            raise _click.ClickException(msg) from err
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


class ClassHierarchicalOption(ParameterGroup):
    def __init__(self, name: str, type: t.Type):
        super().__init__(name)
        self.class_type = type

    def assert_is_supported(self, command: "Command") -> "ClassHierarchicalOption":
        assert self.full_name is not None
        assert hasattr(command, "_assert_signature_is_supported")
        command._assert_signature_is_supported(
            _full_signature(self.class_type), self.full_name + "."
        )
        return self

    def get_params(self, ctx: _click.Context):
        return self._get_params_for_class(self.class_type, ctx)

    def handle_parse_group_result(self, ctx: _click.Context) -> t.Any:
        assert self.name is not None

        ctx.params[self.name] = self._handle_parse_group_result_for_class(
            self.class_type, ctx
        )


class UnionTypeHierarchicalOption(_click.Option, ParameterGroup):
    def __init__(
        self,
        parameter_name: str,
        name: str,
        classes: t.List[t.Type],
        required: bool = False,
        get_class_name: t.Optional[t.Callable[[t.Type], str]] = None,
    ):
        ParameterGroup.__init__(self, name)
        self.classes = classes
        self.get_class_name = get_class_name or _get_class_name
        self._init_click_option(parameter_name, name, type, required)

    def assert_is_supported(self, command: "Command") -> "UnionTypeHierarchicalOption":
        assert self.full_name is not None
        assert hasattr(command, "_assert_signature_is_supported")
        for c in self.classes:
            command._assert_signature_is_supported(
                _full_signature(c), self.full_name + "."
            )
        return self

    def _init_click_option(self, parameter_name, name, type, required):
        opt_name = [f"--{parameter_name}", name.replace(".", "_")]

        if required:
            kwargs: t.Dict[str, t.Any] = dict(required=True)
        else:
            kwargs = dict(default=None)
        _click.Option.__init__(
            self,
            opt_name,
            is_eager=True,
            type=_click.types.Choice(
                [self.get_class_name(x) for x in self.classes], case_sensitive=False
            ),
            help=f"Set {name} to a {self.get_class_name(type)} instance",
            **kwargs,
        )

    def get_params(self, ctx: _click.Context):
        assert isinstance(self.name, str)

        current_value = self._get_or_store_first_value(
            ctx, ctx.params.get(self.name, inspect._empty)
        )
        if current_value is not None:
            assert isinstance(current_value, str)
            class_map = {self.get_class_name(x): x for x in self.classes}

            if current_value not in class_map:
                supported_classes_names = ", ".join(
                    sorted(map(self.get_class_name, self.classes))
                )
                raise _click.ClickException(
                    f'Class with name "{current_value}" was not found in the set of supported classes {{{supported_classes_names}}}'
                )
            assert current_value in class_map
            class_type = class_map[current_value]
            return self._get_params_for_class(class_type, ctx)
        return []

    def handle_parse_group_result(self, ctx: _click.Context) -> t.Any:
        assert isinstance(self.name, str)

        ctx.params.pop(self.name)
        control_parameter_value = self._get_first_value(ctx)
        if control_parameter_value is not None:
            assert isinstance(control_parameter_value, str)
            class_map = {self.get_class_name(x): x for x in self.classes}
            assert control_parameter_value in class_map
            class_type = class_map[control_parameter_value]
            ctx.params[self.name] = self._handle_parse_group_result_for_class(
                class_type, ctx
            )


class OptionalTypeHierarchicalOption(_click.Option, ParameterGroup):
    def __init__(
        self,
        parameter_name: str,
        name: str,
        type: t.Type,
        required: bool = False,
        get_class_name: t.Optional[t.Callable[[t.Type], str]] = None,
    ):
        ParameterGroup.__init__(self, name)
        self.optional_type = type
        self.get_class_name = get_class_name or _get_class_name
        self._init_click_option(parameter_name, name, type, required)

    def _init_click_option(self, parameter_name, name, type, required):
        opt_name = [f"--{parameter_name}", name.replace(".", "_")]

        if required:
            opt_name = [
                f"{x}/--no-{x[2:]}" if x.startswith("--") else x for x in opt_name[:-1]
            ] + [opt_name[-1]]
            _click.Option.__init__(
                self,
                opt_name,
                type=_click.types.BoolParamType(),
                default=False,
                is_eager=True,
                required=True,
                help=f"Set {name} to a {self.get_class_name(type)} instance",
            )
            # NOTE: this is fix for 8.0.x version of click
            self.default = None
        else:
            _click.Option.__init__(
                self,
                opt_name,
                is_flag=True,
                default=False,
                is_eager=True,
                help=f"Set {name} to a {self.get_class_name(type)} instance",
            )

    def assert_is_supported(
        self, command: "Command"
    ) -> "OptionalTypeHierarchicalOption":
        assert self.full_name is not None
        assert hasattr(command, "_assert_signature_is_supported")
        command._assert_signature_is_supported(
            _full_signature(self.optional_type), self.full_name + "."
        )
        return self

    def get_params(self, ctx: _click.Context):
        assert isinstance(self.name, str)

        current_value = self._get_or_store_first_value(
            ctx, ctx.params.get(self.name, inspect._empty)
        )
        if current_value is not None and current_value:
            return self._get_params_for_class(self.optional_type, ctx)
        return []

    def handle_parse_group_result(self, ctx: _click.Context) -> t.Any:
        assert isinstance(self.name, str)

        ctx.params.pop(self.name)
        control_parameter_value = self._get_first_value(ctx)
        assert isinstance(control_parameter_value, bool)
        if control_parameter_value:
            ctx.params[self.name] = self._handle_parse_group_result_for_class(
                self.optional_type, ctx
            )
        else:
            ctx.params[self.name] = None


@dataclass
class _AClickContext:
    param_groups: t.List[ParameterGroup] = field(default_factory=list)
    param_group_contexts: t.Dict[str, t.Any] = field(default_factory=dict)
    use_dashes: bool = True
    configuration_file_loaded: t.Optional[str] = None
