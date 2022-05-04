import copy
import inspect
import re
import typing as t
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

import click as _click

from .types import ClassUnion, List, Tuple
from .utils import (
    _full_signature,
    _get_help_text,
    _is_class,
    _wrap_fn_to_allow_kwargs_instead_of_args,
    get_class_name,
    Literal,
    SignatureWithDescription,
)


_SUPPORTED_TYPES = [str, float, int, bool]


@dataclass
class AClickParameter:
    parameter_type: t.Type
    parameter_name: str
    parameter_default: t.Any = None
    children: t.List["AClickParameter"] = field(default_factory=list)
    is_hierarchical: bool = False
    positional_only: bool = False
    keyword_only: bool = False
    expanded_parameter_type: t.Optional[t.Type] = None
    description: t.Optional[str] = None

    @property
    def name(self):
        return self.parameter_name


class ParameterRenamer(metaclass=ABCMeta):
    """A base class used as parameter renamer for a :class:`Command`.
    The derived class must implement the call method that
    maps from the original name to a new name.
    """

    @abstractmethod
    def __call__(self, name: str) -> str:
        """Maps the parameter name into a new name.

        :param name: Original name of the parameter.
        :return: The new name of the parameter.
        """
        raise NotImplementedError()  # pragma: no cover


@dataclass
class AClickContext:
    params: t.Optional[t.List[AClickParameter]] = None


class Command(_click.Command):
    """
    Command class extend `click.Command` class and it automatically
    parses callback's signature and generates the parameters.
    It also supports dynamic parsing, where based on some
    parameter, other parameters may be added, which is used in the
    hierarchical parsing (see `is_hierarchical`).

    :param name: the name of the command to use unless a group overrides it.
    :param context_settings: an optional dictionary with defaults that are
                             passed to the context object.
    :param callback: the callback to invoke.  This is optional.
    :param signature: signature to be used for automatic parameter registration.
                      If no signature is specified, it is parsed automatically
                      from the callback.
    :param hierarchical: whether to parse complex types using hierarchical
                         parsing instead of inline parsing. In hierarchical
                         parsing the complex class options are expanded into
                         individual options corresponding to individual
                         properties. Note that some types are not supported
                         with hierarchical parsing. Default is to use inline
                         parsing.
    :param map_parameter_name: a function that maps from parameter paths
                               (e.g. param1.property1.subproperty2) into
                               parameter name used in parsing.
    :param params: the parameters to register with this command. This can
                   be either :class:`Option` or :class:`Argument` objects.
                   These parameters override automatically generated params
                   if they share the same name.
    :param help: the help string to use for this command. The callback's
                 docstring is parsed if no help is passed.
    :param epilog: like the help string but it's printed at the end of the
                   help page after everything else.
    :param short_help: the short help to use for this command.  This is
                       shown on the command listing of the parent command.
                       The callback's docstring is parsed if no help is passed.
    :param add_help_option: by default each command registers a ``--help``
                            option.  This can be disabled by this parameter.
    :param no_args_is_help: this controls what happens if no arguments are
                            provided.  This option is disabled by default.
                            If enabled this will add ``--help`` as argument
                            if no arguments are passed
    :param hidden: hide this command from help outputs.
    :param deprecated: issues a message indicating that
                             the command is deprecated.
    """

    def __init__(
        self,
        *args,
        signature: t.Optional[inspect.Signature] = None,
        hierarchical: bool = False,
        map_parameter_name: t.Optional[ParameterRenamer] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if signature is None:
            if self.callback is None:
                raise ValueError("Either signature or callback must be specified")
            signature = _full_signature(self.callback)
            self.callback = _wrap_fn_to_allow_kwargs_instead_of_args(self.callback)
        self.callback_signature: inspect.Signature = signature
        self.hierarchical = hierarchical
        self.map_parameter_name = map_parameter_name
        self._assert_signature_is_supported(self.callback_signature)
        if (self.help is None or self.short_help is None) and isinstance(
            signature, SignatureWithDescription
        ):
            short_help, help = _get_help_text(signature)
            self.short_help = (
                self.short_help if self.short_help is not None else short_help
            )
            self.help = self.help if self.help is not None else help

    def _wrap_context(self, ctx) -> t.Tuple[_click.Context, AClickContext]:
        aclick_context = getattr(ctx, "__aclick_context__", None)
        if aclick_context is None:
            aclick_context = AClickContext()
            setattr(ctx, "__aclick_context__", aclick_context)
        return ctx, aclick_context

    def add_conditional_params(
        self, ctx: _click.Context, aclick_ctx: AClickContext
    ) -> bool:
        params_added = False

        def walk(path, p):
            nonlocal params_added
            name = "_".join(path + [p.parameter_name]).replace("-", "_")
            if (
                self.hierarchical
                and p.is_hierarchical
                and getattr(p.parameter_type, "__origin__", None) is t.Union
                and p.expanded_parameter_type is None
            ):
                # Switch the type if not already expanded
                if name in ctx.params:
                    class_name = ctx.params.get(name)
                    supported_classes = {
                        x for x in p.parameter_type.__args__ if _is_class(x)
                    }
                    parameter_type = next(
                        (
                            x
                            for x in supported_classes
                            if get_class_name(x) == class_name
                        ),
                        None,
                    )
                    if parameter_type is None:
                        supported_classes_names = ", ".join(
                            sorted(map(get_class_name, supported_classes))
                        )
                        raise _click.ClickException(
                            f'Class with name "{class_name}" was not found in the set of supported classes {{{supported_classes_names}}}'
                        )
                    p.expanded_parameter_type = parameter_type
                    p.children = _parse_signature(
                        _full_signature(parameter_type), hierarchical=self.hierarchical
                    )
                    params_added = True

            for pc in p.children:
                walk(path + [p.parameter_name], pc)

        if aclick_ctx.params is not None:
            for p in aclick_ctx.params:
                walk([], p)
        return params_added

    def build_click_parameter(
        self,
        option_name: t.Sequence[str],
        parameter_type: t.Type,
        default: t.Any,
        help: t.Optional[str],
        is_argument: bool,
    ) -> t.Optional[_click.Parameter]:
        kwargs = dict()
        if default is not _empty:
            kwargs["default"] = default
            kwargs["required"] = False
        else:
            kwargs["required"] = True
        if help is not None and not is_argument:
            kwargs["help"] = help
        original_parameter_type = parameter_type
        cls = _click.Argument if is_argument else _click.Option

        # Handle optional
        if (
            getattr(parameter_type, "__origin__", None) is t.Union
            and len(parameter_type.__args__) == 2
            and type(None) in parameter_type.__args__
        ):
            parameter_type = next(
                x for x in parameter_type.__args__ if x not in (type(None),)
            )

        if _is_class(parameter_type):
            kwargs["type"] = ClassUnion([parameter_type])
            return cls(option_name, **kwargs)

        if getattr(parameter_type, "__origin__", None) is Literal:
            values = parameter_type.__args__
            if all(isinstance(x, str) for x in values):
                kwargs["type"] = _click.types.Choice(values, case_sensitive=False)
                return cls(option_name, **kwargs)

        if getattr(parameter_type, "__origin__", None) is t.Union:
            # Handle union type
            nonnull_args = [
                x for x in parameter_type.__args__ if x not in (type(None),)
            ]
            all_classes = all(map(_is_class, nonnull_args))
            if not all_classes:
                raise ValueError(
                    f"Parameter with parameter name {option_name[-1]} of type {original_parameter_type} is not supported. "
                    "Only unions of all classes are supported."
                )

            types = [x for x in parameter_type.__args__ if _is_class(x)]
            kwargs["type"] = ClassUnion(types)
            return cls(option_name, **kwargs)

        if getattr(parameter_type, "__origin__", None) is list:
            kwargs["type"] = List(parameter_type.__args__[0])
            return cls(option_name, **kwargs)
        if getattr(parameter_type, "__origin__", None) is tuple:
            kwargs["type"] = Tuple(list(parameter_type.__args__))
            return cls(option_name, **kwargs)
        if parameter_type in _SUPPORTED_TYPES:
            kwargs["type"] = parameter_type
            if parameter_type is bool and not is_argument:
                kwargs["type"] = _click.types.BoolParamType()
                option_name = [
                    f"{x}/--no-{x[2:]}" if x.startswith("--") else x
                    for x in option_name[:-1]
                ] + [option_name[-1]]
                option = _click.Option(option_name, **kwargs)
                if "default" not in kwargs:
                    # NOTE: this is fix for 8.0.x version of click
                    option.default = None
                return option
            else:
                return cls(option_name, **kwargs)
        return None

    def _assert_signature_is_supported(self, signature: inspect.Signature) -> None:
        aclick_params = _parse_signature(signature)

        def walk(path, p):
            output_parameter_name = "_".join(path + [p.parameter_name])
            if p.parameter_type is _empty:
                raise ValueError(
                    f'Parameter named "{output_parameter_name}" does not have it\'s type specified.'
                )

            if p.positional_only:
                if len(path) > 0:
                    raise ValueError(
                        f'Positional arguments are allowed only on top level. Failed argument name is "{output_parameter_name}".'
                    )
                assert not self.hierarchical or not p.is_hierarchical, (
                    f"Hierarchical arguments (name: {output_parameter_name}) are not supported. "
                    "Only simple types can be used in hierarchical parsing"
                )
                parameter = self.build_click_parameter(
                    [output_parameter_name],
                    p.parameter_type,
                    p.parameter_default,
                    help=p.description,
                    is_argument=True,
                )
                if parameter is None:
                    raise ValueError(
                        f"Argument with name {output_parameter_name} of type {p.parameter_type} could not be converted to a click Argument"
                    )

            elif self.hierarchical and p.is_hierarchical:
                # For hierarchical, we support only union of classes or optional
                if getattr(p.parameter_type, "__origin__", None) is t.Union:
                    nonnull_args = [
                        x for x in p.parameter_type.__args__ if x not in (type(None),)
                    ]
                    all_classes = all(map(_is_class, nonnull_args))
                    if not all_classes:
                        raise ValueError(
                            f"Parameter with parameter name {output_parameter_name} of type {p.parameter_type} is not supported. "
                            "Only unions of all classes are supported."
                        )

                    for x in nonnull_args:
                        for pc in _parse_signature(
                            _full_signature(x), hierarchical=self.hierarchical
                        ):
                            walk(path + [p.parameter_name, pc.parameter_name], pc)
                elif _is_class(p.parameter_type):
                    for pc in _parse_signature(_full_signature(p.parameter_type)):
                        walk(path + [p.parameter_name, pc.parameter_name], pc)
            else:
                parameter = self.build_click_parameter(
                    [f"--{output_parameter_name}", output_parameter_name],
                    p.parameter_type,
                    p.parameter_default,
                    help=p.description,
                    is_argument=False,
                )
                if parameter is None:
                    raise ValueError(
                        f"Parameter with parameter name {output_parameter_name} of type {p.parameter_type} could not be converted to a click Option"
                    )

        for p in aclick_params:
            walk([], p)

    def _get_click_parameters(self, parameters: t.List[AClickParameter]):
        def walk(path, p):
            kwargs: t.Dict[str, t.Any] = dict()
            children = p.children
            output_parameter_name = "_".join(path + [p.parameter_name])
            if p.positional_only:
                opt_name = [output_parameter_name]
            else:
                name = "-".join(path + [p.parameter_name]).replace("_", "-")
                name = ".".join(path + [p.parameter_name])
                if self.map_parameter_name is not None:
                    name = self.map_parameter_name(name)
                name = name.replace(".", "-").replace("_", "-")
                opt_name = [f"--{name}", output_parameter_name]
            parameter_type = p.expanded_parameter_type or p.parameter_type

            if not p.positional_only and self.hierarchical and p.is_hierarchical:
                original_parameter_type = p.parameter_type
                if (
                    getattr(original_parameter_type, "__origin__", None) is t.Union
                    and sum(1 for x in original_parameter_type.__args__ if _is_class(x))
                    > 1
                ):
                    values = [
                        get_class_name(x)
                        for x in original_parameter_type.__args__
                        if _is_class(x)
                    ]
                    kwargs["type"] = _click.types.Choice(values, case_sensitive=False)
                    kwargs["is_eager"] = True
                    yield _click.Option(opt_name, **kwargs)
            else:
                parameter = self.build_click_parameter(
                    opt_name,
                    parameter_type,
                    p.parameter_default,
                    help=p.description,
                    is_argument=p.positional_only,
                )
                assert (
                    parameter is not None
                ), f"Parameter with parameter name {output_parameter_name} of type {p.parameter_type} could not be converted to a click Parameter"
                yield parameter

            if not p.positional_only and self.hierarchical and p.is_hierarchical:
                for pc in children:
                    yield from walk(path + [p.parameter_name], pc)

        for p in parameters:
            yield from walk([], p)

    def get_params(self, ctx: _click.Context):
        ctx, aclick_ctx = self._wrap_context(ctx)
        assert (
            aclick_ctx.params is not None
        ), "parse_args method was not called before get_params method"

        aclick_params = aclick_ctx.params
        params = super().get_params(ctx)
        existing_names = {x.name for x in params}
        params.extend(
            x
            for x in self._get_click_parameters(aclick_params)
            if x.name not in existing_names
        )
        return params

    def parse_args(self, ctx: _click.Context, args: t.List[str]) -> t.List[str]:
        original_args = args
        ctx, aclick_ctx = self._wrap_context(ctx)
        aclick_ctx.params = _parse_signature(
            self.callback_signature, hierarchical=self.hierarchical
        )
        if self.hierarchical:
            arguments_added = True
            while arguments_added:
                local_ctx = copy.deepcopy(ctx)
                local_ctx.ignore_unknown_options = True
                arguments_added = False
                parser = self.make_parser(local_ctx)
                opts, args, param_order = parser.parse_args(
                    args=copy.deepcopy(original_args)
                )

                params = self.get_params(local_ctx)
                for param in _click.core.iter_params_for_processing(
                    param_order, params
                ):
                    value, args = param.handle_parse_result(local_ctx, opts, args)

                # Add new params
                arguments_added = self.add_conditional_params(local_ctx, aclick_ctx)
        result = super().parse_args(ctx, original_args)

        # Bind hierarchical parameters to correct types
        if self.hierarchical:
            params = ctx.params
            remove_parameters: t.Set[str] = set()

            def build_obj(path, p, used_parameters):
                name = "_".join(path + [p.parameter_name]).replace("-", "_")
                if not p.is_hierarchical or not self.hierarchical:
                    if name in ctx.params:
                        used_parameters.add(name)
                    return ctx.params.get(name, _empty)

                kwargs = dict()
                for pc in p.children:
                    kwargs[pc.parameter_name] = build_obj(
                        path + [p.parameter_name], pc, used_parameters
                    )
                parameter_type = p.expanded_parameter_type or p.parameter_type
                if _is_optional(parameter_type) and name not in params:
                    # If no class was selected, we return default
                    return (
                        None if p.parameter_default is _empty else p.parameter_default
                    )

                assert (
                    getattr(parameter_type, "__origin__", None) is not t.Union
                ), f'Parameter with name "{name}" was not expanded'
                return parameter_type(**kwargs)

            keep_parameters = set()
            for p in aclick_ctx.params:
                if self.hierarchical and p.is_hierarchical:
                    assert p.parameter_name is not None
                    assert not p.positional_only
                    params[p.parameter_name] = build_obj([], p, remove_parameters)
                    keep_parameters.add(p.name)
            remove_parameters.difference_update(keep_parameters)
            for pr in remove_parameters:
                params.pop(pr)
        return result


class Group(_click.Group, Command):
    @t.overload
    def command(self, __func: t.Callable[..., t.Any]) -> Command:
        ...  # pragma: no cover

    @t.overload
    def command(
        self, *args: t.Any, **kwargs: t.Any
    ) -> t.Callable[[t.Callable[..., t.Any]], Command]:
        ...  # pragma: no cover

    def command(
        self, *args: t.Any, **kwargs: t.Any
    ) -> t.Union[t.Callable[[t.Callable[..., t.Any]], Command], Command]:
        """A shortcut decorator for declaring and attaching a command to
        the group. This takes the same arguments as :func:`command` and
        immediately registers the created command with this group by
        calling :meth:`add_command`.
        To customize the command class used, set the
        :attr:`command_class` attribute.
        """
        from .decorators import command

        if self.command_class and kwargs.get("cls") is None:
            kwargs["cls"] = self.command_class

        func: t.Optional[t.Callable] = None

        if args and callable(args[0]):
            assert (
                len(args) == 1 and not kwargs
            ), "Use 'command(**kwargs)(callable)' to provide arguments."
            (func,) = args
            args = ()

        def decorator(f: t.Callable[..., t.Any]) -> Command:
            cmd: Command = command(*args, **kwargs)(f)
            self.add_command(cmd)
            return cmd

        if func is not None:
            return decorator(func)

        return decorator

    @t.overload
    def group(self, __func: t.Callable[..., t.Any]) -> "Group":
        ...  # pragma: no cover

    @t.overload
    def group(
        self, *args: t.Any, **kwargs: t.Any
    ) -> t.Callable[[t.Callable[..., t.Any]], "Group"]:
        ...  # pragma: no cover

    def group(
        self, *args: t.Any, **kwargs: t.Any
    ) -> t.Union[t.Callable[[t.Callable[..., t.Any]], "Group"], "Group"]:
        """A shortcut decorator for declaring and attaching a group to
        the group. This takes the same arguments as :func:`group` and
        immediately registers the created group with this group by
        calling :meth:`add_command`.
        To customize the group class used, set the :attr:`group_class`
        attribute.
        """
        from .decorators import group

        func: t.Optional[t.Callable] = None

        if args and callable(args[0]):
            assert (
                len(args) == 1 and not kwargs
            ), "Use 'group(**kwargs)(callable)' to provide arguments."
            (func,) = args
            args = ()

        if self.group_class is not None and kwargs.get("cls") is None:
            if self.group_class is type:
                kwargs["cls"] = type(self)
            else:
                kwargs["cls"] = self.group_class

        def decorator(f: t.Callable[..., t.Any]) -> "Group":
            cmd: Group = group(*args, **kwargs)(f)
            self.add_command(cmd)
            return cmd

        if func is not None:
            return decorator(func)

        return decorator


class RegexParameterRenamer:
    """The :class:`RegexParameterRenamer` replaces parameter name
    using regex matching and substitution. The first pattern
    that matches the parameter name is used and the matched part
    of the parameter name is substituted with the replacement value.
    If no regex expression matches the parameter name, the name is
    returned unchanged.

    :attr patterns: List of pairs of patterns and replacement values.
    """

    def __init__(self, patterns: t.List[t.Tuple[str, str]]):
        """The :class:`RegexParameterRenamer` replaces parameter name
        using regex matching and substitution. The first pattern
        that matches the parameter name is used and the matched part
        of the parameter name is substituted with the replacement value.
        If no regex expression matches the parameter name, the name is
        returned unchanged.

        :param patterns: List of pairs of patterns and the associated
                         replacement values.
        """
        self.patterns = patterns
        self._compiled_patterns = [re.compile(x) for x, _ in patterns]
        self._fast_pattern = re.compile(
            "(?:"
            + "|".join(f"(?P<pp{i}>{x})" for i, (x, _) in enumerate(patterns))
            + ")"
        )

    def __call__(self, name: str) -> str:
        """Maps the parameter name into a new name.

        :param name: Original name of the parameter.
        :return: The new name of the parameter.
        """
        found_pattern_match = self._fast_pattern.match(name)
        if not found_pattern_match:
            return name
        found_pattern_dict = found_pattern_match.groupdict()
        found_pattern_index = next(
            i for i in range(len(self.patterns)) if found_pattern_dict[f"pp{i}"]
        )
        found_pattern, replacement = (
            self._compiled_patterns[found_pattern_index],
            self.patterns[found_pattern_index][1],
        )
        return found_pattern.sub(replacement, name)


class FlattenParameterRenamer(RegexParameterRenamer):
    """The :class:`FlattenParameterRenamer` is used to map
    parameters into lower level parameters (e.g. `"class1.prop1"` into `"prop1"`).
    This is especially useful for hierarchical parsing, where properties
    deep in the class structure can have long names.
    The number of levels to remove is a parameter of the class.
    """

    def __init__(self, num_levels: int = -1):
        """The :class:`FlattenParameterRenamer` is used to map
        parameters into lower level parameters (e.g. `"class1.prop1"` into `"prop1"`).
        This is especially useful for hierarchical parsing, where properties
        deep in the class structure can have long names.

        :param num_levels: This parameter specifies how many levels to remove.
                           E.g. with `num_levels=1`, `"class1.a.prop1"` becomes `"a.prop1"`.
                           If the value is `-1` (default), infinitely many levels are removed
                           and only the last part of the parameter path is used.
        """
        self._num_levels = num_levels
        num_levels_str = str(num_levels) if num_levels != -1 else ""
        super().__init__([(fr"(?:[^\.]+\.){{,{num_levels_str}}}(.*)", r"\1")])


class _empty:
    """Marker object for Signature.empty and Parameter.empty."""


def _parse_signature(
    signature: t.Union[inspect.Signature, SignatureWithDescription],
    hierarchical: bool = False,
) -> t.List[AClickParameter]:
    params = []
    for p in signature.parameters.values():
        if (
            p.kind == inspect.Parameter.KEYWORD_ONLY
            or p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            or p.kind == inspect.Parameter.POSITIONAL_ONLY
        ):
            children = []
            parameter_type = p.annotation
            parameter_default = _empty if p.default is inspect._empty else p.default
            is_hierarchical = False
            if parameter_type is inspect._empty:
                parameter_type = _empty
            elif p.kind != inspect.Parameter.POSITIONAL_ONLY:
                if _is_class(parameter_type):
                    is_hierarchical = True
                    if hierarchical:
                        child_signature = _full_signature(parameter_type)
                        children = _parse_signature(
                            child_signature, hierarchical=hierarchical
                        )
                elif getattr(parameter_type, "__origin__", None) is t.Union and any(
                    _is_class(x) for x in parameter_type.__args__
                ):
                    is_hierarchical = True
            if is_hierarchical:
                if parameter_default is not _empty and parameter_default is not None:
                    raise ValueError(
                        "Cannot use non-default parameter with hierarchical parsing.\n"
                        "Name of the bad parameter is: {p.name}"
                    )

            param = AClickParameter(
                parameter_name=p.name,
                parameter_type=parameter_type,
                children=children,
                is_hierarchical=is_hierarchical,
                positional_only=(p.kind == inspect.Parameter.POSITIONAL_ONLY),
                keyword_only=(p.kind == inspect.Parameter.KEYWORD_ONLY),
                parameter_default=parameter_default,
                description=getattr(p, "description", None),
            )

            params.append(param)
    return params


def _is_optional(tp):
    return (
        getattr(tp, "__origin__", None) is t.Union
        and sum(1 for x in tp.__args__ if x not in (type(None),)) == 1
    )
