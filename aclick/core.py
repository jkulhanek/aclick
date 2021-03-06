import copy
import inspect
import re
import sys
import typing as t
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import click as _click

from .types import (
    AllParameters,
    ClassHierarchicalOption,
    ClassUnion,
    DEFAULT_USE_DASHES,
    Dict as DictParam,
    List,
    OptionalTypeHierarchicalOption,
    ParameterGroup,
    Tuple,
    UnionTypeHierarchicalOption,
)

from .utils import (
    _class_to_str_with_dashes_option,
    _full_signature,
    _get_help_text,
    _is_class,
    _SUPPORTED_TYPES,
    _wrap_fn_to_allow_kwargs_instead_of_args,
    as_dict,
    build_examples,
    get_class_name,
    Literal,
    ParameterWithDescription,
    SignatureWithDescription,
)


_get_class_name = get_class_name


class Context(_click.Context):
    def __init__(
        self,
        *args,
        callback_signature: t.Optional[inspect.Signature] = None,
        callback: t.Optional[t.Callable] = None,
        param_groups: t.Optional[t.List[ParameterGroup]] = None,
        param_group_contexts: t.Optional[t.Dict[str, t.Any]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.callback = callback
        if callback_signature is None:
            assert (
                callback is not None
            ), "Either callback or callback_signature must be specified"
            self.callback_signature: inspect.Signature = _full_signature(callback)
        else:
            self.callback_signature = callback_signature
        self.param_groups = param_groups or []
        self.param_group_contexts = param_group_contexts or dict()
        self.configuration_file_loaded: t.Optional[str] = None


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


def handle_parse_all_parameters_group_result(ctx: Context):
    param_subscribers = []
    for p in ctx.callback_signature.parameters.values():
        if getattr(p.annotation, "__origin__", None) is AllParameters:
            param_subscribers.append((p.name, next(iter(p.annotation.__args__), "")))
    out_dict = OrderedDict(
        (
            (k, as_dict(v, ctx.callback_signature.parameters[k].annotation))
            for k, v in ctx.params.items()
        )
    )
    for pname, path in param_subscribers:
        if sys.version < "3.9":
            # path can be ForwardRef
            path = getattr(path, "__forward_arg__", path)
        local_out_dict = out_dict
        if path != "":
            for part in path.split("."):
                local_out_dict = local_out_dict[part]
        ctx.params[pname] = copy.deepcopy(local_out_dict)


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
                         with hierarchical parsing. Default is to use hierarchical
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
    :param show_defaults: whether to show defaults for all parameters.
                          Default is ``True``.
    :param use_dashes: use dashes instead of underscores in parameter and class
                       names. Default is ``True``.
    :param num_inline_examples_help: limits the number of examples to show in
                                     help for each inline type argument. If the
                                     value is -1, all possible values are shown.
                                     Default is 0.
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

    context_class = Context

    def __init__(
        self,
        *args,
        signature: t.Optional[inspect.Signature] = None,
        hierarchical: bool = True,
        map_parameter_name: t.Optional[ParameterRenamer] = None,
        show_defaults: t.Optional[bool] = True,
        use_dashes: bool = DEFAULT_USE_DASHES,
        num_inline_examples_help: int = 0,
        **kwargs,
    ):
        assert num_inline_examples_help >= -1
        super().__init__(*args, **kwargs)
        if signature is None:
            if self.callback is None:
                raise ValueError("Either signature or callback must be specified")
            signature = _full_signature(self.callback)
        self.callback_signature: inspect.Signature = signature
        self.hierarchical = hierarchical
        self.map_parameter_name = map_parameter_name
        self.show_defaults = show_defaults
        self.use_dashes = use_dashes
        self.num_inline_examples_help = num_inline_examples_help
        self._assert_signature_is_supported(self.callback_signature)
        if (self.help is None or self.short_help is None) and isinstance(
            signature, SignatureWithDescription
        ):
            short_help, help = _get_help_text(signature)
            self.short_help = (
                self.short_help if self.short_help is not None else short_help
            )
            self.help = self.help if self.help is not None else help

    def make_context(self, *args, **kwargs):
        ctx = super().make_context(
            *args,
            **kwargs,
            callback_signature=self.callback_signature,
            callback=self.callback,
        )
        return ctx

    def build_click_parameter(
        self,
        full_name: str,
        param: t.Union[inspect.Parameter, ParameterWithDescription],
    ) -> t.Optional[t.Union[_click.Parameter, ParameterGroup]]:
        if param.kind not in {
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        }:
            # Do not add the parameter for *args and **kwargs
            return None

        if param.annotation is inspect._empty:
            raise ValueError(
                f'Parameter named "{full_name}" does not have it\'s type specified.'
            )

        is_argument = param.kind == inspect.Parameter.POSITIONAL_ONLY
        is_hierarchical = self.hierarchical and _is_hierarchical(param)
        parameter_type = param.annotation
        option_name = []

        if getattr(param.annotation, "__origin__", None) is AllParameters:
            # We ignore this parameter now, it will be used for binding later
            if "." in full_name:
                raise ValueError(
                    "AllParameters arguments are allowed only at the root level"
                )
            return None

        if param.kind == inspect.Parameter.POSITIONAL_ONLY and "." in full_name:
            raise ValueError(
                f'Positional arguments are allowed only on top level. Failed argument name is "{full_name}".'
            )

        parameter_name = full_name.replace(".", "_")
        option_name = [parameter_name]
        if not is_argument:
            parameter_name = full_name
            if self.map_parameter_name is not None:
                parameter_name = self.map_parameter_name(parameter_name)
            parameter_name = parameter_name.replace(".", "-").replace("_", "-")
            option_name.insert(0, f"--{parameter_name}")

        if is_hierarchical:
            assert not is_argument, (
                f"Hierarchical arguments (name: {full_name}) are not supported. "
                "Only simple types can be used in hierarchical parsing"
            )

            if getattr(param.annotation, "__origin__", None) is t.Union:
                # Validate hierarchical param
                nonnull_args = [
                    x for x in param.annotation.__args__ if x not in (type(None),)
                ]
                all_classes = all(map(_is_class, nonnull_args))
                if not all_classes:
                    raise ValueError(
                        f"Parameter with parameter name {full_name} of type {param.annotation} is not supported. "
                        "Only unions of all classes are supported."
                    )

                if not _is_optional(param.annotation) and len(
                    param.annotation.__args__
                ) != len(nonnull_args):
                    raise ValueError(
                        f"Parameter with parameter name {full_name} of type {param.annotation} is not supported. "
                        "Only unions of all classes are supported (without None)."
                    )

                if _is_optional(param.annotation) and param.default not in (
                    None,
                    inspect._empty,
                ):
                    raise ValueError(
                        f"Parameter with parameter name {full_name} of type {param.annotation} with default {param.default} is not supported. "
                        "In hierarchical parsing, optional type only supports None as the default value."
                    )

                # Construct the parameter
                if _is_optional(param.annotation):
                    parameter_type = next(
                        x for x in param.annotation.__args__ if x not in (type(None),)
                    )
                    return OptionalTypeHierarchicalOption(
                        parameter_name,
                        full_name,
                        parameter_type,
                        required=param.default is inspect._empty,
                        get_class_name=self._get_custom_class_name,
                    ).assert_is_supported(self)
                elif getattr(param.annotation, "__origin__", None) is t.Union:
                    values = [x for x in param.annotation.__args__ if _is_class(x)]
                    return UnionTypeHierarchicalOption(
                        parameter_name,
                        full_name,
                        classes=values,
                        required=True,
                        get_class_name=self._get_custom_class_name,
                    ).assert_is_supported(self)

            elif _is_class(param.annotation):
                if param.default is not inspect._empty:
                    raise ValueError(
                        "Cannot use a parameter with a default in hierarchical parsing.\n"
                        f"Name of the bad parameter is: {full_name}"
                    )
                return ClassHierarchicalOption(
                    full_name, param.annotation
                ).assert_is_supported(self)

        kwargs = dict()
        if param.default is not inspect._empty:
            default = param.default
            if callable(default):
                # Fix for click
                old_default = default
                default = lambda: old_default
                setattr(default, "__str__", old_default.__str__)

            kwargs["default"] = default
            kwargs["required"] = False
            if not is_argument and self.show_defaults is not None:
                kwargs["show_default"] = self.show_defaults
        else:
            kwargs["required"] = True
        if help is not None and not is_argument and hasattr(param, "description"):
            kwargs["help"] = getattr(param, "description")
        original_parameter_type = parameter_type
        cls = _click.Argument if is_argument else _click.Option

        # Handle optional
        if _is_optional(parameter_type):
            parameter_type = next(
                x for x in parameter_type.__args__ if x not in (type(None),)
            )

        if _is_class(parameter_type):
            if self.num_inline_examples_help != 0:
                # Add help for inline examples
                if "help" not in kwargs or not kwargs["help"]:
                    kwargs["help"] = ""
                if kwargs["help"]:
                    kwargs["help"] += "\n\n"
                kwargs["help"] += (
                    "\b\n"
                    + _build_inline_class_union_help(
                        [parameter_type], self.use_dashes, self.num_inline_examples_help
                    ).replace("\n\n", "\n\n\b\n")
                    + "\n\n "
                )
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
            if self.num_inline_examples_help != 0:
                # Add help for inline examples
                if "help" not in kwargs or not kwargs["help"]:
                    kwargs["help"] = ""
                if kwargs["help"]:
                    kwargs["help"] += "\n\n"
                kwargs["help"] += (
                    "\b\n"
                    + _build_inline_class_union_help(
                        types, self.use_dashes, self.num_inline_examples_help
                    ).replace("\n\n", "\n\n\b\n")
                    + "\n\n"
                )
            kwargs["type"] = ClassUnion(types)
            return cls(option_name, **kwargs)

        if getattr(parameter_type, "__origin__", None) is list:
            kwargs["type"] = List(parameter_type.__args__[0])
            if (
                "default" in kwargs
                and kwargs.get("show_default", False)
                and isinstance(kwargs["default"], list)
            ):
                kwargs["show_default"] = _class_to_str_with_dashes_option(
                    kwargs["default"], use_dashes=self.use_dashes
                )[1:-1]
            return cls(option_name, **kwargs)
        if getattr(parameter_type, "__origin__", None) in (dict, OrderedDict):
            key_type, value_type = parameter_type.__args__
            container_type = parameter_type.__origin__
            if key_type is str:
                kwargs["type"] = DictParam(
                    value_type, key_type=key_type, container_type=container_type
                )
                if (
                    "default" in kwargs
                    and kwargs.get("show_default", False)
                    and isinstance(kwargs["default"], (dict, OrderedDict))
                ):
                    kwargs["show_default"] = _class_to_str_with_dashes_option(
                        kwargs["default"], use_dashes=self.use_dashes
                    )[1:-1]
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
        raise ValueError(
            f"Could not build a Click parameter for parameter with name: {full_name} and type: {parameter_type}"
        )

    def _assert_signature_is_supported(
        self, signature: inspect.Signature, base_path: str = ""
    ) -> None:
        for p in signature.parameters.values():
            self.build_click_parameter(base_path + p.name, p)

    def _get_custom_class_name(self, cls):
        return get_class_name(cls).replace("_", "-")

    def get_params(self, ctx: _click.Context):
        assert isinstance(ctx, Context), "Context must be an instance of aclick.Context"
        ctx.param_groups = []
        params = super().get_params(ctx)
        existing_names = {x.name for x in params}
        signature_params = []
        for p in ctx.callback_signature.parameters.values():
            click_param = self.build_click_parameter(p.name, p)
            if click_param is not None:
                signature_params.append(click_param)

        def walk(p):
            assert isinstance(
                ctx, Context
            ), "Context must be an instance of aclick.Context"  # only because of mypy
            if isinstance(p, _click.Parameter) and p.name not in existing_names:
                params.append(p)
            if isinstance(p, ParameterGroup):
                ctx.param_groups.append(p)
                for p_child in p.get_params(ctx):
                    walk(p_child)

        for param in signature_params:
            walk(param)
        return params

    def parse_args(self, ctx: _click.Context, args: t.List[str]) -> t.List[str]:
        assert isinstance(ctx, Context), "Context must be an instance of aclick.Context"
        original_args = args
        if self.hierarchical:
            last_num_params = 0
            local_ctx = copy.deepcopy(ctx)

            # Share param group contexts with the local context
            local_ctx.param_group_contexts = ctx.param_group_contexts
            local_ctx.command = ctx.command
            local_ctx.ignore_unknown_options = True
            local_ctx.resilient_parsing = True
            iters_left = None
            while True:
                parser = self.make_parser(local_ctx)
                opts, args, param_order = parser.parse_args(
                    args=copy.deepcopy(original_args)
                )

                params = self.get_params(local_ctx)
                if last_num_params == len(params):
                    if iters_left is None or iters_left <= 0:
                        break
                    if iters_left is not None:
                        iters_left -= 1
                last_num_params = len(params)
                for param in _click.core.iter_params_for_processing(
                    param_order, params
                ):
                    if param.callback is not None and iters_left is None:
                        iters_left = 1
                    value, args = param.handle_parse_result(local_ctx, opts, args)

            ctx.configuration_file_loaded = local_ctx.configuration_file_loaded
            ctx.callback = local_ctx.callback
            ctx.callback_signature = local_ctx.callback_signature
        super().parse_args(ctx, original_args)

        for param_group in reversed(ctx.param_groups):
            param_group.handle_parse_group_result(ctx)

        # Handle AllParameters group
        handle_parse_all_parameters_group_result(ctx)

        return ctx.args

    def invoke(self, ctx: _click.Context) -> t.Any:
        assert isinstance(ctx, Context), "Context must be an instance of aclick.Context"
        callback = ctx.callback
        if callback is not None:
            callback = _wrap_fn_to_allow_kwargs_instead_of_args(callback)
        old_callback, self.callback = self.callback, callback
        try:
            result = super().invoke(ctx)
        finally:
            self.callback = old_callback
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


def _is_hierarchical(param: inspect.Parameter) -> bool:
    is_hierarchical = False
    if param.annotation is not inspect._empty and param.kind in {
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }:
        if _is_class(param.annotation) and not hasattr(param.annotation, "from_str"):
            is_hierarchical = True
        elif getattr(param.annotation, "__origin__", None) is t.Union and any(
            _is_class(x) for x in param.annotation.__args__
        ):
            is_hierarchical = True
    return is_hierarchical


def _is_optional(tp):
    return (
        getattr(tp, "__origin__", None) is t.Union
        and sum(1 for x in tp.__args__ if x not in (type(None),)) == 1
    )


def _build_inline_class_union_help(types, use_dashes: bool, num_examples: int = 0):
    if num_examples == 0:
        return ""
    union_type: t.Type = t.cast(t.Type, t.Union[tuple(types)])
    out = "Examples of allowed values:\n"
    limit_examples: t.Optional[int] = num_examples if num_examples >= 0 else None
    out += "\n\n".join(
        build_examples(union_type, use_dashes=use_dashes, limit=limit_examples)
    )
    return out
