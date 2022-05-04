import builtins
import dataclasses
import inspect
import re
import types
import typing as t
from collections import deque, OrderedDict
from functools import partial
from itertools import chain

import docstring_parser

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

T = t.TypeVar("T", bound=t.Callable)
TType = t.TypeVar("TType", bound=t.Type)


class ParameterWithDescription(inspect.Parameter):
    """
    Extends `inspect.Parameter` class with description parameter.

    :attr name: The name of the parameter as a string.
    :attr default: The default value for the parameter if specified, otherwise `inspect._empty`.
    :attr annotation: The annotation for the parameter if specified, otherwise `inspect._empty`.
    :attr kind: Describes how argument values are bound to the parameter.
    :attr description: String description of the parameter.
    """

    def __init__(self, *args, description: t.Optional[str] = None, **kwargs):
        """
        Extends `inspect.Parameter` class with description parameter.

        :param name: The name of the parameter as a string.
        :param default: The default value for the parameter if specified, otherwise `inspect._empty`.
        :param annotation: The annotation for the parameter if specified, otherwise `inspect._empty`.
        :param kind: Describes how argument values are bound to the parameter.
        :param description: String description of the parameter.
        """
        super().__init__(*args, **kwargs)
        self.description = description

    @classmethod
    def from_parameter(cls, parameter: inspect.Parameter):
        """
        Creates :class:`ParameterWithDescription` instance from :class:`inspect.Parameter` instance.

        :param parameter: Parameter to convert into :class:`ParameterWithDescription`
        """
        return cls(
            name=parameter.name,
            kind=parameter.kind,
            default=parameter.default,
            annotation=parameter.annotation,
        )

    def to_parameter(self) -> inspect.Parameter:
        """
        Converts this instance of :class:`ParameterWithDescription` into a :class:`inspect.Parameter` instance.
        """
        return inspect.Parameter(
            name=self.name,
            kind=self.kind,
            default=self.default,
            annotation=self.annotation,
        )

    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, ParameterWithDescription):
            return self.description == other.description
        return False

    def __hash__(self):
        return hash((super().__hash__(), self.description))

    def replace(self, *, description=inspect._empty, **kwargs):
        """
        Generates a new instance and changes some of the properties.

        :param name: The name of the parameter as a string.
        :param default: The default value for the parameter if specified, otherwise `inspect._empty`.
        :param annotation: The annotation for the parameter if specified, otherwise `inspect._empty`.
        :param kind: Describes how argument values are bound to the parameter.
        :param description: String description of the parameter.
        """
        value = super().replace(**kwargs)
        value.description = self.description
        assert isinstance(value, ParameterWithDescription)
        if description is not inspect._empty:
            value.description = description
        return value


class SignatureWithDescription(inspect.Signature):
    """
    Extends `inspect.Signature` class with short and long description parameters.

    :attr params: Signature parameters.
    :attr return_annotation: Type annotation of the value returned by the function.
    :attr short_description: Short description of the function.
    :attr long_description: Long description of the function.
    """

    def __init__(
        self,
        *args,
        short_description: t.Optional[str] = None,
        long_description: t.Optional[str] = None,
        **kwargs,
    ):
        """
        Extends `inspect.Signature` class with short and long description parameters.

        :param params: Signature parameters.
        :param return_annotation: Type annotation of the value returned by the function.
        :param short_description: Short description of the function.
        :param long_description: Long description of the function.
        """
        super().__init__(*args, **kwargs)
        self.short_description = short_description
        self.long_description = long_description

    @classmethod
    def from_callable(cls, obj: t.Callable, **kwargs) -> "SignatureWithDescription":
        """
        Generates the signature from a callable object.

        :param obj: Callable function to generate signature from
        """
        signature = super(cls, cls).from_callable(obj, **kwargs)
        parameters = []

        # Add help
        doc = docstring_parser.parse(obj.__doc__ or "")
        doc_params = {p.arg_name: p.description for p in doc.params}
        for p in signature.parameters.values():
            if not isinstance(p, ParameterWithDescription):
                p = t.cast(
                    ParameterWithDescription, ParameterWithDescription.from_parameter(p)
                )
                p.description = doc_params.get(p.name)
            parameters.append(p)
        signature = SignatureWithDescription(
            parameters=parameters, return_annotation=signature.return_annotation
        )
        signature.long_description = doc.long_description
        signature.short_description = doc.short_description
        return signature

    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, SignatureWithDescription):
            return (
                self.short_description == other.short_description
                and self.long_description == other.long_description
            )
        return False

    def __hash__(self):
        return hash((super().__hash__(), self.long_description, self.short_description))

    def replace(
        self,
        *,
        short_description=inspect._empty,
        long_description=inspect._empty,
        **kwargs,
    ):
        """
        Generates a new instance and changes some of the properties.

        :param params: Signature parameters.
        :param return_annotation: Type annotation of the value returned by the function.
        :param short_description: Short description of the function.
        :param long_description: Long description of the function.
        """
        value = super().replace(**kwargs)
        assert isinstance(value, SignatureWithDescription)
        value.short_description = self.short_description
        value.long_description = self.long_description
        if short_description is not inspect._empty:
            value.short_description = short_description
        if long_description is not inspect._empty:
            value.long_description = long_description
        return value


def signature_with_description(obj: t.Callable, **kwargs) -> SignatureWithDescription:
    """
    Takes a signature of a callable and parses its docstring to
    add its description and the description of all its parameters.

    :param obj: Callable function to generate signature from
    """
    return SignatureWithDescription.from_callable(obj, **kwargs)


def copy_signature(
    other_function: t.Callable, copy_return_annotation: bool = False
) -> t.Callable[[T], T]:
    """This is a decorator used to copy signature from another function
    called inside this function. The parameters from the called function
    are added only if current function is able to receive them (it has
    either `*args`, or `**kwargs` argument. The signature is copied including
    the description of all parameters.

    Example usage::

        def a(param1: int):
            pass

        @copy_signature(a)
        def b(param2, *args, **kwargs):
            a(*args, **kwargs)

    :param other_function: Signature to extend the current function with (i.e. the called function)
    :param copy_return_annotation: Whether the return type annotation should be taken from the called function.
    """

    def wrap(fn: T) -> T:
        other_signature = signature_with_description(other_function)
        current_signature = signature_with_description(fn)
        signature = _merge_signatures(
            current_signature,
            other_signature,
            copy_return_annotation=copy_return_annotation,
        )
        setattr(fn, "__signature__", signature)
        return fn

    return wrap


def get_class_name(class_type: t.Type) -> str:
    """
    Returns customizable name of the class in a human-readable format.
    By default a name in the PascalCase is converted to the snake_case.

    :param class_type: Class to take the name from.
    """
    if hasattr(class_type, "_get_class_name"):
        return class_type._get_class_name()
    name = class_type.__name__
    name = _pascal_to_snake_case(name)
    return name


@t.overload
def default_from_str() -> t.Callable[[TType], TType]:
    ...  # pragma: no cover


@t.overload
def default_from_str(cls: TType) -> TType:
    ...  # pragma: no cover


def default_from_str(cls=None):
    """A decorator that extends the class with the :func:`from_str`, :func:`__str__`, and :func:`__str_with_dashes_option__`
    methods. The :func:`from_str` method is a classmethod and it parses a string value into
    the instance of the class. If some class derive this class, this class will be able to parse
    the string value as these instances as well. In this case, :func:`from_str` should be called on this
    type (parent class).

    :func:`__str__` and :func:`__str_with_dashes_option__` are implemented as inverse of :func:`from_str` (i.e., the
    strings generated by these methods can be parsed using :func:`from_str` method). :func:`__str_with_dashes_option__`
    further allows to specify if the underscores in class names and property names should be converted to dashes.

    Example usage::

        @default_from_str
        class A:
            def __init__(self, prop1: int):
                self.prop1 = prop1

        str(A(42)) == "a(prop1=42)"

    """

    def decorator(cls: TType) -> TType:
        _validate_class_supports_to_str(cls)
        setattr(
            cls,
            "from_str",
            classmethod(
                partial(
                    _parse_class_structure_for_all_descendents,
                    base_cls=cls,
                    method_name="from_str",
                )
            ),
        )
        setattr(cls, "__str_with_dashes_option__", _class_to_str_with_dashes_option)
        setattr(cls, "__str__", _class_to_str)
        return cls

    if cls is not None:
        return decorator(cls)
    return decorator


def parse_class_structure(
    value: str,
    classes: t.List[t.Type],
    known_classes: t.Optional[t.List[t.Type]] = None,
) -> t.Any:
    """Parses a string representing a structure of classes into a
    tree representation of the classes. This method is directly
    used by the :func:`default_from_str` decorator.

    The following conventions are used:

    - Classes are represented as `class_name(arg1, arg2, named_arg1=val3, ...)`
    - Lists are represented as `[val1, val2, ...]`
    - Tuples are represented as `(val1, val2, ...)`
    - Dicts are represented as `{key1=val1, key2=val2, ...}`
    - Strings can be represented as `value` or `"complicated 'value'"` or `'complicated "value"'`
    - Numbers and bool are represented as strings: `15`, `16.2`, `true`

    :param value: String value to parse.
    :param classes: List of classes to parse the string into.
    :param known_classes: List of known classes that can be used during parsing on
                          lower levels (e.g. as a property value).
    :return: An instance of one of the classes is returned.
    """
    known_classes = _find_known_classes(classes)
    known_classes_map = {get_class_name(x): x for x in chain(classes, known_classes)}
    class_map = {get_class_name(x): x for x in classes}

    def _map_class(class_argument, value_type, is_root=True):
        if value_type is not None and hasattr(value_type, "from_str"):
            return value_type.from_str(str(class_argument))
        elif isinstance(class_argument, _ClassArgument):
            known_classes = class_map if is_root else known_classes_map
            if class_argument.name not in known_classes:
                raise RuntimeError(
                    f'Could not find class with name "{class_argument.name}" in the list of registered classes: {", ".join(sorted(known_classes.keys()))}'
                )
            class_type = known_classes[class_argument.name]
            signature = _full_signature(class_type)
            args_types, kwargs_types = _get_signature_args_kwargs(signature)
            num_pos_args_required = sum(
                1
                for p in signature.parameters.values()
                if p.kind == inspect.Parameter.POSITIONAL_ONLY
                and p.default is inspect._empty
            )
            required_keys = {
                p.name
                for p in signature.parameters.values()
                if p.kind
                in {
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                }
                and p.default is inspect._empty
            }
            if len(class_argument.args) > len(args_types):
                raise RuntimeError(
                    f"Number of passed positional arguments ({len(class_argument.args)}) to class {class_argument.name} exceeds the number of allowed arguments ({len(args_types)})."
                )
            elif len(class_argument.args) < num_pos_args_required:
                raise RuntimeError(
                    f"Number of passed positional arguments ({len(class_argument.args)}) to class {class_argument.name} is lower then the number of expected arguments ({num_pos_args_required})."
                )
            unknown_keys = set(class_argument.kwargs.keys()).difference(
                set(kwargs_types.keys())
            )
            if unknown_keys:
                raise RuntimeError(
                    f"There were unknown parameters {{{', '.join(sorted(unknown_keys))}}} to class {class_argument.name}. Allowed parameters are: {{{', '.join(sorted(kwargs_types.keys()))}}}"
                )
            missing_keys = required_keys.difference(class_argument.kwargs.keys())
            missing_keys.difference_update(
                x for x, _ in args_types[: len(class_argument.args)]
            )
            if missing_keys:
                raise RuntimeError(
                    f'Parameters {{{", ".join(sorted(missing_keys))}}} to class {class_argument.name} are missing. Passed arguments are: {{{", ".join(sorted(class_argument.kwargs.keys()))}}}.'
                )
            args = [
                _map_class(x, tp, is_root=False)
                for x, (_, tp) in zip(class_argument.args, args_types)
            ]
            kwargs = OrderedDict(
                (k, _map_class(v, kwargs_types[k], is_root=False))
                for k, v in class_argument.kwargs.items()
            )
            return class_type(*args, **kwargs)

        assert value_type is not None, "Type of value must be supplied"
        if getattr(value_type, "__origin__", None) is t.Union:
            # Add Optional support
            if sum(1 for x in value_type.__args__ if x not in {type(None)}) == 1:
                value_type = next(
                    x for x in value_type.__args__ if x not in {type(None)}
                )
                if str(class_argument).lower() in {"none", "null", ""}:
                    return None

        if getattr(value_type, "__origin__", None) in (list,):
            class_argument = [
                _map_class(x, value_type.__args__[0], is_root=False)
                for x in class_argument
            ]
            value_type = value_type.__origin__
        if getattr(value_type, "__origin__", None) in (Literal,):
            class_argument = _map_class(class_argument, str, is_root=False)
            if class_argument not in value_type.__args__:
                raise RuntimeError(
                    f"Literal value {class_argument} is not in the set of supported values {{{', '.join(sorted(value_type.__args__))}}}"
                )
            value_type = str
        if getattr(value_type, "__origin__", None) in (tuple,):
            if len(class_argument) != len(value_type.__args__):
                raise RuntimeError(
                    f"Cannot parse {len(class_argument)} values as tuple of length {len(value_type.__args__)}"
                )
            class_argument = [
                _map_class(x, tp, is_root=False)
                for tp, x in zip(value_type.__args__, class_argument)
            ]
            value_type = value_type.__origin__

        if getattr(value_type, "__origin__", None) in (dict, OrderedDict):
            if not isinstance(class_argument, OrderedDict):
                raise RuntimeError(
                    f"Cannot parse {class_argument} as a {value_type.__origin__.__name__} instance"
                )
            class_argument = [
                (
                    _map_class(k, value_type.__args__[0], is_root=False),
                    _map_class(v, value_type.__args__[1], is_root=False),
                )
                for k, v in class_argument.items()
            ]
            value_type = value_type.__origin__

        if value_type == bool:
            if not isinstance(class_argument, str):
                raise RuntimeError(f"Cannot parse {class_argument} as bool")

            norm = class_argument.strip().lower()
            if norm in {"1", "true", "t", "yes", "y", "on"}:
                return True

            if norm in {"0", "false", "f", "no", "n", "off"}:
                return False

            raise RuntimeError(f"Cannot parse {class_argument} as bool")
        return value_type(class_argument)

    assert isinstance(value, str)
    class_tree = _ClassArgument.from_str(value)
    value = _map_class(class_tree, None)
    return value


TSignature = t.TypeVar("TSignature", bound=inspect.Signature)


def _merge_signatures(
    current_signature: TSignature,
    *signatures: TSignature,
    copy_return_annotation: bool = False,
) -> TSignature:
    if len(signatures) == 0:
        return current_signature

    other_signature = signatures[0]
    parameters = []
    has_args = False
    for p in current_signature.parameters.values():
        if p.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            parameters.append(p)
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            for p2 in other_signature.parameters.values():
                if p2.kind in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                }:
                    parameters.append(p2)
            has_args = True
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            for p2 in other_signature.parameters.values():
                if p2.kind in {
                    inspect.Parameter.KEYWORD_ONLY,
                    inspect.Parameter.VAR_KEYWORD,
                }:
                    parameters.append(p2)
                elif (
                    p2.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and not has_args
                ):
                    parameters.append(
                        p2.replace(
                            kind=inspect.Parameter.KEYWORD_ONLY,
                        )
                    )
    return_annotation = (
        other_signature.return_annotation
        if copy_return_annotation
        else current_signature.return_annotation
    )
    signature = current_signature.replace(
        parameters=parameters, return_annotation=return_annotation
    )
    if len(signatures) > 1:
        signature = _merge_signatures(
            signature, *signatures[1:], copy_return_annotation=copy_return_annotation
        )
    return signature


def _full_signature(fn: t.Callable) -> SignatureWithDescription:
    assert callable(fn)

    if inspect.isclass(fn):
        signature = _merge_signatures(
            *[signature_with_description(x) for x in fn.__mro__]
        )
    else:
        signature = signature_with_description(fn)
    return signature


class _ClassArgument:
    def __init__(self, name, args, kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        args = [
            self.escape_string(v) if isinstance(v, str) else repr(v) for v in self.args
        ] + [
            f"{k}={self.escape_string(v) if isinstance(v, str) else repr(v)}"
            for k, v in self.kwargs.items()
        ]
        return f'{self.name}({", ".join(args)})'

    def __str__(self):
        return repr(self)

    class ParseError(ValueError):
        def __init__(self, message, string, span):
            string = (
                string[: span[0]]
                + "\033[4;31m"
                + string[span[0] : span[1]]
                + "\033[0;0m"
                + string[span[1] :]
            )
            super().__init__(message + "\n" + f'Error in string "{string}" {span}')

    @staticmethod
    def match_string(string, span):
        local_string = string[slice(*span)]
        string_match = re.match(
            r"^(?:(['\"])(?:(?!\1)[^\\]|\\\\|\\\1)*\1|[^\),\]\}]*)", local_string
        )
        assert string_match, f'Could not parse string "{local_string}"'
        return tuple(span[0] + x for x in string_match.span())

    @staticmethod
    def unescape_string(string):
        if len(string) == 0 or string[0] not in {'"', "'"}:
            return string
        assert len(string) > 1
        quotation_mark = string[0]
        string = string[1:-1]
        string = string.replace(f"\\{quotation_mark}", quotation_mark).replace(
            "\\\\", "\\"
        )
        return string

    @staticmethod
    def escape_string(string: str) -> str:
        if {'"', ")", "]", "}", ",", " ", "\r", "\n", "\t"}.intersection(string):
            string = string.replace("\\", "\\\\").replace('"', '\\"')
            return '"' + string + '"'
        return string

    @staticmethod
    def parse_argument(string, span=None):
        if span is None:
            span = 0, len(string)
        out_name = None
        name_match = re.match(r"^([a-zA-Z0-9_-]+)\s*=\s*", string[slice(*span)])
        if name_match:
            out_name = name_match.group(1).replace("-", "_")
            span = (span[0] + name_match.end(), span[1])
        arguments = []
        kwargs = OrderedDict()

        class_match = re.match(r"^([a-zA-Z0-9_-]+)\(", string[slice(*span)])
        if class_match:
            # This is a nested class argument
            end = class_match.end() + span[0]
            if len(string) > end and string[end] != ")":
                while True:
                    (_, new_end), name, val = _ClassArgument.parse_argument(
                        string, span=(end, span[1])
                    )
                    if name is None:
                        arguments.append(val)
                    else:
                        name = name.replace("-", "_")
                        kwargs[name] = val
                    end = new_end
                    par_match = re.match(r"^\s*(,|\))\s*", string[end : span[1]])
                    if par_match:
                        end += par_match.end()
                        if par_match.group(0).strip() == ")":
                            break
                    else:
                        raise _ClassArgument.ParseError(
                            "Unexpected end of argument", string, (end, span[1])
                        )
            elif len(string) > end and string[end] in {")"}:
                end += 1
            else:
                raise _ClassArgument.ParseError(
                    "Unexpected end of argument", string, (end, span[1])
                )
            span = (span[0], end)
            class_name = class_match.group(1).replace("-", "_")
            return (
                span,
                out_name,
                _ClassArgument(class_name, arguments, kwargs),
            )

        list_dict_match = re.match(r"^(\(|\[|\{)\s*", string[slice(*span)])
        if list_dict_match:
            # This is a list argument
            end = list_dict_match.end() + span[0]
            is_dict = list_dict_match.group(1) == "{"
            arguments = []
            if len(string) > end and string[end] not in {")", "}", "]"}:
                while True:
                    (_, new_end), name, val = _ClassArgument.parse_argument(
                        string, span=(end, span[1])
                    )
                    if is_dict:
                        if name is None:
                            raise _ClassArgument.ParseError(
                                "Missing dict entry name", string, (end, new_end)
                            )
                        arguments.append((name, val))
                    elif not is_dict:
                        if name is not None:
                            name, val = None, _ClassArgument.unescape_string(
                                string[end:new_end]
                            )
                        arguments.append(val)

                    end = new_end
                    par_match = re.match(r"^\s*(,|\)|\]|\})\s*", string[end : span[1]])
                    if par_match:
                        end += par_match.end()
                        if par_match.group(0).strip() in {")", "}", "]"}:
                            break
                    if not par_match or end == span[1]:
                        raise _ClassArgument.ParseError(
                            "Unexpected end of argument", string, (end, span[1])
                        )
            elif len(string) > end and string[end] in {")", "}", "]"}:
                end += 1
            else:
                raise _ClassArgument.ParseError(
                    "Unexpected end of argument", string, (end, span[1])
                )

            span = (span[0], end)
            out_type = {"{": OrderedDict, "[": list, "(": tuple}[
                list_dict_match.group(1)
            ]

            return (
                span,
                out_name,
                out_type(arguments),
            )

        # Else:
        span = _ClassArgument.match_string(string, span)
        argument_value = string[slice(*span)]
        argument_value = _ClassArgument.unescape_string(argument_value)
        return span, out_name, argument_value

    @staticmethod
    def from_str(string):
        return _ClassArgument.parse_argument(string)[-1]


def _parse_class_structure_for_all_descendents(
    cls, value: str, *, base_cls, method_name="from_str"
):
    def add_classes_rec(cls):
        yield cls
        for c in cls.__subclasses__():
            yield from add_classes_rec(c)

    classes = list(add_classes_rec(base_cls))
    if base_cls != cls:
        raise RuntimeError(
            f"Method {method_name} should be called on {base_cls}, not {cls}"
        )
    return parse_class_structure(value, classes)


def _validate_class_supports_to_str(cls):
    original_cls = cls

    def inner(cls, ignore_from_str=False):
        if not ignore_from_str and hasattr(cls, "from_str"):
            pass  # from_str objects are supported
        elif getattr(cls, "__origin__", None) == Literal:
            if not all(isinstance(x, str) for x in cls.__args__):
                raise ValueError(
                    f"Class {original_cls} is not supported because it contains a Literal property of non-str types"
                )
        elif getattr(cls, "__origin__", None) in {
            t.Union,
            tuple,
            list,
            set,
            dict,
            OrderedDict,
        }:
            for tp in cls.__args__:
                inner(tp)
            if getattr(cls, "__origin__", None) in (dict, OrderedDict):
                if cls.__args__[0] not in (str,):
                    raise RuntimeError(
                        f"Class {original_cls} is not supported because it contains a dictionary with key of type {cls.__args__[0]}. Only str type is supported"
                    )
        elif cls in (type(None), str, bool, float, int, bytes):
            pass
        elif dataclasses.is_dataclass(cls):
            # Dataclasses are supported
            for f in dataclasses.fields(cls):
                inner(f.type)
        elif _is_class(cls):
            # Classes are supported
            # but they must set all constructor parameters as properties
            # this is verified on runtime
            args_types, kwargs_types = _get_signature_args_kwargs(cls, exclusive=True)
            for p_name, p_type in chain(args_types, kwargs_types.items()):
                if p_type is inspect._empty:
                    raise ValueError(
                        f"Class {original_cls} is not supported because field {p_name} in {cls} does not have a type annotation"
                    )
                inner(p_type)
        else:
            raise ValueError(
                f"Class {original_cls} is not supported because it contains type {cls} which is not supported"
            )

    inner(cls, ignore_from_str=True)


def _class_to_str_with_dashes_option(self, use_dashes=False) -> str:
    def inner(obj, ignore_from_str=True, use_dashes=False) -> str:
        if not ignore_from_str and hasattr(obj, "from_str"):
            if hasattr(obj, "__str_with_dashes_option__"):
                return _ClassArgument.escape_string(
                    obj.__str_with_dashes_option__(use_dashes=use_dashes)
                )
            else:
                return _ClassArgument.escape_string(str(obj))
        elif isinstance(obj, tuple):
            return (
                "("
                + ", ".join(inner(x, False, use_dashes=use_dashes) for x in obj)
                + ")"
            )
        elif isinstance(obj, (list, set)):
            return (
                "["
                + ", ".join(inner(x, False, use_dashes=use_dashes) for x in obj)
                + "]"
            )
        elif isinstance(obj, (dict, OrderedDict)):
            return (
                "{"
                + ", ".join(
                    str(k) + "=" + inner(x, False, use_dashes=use_dashes)
                    for k, x in obj.items()
                )
                + "}"
            )
        elif isinstance(obj, (bool, float, int, bytes, type(None))):
            return str(obj)
        elif isinstance(obj, str):
            return _ClassArgument.escape_string(obj)
        elif dataclasses.is_dataclass(obj):
            class_name = get_class_name(type(obj))
            args: t.List[str] = []
            kwargs = {}
            for f in dataclasses.fields(obj):
                name = f.name
                if use_dashes:
                    name = name.replace("_", "-")
                kwargs[name] = inner(getattr(obj, f.name), False, use_dashes=use_dashes)
                # if getattr(f, 'kw_only', False):
                # else:
                #     args.append(getattr(self, f.name))
            args.extend((f"{k}={v}" for k, v in kwargs.items()))
            if use_dashes:
                class_name = class_name.replace("_", "-")
            return f'{class_name}({", ".join(args)})'
        elif _is_class(type(obj)):
            class_name = get_class_name(type(obj))
            args_par, kwargs_par = _get_signature_args_kwargs(type(obj), exclusive=True)
            args = []
            kwargs = {}
            for p_name, _ in args_par:
                if not hasattr(obj, p_name):
                    raise RuntimeError(
                        f"Cannot serialize class {type(obj)}, because the constructor parameter {p_name} is missing from class's properties"
                    )
                value = getattr(obj, p_name)
                args.append(inner(value, False, use_dashes=use_dashes))
            for name, p in kwargs_par.items():
                if not hasattr(obj, name):
                    raise RuntimeError(
                        f"Cannot serialize class {type(obj)}, because the constructor parameter {name} is missing from class's properties"
                    )
                value = getattr(obj, name)
                if use_dashes:
                    name = name.replace("_", "-")
                args.append(name + "=" + inner(value, False, use_dashes=use_dashes))
            if use_dashes:
                class_name = class_name.replace("_", "-")
            return f'{class_name}({", ".join(args)})'
        else:
            raise RuntimeError(f"Type {type(obj)} is not supported")

    return inner(self, use_dashes=use_dashes)


def _class_to_str(self):
    return self.__str_with_dashes_option__(use_dashes=False)


def _is_class(tp):
    if not inspect.isclass(tp):
        return False

    if tp in (type(None), types.FunctionType):
        return False

    if tp in vars(builtins).values():
        return False

    return True


def _pascal_to_snake_case(val):
    out = ""
    for x in val:
        if x.isupper():
            if len(out) > 0:
                out += "_"
            out += x.lower()
        else:
            out += x
    return "".join(out)


def _find_known_classes(classes) -> t.List[t.Type]:
    known_classes = []
    classes = deque(classes)
    while classes:
        c = classes.pop()
        known_classes.append(c)
        for p in _full_signature(c).parameters.values():
            if getattr(p.annotation, "__origin__", None) in {
                list,
                set,
                tuple,
                dict,
                t.Union,
                OrderedDict,
            }:
                classes.extend(x for x in p.annotation.__args__ if _is_class(x))
            elif _is_class(p.annotation):
                classes.append(p.annotation)
    return known_classes


def _get_help_text(signature) -> t.Tuple[str, str]:
    vals = [x for x in (signature.short_description, signature.long_description) if x]
    return (
        inspect.cleandoc(signature.short_description or ""),
        inspect.cleandoc("\n\n".join(vals)),
    )


def _get_signature_args_kwargs(signature, exclusive=False):
    if not isinstance(signature, inspect.Signature):
        signature = _full_signature(signature)
    signature = list(signature.parameters.values())
    args = [
        (p.name, p.annotation)
        for p in signature
        if p.kind
        in (
            {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
            if not exclusive
            else {
                inspect.Parameter.POSITIONAL_ONLY,
            }
        )
    ]
    kwargs = OrderedDict(
        (p.name, p.annotation)
        for p in signature
        if p.kind
        in {inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    )
    return args, kwargs


def _wrap_fn_to_allow_kwargs_instead_of_args(fn):
    signature = _full_signature(fn)
    args_params = [
        p
        for p in signature.parameters.values()
        if p.kind in {inspect.Parameter.POSITIONAL_ONLY}
    ]
    kwargs_params = [
        p
        for p in signature.parameters.values()
        if p.kind
        in {inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    if len(args_params) == 0:
        return fn

    def inner_fn(**kwargs):
        args = []
        for p in args_params:
            if p.name not in kwargs:
                break
            args.append(kwargs.pop(p.name))
        return fn(*args, **kwargs)

    parameters = []
    for p in args_params:
        parameters.append(
            p.replace(
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
    parameters.extend(kwargs_params)
    setattr(
        inner_fn,
        "__signature__",
        signature.replace(
            parameters=parameters, return_annotation=signature.return_annotation
        ),
    )
    setattr(inner_fn, "__doc__", fn.__doc__)
    return inner_fn
