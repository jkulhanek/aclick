import typing as t

import click as _click

from .core import Command, Group
from .utils import _full_signature, _get_help_text


CmdType = t.TypeVar("CmdType", bound=Command)
ClickCmdType = t.TypeVar("ClickCmdType", bound=_click.Command)
F = t.TypeVar("F", bound=t.Callable[..., t.Any])


@t.overload
def command(
    __func: t.Callable[..., t.Any],
) -> Command:
    ...  # pragma: no cover


@t.overload
def command(
    name: t.Optional[str] = None,
    **attrs: t.Any,
) -> t.Callable[..., Command]:
    ...  # pragma: no cover


@t.overload
def command(
    name: t.Optional[str] = None,
    cls: t.Type[CmdType] = ...,
    **attrs: t.Any,
) -> t.Callable[..., CmdType]:
    ...  # pragma: no cover


@t.overload
def command(
    name: t.Optional[str] = None,
    cls: t.Type[ClickCmdType] = ...,
    **attrs: t.Any,
) -> t.Callable[..., ClickCmdType]:
    ...  # pragma: no cover


def command(
    name: t.Union[str, t.Callable, None] = None,
    cls: t.Optional[t.Type[_click.Command]] = None,
    **attrs: t.Any,
) -> t.Union[_click.Command, t.Callable[..., _click.Command]]:
    r"""Creates a new :class:`Command` and uses the decorated function as
    callback. This will automatically parse the function signature (if
    no other signature is supplied) and generates `Option` and `Argument`
    instance accordingly. User can override individual parameters either
    by using `click.option` and `click.argument` decorators or by passing
    custom parameters with the same name as the parameters to override.
    The name of the command defaults to the name of the function with
    underscores replaced by dashes. If you want to change that, you can
    pass the intended name as the first argument.
    All keyword arguments are forwarded to the underlying command class.
    Once decorated the function turns into a :class:`Command` instance
    that can be invoked as a command line utility or be attached to a
    command :class:`Group`.

    NOTE: This decorator should always be used instead of `click` command
    decorator, because in `click`, the help is automatically populated
    with the callback's `__doc__` and it is not parsed correctly.

    :param name: the name of the command.  This defaults to the function
                 name with underscores replaced by dashes.
    :param cls: the command class to instantiate.  This defaults to
                :class:`Command`. If the class is not a subclass of :class:`Command`,
                it is automatically wrapped.
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
    """
    func: t.Optional[t.Callable] = None

    if callable(name):
        func = name
        name = None
        assert cls is None, "Use 'command(cls=cls)(callable)' to specify a class."
        assert not attrs, "Use 'command(**kwargs)(callable)' to provide arguments."

    command_name: t.Optional[str] = name

    if cls is None:
        cls = Command
    elif not issubclass(cls, Command):
        cls = type(f"{cls.__name__}AClickWrapper", (Command, cls), {})

    def decorator(fn):
        if attrs.get("help") is None or attrs.get("short_help") is None:
            short_help, help = _get_help_text(_full_signature(fn))
            if attrs.get("help") is None:
                attrs["help"] = help
            if attrs.get("short_help") is None:
                attrs["short_help"] = short_help
        click_decorator = _click.command(name=command_name, cls=cls, **attrs)
        return click_decorator(fn)

    if func is not None:
        return decorator(func)

    return decorator


@t.overload
def group(
    __func: t.Callable[..., t.Any],
) -> Group:
    ...  # pragma: no cover


@t.overload
def group(
    name: t.Optional[str] = None,
    **attrs: t.Any,
) -> t.Callable[[F], Group]:
    ...  # pragma: no cover


def group(
    name: t.Union[str, t.Callable[..., t.Any], None] = None, **attrs: t.Any
) -> t.Union[Group, t.Callable[[F], Group]]:
    """
    Creates a new :class:`Group` with a function as callback.  This
    works otherwise the same as :func:`command` just that the `cls`
    parameter is set to :class:`Group`.
    """
    if attrs.get("cls") is None:
        attrs["cls"] = Group

    if callable(name):
        grp: t.Callable[[F], Group] = t.cast(Group, command(**attrs))
        return grp(name)

    return t.cast(Group, command(name, **attrs))