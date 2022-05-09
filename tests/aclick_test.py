import inspect
import traceback
import typing as t
from dataclasses import dataclass
from typing import List, Optional, Tuple

import aclick as click
import click as _click
import pytest
from aclick.utils import Literal

from ._common import click_test, click_test_error  # noqa: F401


@click_test()
def test_click_defaults(b: str = "test", a: int = 5, c: float = 3.4):
    assert b == "test"
    assert a == 5
    assert c == 3.4


@click_test_error("--help")
def test_click_defaults_in_help(err, b: str = "test", a: int = 5, c: float = 3.4):
    status, msg = err
    assert status == 0
    assert "test" in msg
    assert "5" in msg
    assert "3.4" in msg


@click_test()
@_click.option("--some-option", "b", default="test ok")
def test_click_ignore_defined_options(b: str = "test", a: int = 5, c: float = 3.4):
    assert b == "test ok"
    assert a == 5
    assert c == 3.4


@click_test()
@_click.argument("b", default="test ok")
def test_click_ignore_defined_args(b: str = "test", a: int = 5, c: float = 3.4):
    assert b == "test ok"
    assert a == 5
    assert c == 3.4


@click_test("--a", "3", "--b", "tk", "--c", "1.8")
def test(b: str, a: int = 5, c: float = 3.4):
    assert b == "tk"
    assert a == 3
    assert c == 1.8


@click_test("--a")
def test_bool_true(a: bool = False):
    assert a


@click_test("--no-a")
def test_bool_false(a: bool = True):
    assert not a


@click_test("--no-a")
def test_bool_false_when_no_default(a: bool):
    assert not a


@click_test("--a")
def test_bool_true_when_no_default(a: bool):
    assert a


@click_test_error()
def test_bool_error_when_no_default(exc, a: bool):
    status, msg = exc
    assert status != 0
    assert "missing option" in str(msg).lower()


@click_test("--a", "test,test2")
def test_list_of_str(a: List[str]):
    assert len(a) == 2
    assert a[0] == "test"
    assert a[1] == "test2"


@click_test("--b", "1", "test,test2")
def test_tuple_of_list_of_str(b: t.Tuple[int, List[str]]):
    assert len(b) == 2
    assert b[0] == 1

    a = b[1]
    assert len(a) == 2
    assert a[0] == "test"
    assert a[1] == "test2"


@click_test_error("--help")
def test_list_of_str_help(e, a: List[str]):
    status, msg = e
    assert status == 0
    assert "LIST OF TEXTS" in msg


@click_test("--a", "\"test,\\\"' failed\",'test2, )\\'\" pased',ok")
def test_list_of_str2(a: List[str]):
    assert len(a) == 3
    assert a[0] == "test,\"' failed"
    assert a[1] == "test2, )'\" pased"


@click_test("--a", "1,2")
def test_list_of_int(a: List[int]):
    assert len(a) == 2
    assert a[0] == 1
    assert a[1] == 2


@click_test("--a", "1.5,2.5")
def test_list_of_float(a: List[float]):
    assert len(a) == 2
    assert a[0] == 1.5
    assert a[1] == 2.5


@click_test("--a", "true,false")
def test_list_of_bool(a: List[bool]):
    assert len(a) == 2
    assert a[0] is True
    assert a[1] is False


@click_test("--a", "true", "false")
def test_tuple_of_bool(a: Tuple[bool, bool]):
    assert len(a) == 2
    assert a[0] is True
    assert a[1] is False


@click_test()
def test_optional_str(a: Optional[str] = None):
    assert a is None


@click_test("--a", "test")
def test_optional_str2(a: Optional[str] = None):
    assert a == "test"


@click_test("--a", "test", "test2")
def test_tuple(a: Tuple[str, str]):
    assert len(a) == 2
    assert a[0] == "test"
    assert a[1] == "test2"


def test_unsupported_types_verified_at_declaration():
    with pytest.raises(ValueError) as excinfo:

        @click.command()
        def main(a: t.Callable[[], None]):
            pass

    assert "Could not build a Click parameter" in "\n".join(
        traceback.format_exception(None, excinfo.value, None)
    )


def test_unsupported_untyped_parameter():
    with pytest.raises(ValueError) as excinfo:

        @click.command()
        def main(a):
            pass

    assert "does not have it's type specified" in str(excinfo.value)


def test_unsupported_union_of_simple_types_and_classes():
    @dataclass
    class D:
        pass

    with pytest.raises(ValueError) as excinfo:

        @click.command()
        def main(a: t.Union[D, str]):
            pass

    assert "is not supported" in "\n".join(
        traceback.format_exception(None, excinfo.value, None)
    )


#
# Literal tests
#
@click_test("--a", "b")
def test_literal(a: Literal["a", "b"]):
    assert a == "b"


@click_test_error("--a", "b")
def test_literal_nonuniform_types(exc, a: Literal["a", 1]):
    pass


@click_test_error("--a", "b")
def test_literal_nonstr(exc, a: Literal[2, 1]):
    pass


#
# Test different initialization
#
def test_decorator_without_arguments(monkeypatch):
    import sys

    args = ["--a", "test"]
    monkeypatch.setattr(sys, "argv", ["prg.py"] + list(args))
    monkeypatch.setattr(sys, "exit", lambda *args, **kwargs: None)
    was_called = False

    @click.command
    def main(a: str):
        nonlocal was_called
        was_called = True

        assert a == "test"

    main()
    assert was_called, "Function was not called"


def test_decorator_with_different_class(monkeypatch):
    import sys

    args = ["--a", "test"]
    monkeypatch.setattr(sys, "argv", ["prg.py"] + list(args))
    monkeypatch.setattr(sys, "exit", lambda *args, **kwargs: None)
    was_called = False

    class Cmd(_click.Command):
        def invoke(self, ctx):
            nonlocal was_called
            assert ctx.params["a"] == "test"
            was_called = True

    @click.command(cls=Cmd)
    def main(a: str):
        raise RuntimeError("This should not have been called")

    main()
    assert was_called, "Function was not called"


def test_decorator_must_specify_either_callback_or_signature():
    with pytest.raises(ValueError) as excinfo:
        click.Command(name="test")
    assert "signature or callback must be specified" in str(excinfo.value)


def test_group_decorator_constructs_aclick_types():
    @click.group()
    def g():
        pass

    assert isinstance(g, click.Group)

    @click.group
    def g():
        pass

    assert isinstance(g, click.Group)

    @g.command()
    def a():
        """
        Test

        @param test: failed
        """
        pass

    assert isinstance(a, click.Command)
    assert "@param" not in a.help
    assert "Test" in a.help

    @g.group()
    def g2():
        pass

    assert isinstance(g2, click.Group)

    @g.group
    def g3():
        pass

    assert isinstance(g3, click.Group)

    @g2.command
    def b():
        """
        Test

        @param test: failed
        """
        pass

    assert isinstance(b, click.Command)
    assert "@param" not in b.help
    assert "Test" in b.help

    class G(click.Group):
        pass

    g.group_class = G

    @g.group
    def g4():
        pass

    assert isinstance(g4, G)

    G.group_class = type

    @g4.group()
    def g5():
        pass

    assert isinstance(g5, G)

    G.group_class = None

    class C(_click.Command):
        pass

    g.command_class = C

    @g.command()
    def c5():
        pass

    assert isinstance(c5, C)


def test_command_with_signature_only(monkeypatch):
    import sys

    args = ["--a", "test"]
    monkeypatch.setattr(sys, "argv", ["prg.py"] + list(args))
    monkeypatch.setattr(sys, "exit", lambda *args, **kwargs: None)
    was_called = False

    def b(a: str = "ok"):
        pass

    @click.command(signature=inspect.signature(b))
    def main(**kwargs):
        nonlocal was_called
        was_called = True
        assert kwargs == dict(a="test")

    main()
    assert was_called, "Function was not called"
