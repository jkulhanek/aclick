import inspect
import json
import typing as t
from dataclasses import dataclass
from tempfile import NamedTemporaryFile

import aclick

from aclick.utils import _is_class, fill_signature_defaults_from_config, Literal

from ._common import _call_fn_empty, click_test


def _store_config(cfg):
    file = NamedTemporaryFile("w+")
    json.dump(cfg, file)
    file.flush()
    return file


def test_set_signature_defaults_simple_types():
    @fill_signature_defaults_from_config(dict(a=6))
    def fn(a: int = 5):
        assert a == 6

    fn()

    @fill_signature_defaults_from_config(dict(a=6.1))
    def fn(a: float = 5.1):
        assert a == 6.1

    fn()

    @fill_signature_defaults_from_config(dict(a=False))
    def fn(a: bool = True):
        assert a is False

    fn()

    @fill_signature_defaults_from_config(dict(a="ok"))
    def fn(a: str = "fail"):
        assert a == "ok"

    fn()

    @fill_signature_defaults_from_config(dict(a="ok"))
    def fn(a: Literal["ok", "fail"] = "fail"):
        assert a == "ok"

    fn()


def test_set_signature_defaults_class_type():
    @dataclass
    class A:
        a: str = "test"

    @fill_signature_defaults_from_config(dict(a=dict(a="ok")))
    def fn(a: A):
        assert isinstance(a, A)
        assert type(a) == A
        assert a.a == "ok"

    sig = inspect.signature(fn)
    assert _is_class(sig.parameters["a"].annotation)
    assert issubclass(sig.parameters["a"].annotation, A)
    _call_fn_empty(fn)

    @dataclass
    class A:
        a: bool = False

    @fill_signature_defaults_from_config(dict(a=dict(a=True)))
    def fn(a: A):
        assert isinstance(a, A)
        assert type(a) == A
        assert a.a is True

    sig = inspect.signature(fn)
    assert _is_class(sig.parameters["a"].annotation)
    assert issubclass(sig.parameters["a"].annotation, A)
    _call_fn_empty(fn)


def test_set_signature_defaults_optional_type():
    @dataclass
    class A:
        a: t.Optional[str] = None

    @fill_signature_defaults_from_config(dict(a=dict(a="ok")))
    def fn(a: A):
        assert isinstance(a, A)
        assert type(a) == A
        assert a.a == "ok"

    sig = inspect.signature(fn)
    assert _is_class(sig.parameters["a"].annotation)
    assert issubclass(sig.parameters["a"].annotation, A)
    _call_fn_empty(fn)

    @fill_signature_defaults_from_config(dict(a="ok"))
    def fn(a: t.Optional[str]):
        assert a == "ok"
        assert isinstance(a, str)

    _call_fn_empty(fn)

    @fill_signature_defaults_from_config(dict(a=dict()))
    def fn(a: t.Optional[A]):
        assert a is not None
        assert a.a is None

    _call_fn_empty(fn)

    @fill_signature_defaults_from_config(dict(a=dict(a="ok")))
    def fn(a: t.Optional[A]):
        assert a is not None
        assert a.a == "ok"

    _call_fn_empty(fn)


def test_set_signature_defaults_union_type():
    @dataclass
    class A:
        a: t.Union[str, int] = None

    @fill_signature_defaults_from_config(dict(a=dict(a="ok")))
    def fn(a: A):
        assert isinstance(a, A)
        assert type(a) == A
        assert a.a == "ok"

    sig = inspect.signature(fn)
    assert _is_class(sig.parameters["a"].annotation)
    assert issubclass(sig.parameters["a"].annotation, A)
    _call_fn_empty(fn)

    @fill_signature_defaults_from_config(dict(a=dict(a=3)))
    def fn(a: A):
        assert isinstance(a, A)
        assert type(a) == A
        assert a.a == 3

    sig = inspect.signature(fn)
    assert _is_class(sig.parameters["a"].annotation)
    assert issubclass(sig.parameters["a"].annotation, A)
    _call_fn_empty(fn)

    @fill_signature_defaults_from_config(dict(a="ok"))
    def fn(a: t.Union[str, int]):
        assert a == "ok"
        assert isinstance(a, str)

    _call_fn_empty(fn)

    @fill_signature_defaults_from_config(dict(a=3))
    def fn(a: t.Union[str, int]):
        assert a == 3
        assert isinstance(a, int)

    _call_fn_empty(fn)


def test_set_signature_defaults_union_class_type():
    @dataclass
    class A:
        a: str = "test"

    @dataclass
    class B:
        a: str = "passed"
        b: int = 3
        c: int = 2

        @classmethod
        def _get_class_name(cls):
            return "c"

    @fill_signature_defaults_from_config(dict(a=dict(__class__="c", a="ok", b=4)))
    def fn(a: t.Union[A, B]):
        assert isinstance(a, B)
        assert type(a) == B
        assert a.a == "ok"
        assert a.b == 4
        assert a.c == 2

    sig = inspect.signature(fn)
    assert _is_class(sig.parameters["a"].annotation)
    assert issubclass(sig.parameters["a"].annotation, B)
    _call_fn_empty(fn)

    @fill_signature_defaults_from_config(dict(a=dict(a="ok")))
    def fn(a: t.Union[A, B]):
        pass

    sig = inspect.signature(fn)
    assert getattr(sig.parameters["a"].annotation, "__origin__", None) is t.Union
    i = -1
    for i, c in enumerate(sig.parameters["a"].annotation.__args__):
        assert c().a == "ok"
    assert i == 1


def test_config_option(monkeypatch):
    @dataclass
    class A:
        b: str = "ok"

    with _store_config(dict(a=dict(b="passed"))) as cfg:

        @click_test("--configuration", cfg.name)
        @aclick.configuration_option()
        def main(a: A):
            assert isinstance(a, A)
            assert a.b == "passed"

        main(monkeypatch)

    with _store_config(dict(a=dict(b="passed"))) as cfg:

        @click_test("--config", cfg.name)
        @aclick.configuration_option("--config")
        def main(a: A):
            assert isinstance(a, A)
            assert a.b == "passed"

        main(monkeypatch)


def test_config_option_union_type(monkeypatch):
    @dataclass
    class A:
        a: str = "test"

    @dataclass
    class B:
        a: str = "passed"
        b: int = 3
        c: int = 2

        @classmethod
        def _get_class_name(cls):
            return "c"

    with _store_config(dict(a=dict(__class__="c", a="ok", b=4))) as cfg:

        @click_test("--config", cfg.name)
        @aclick.configuration_option("--config")
        def main(a: t.Union[A, B]):
            assert isinstance(a, B)
            assert type(a) == B
            assert a.a == "ok"
            assert a.b == 4
            assert a.c == 2

        main(monkeypatch)
