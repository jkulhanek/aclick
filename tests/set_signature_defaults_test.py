import inspect
import json
import typing as t
from dataclasses import dataclass
from tempfile import NamedTemporaryFile

import aclick
import pytest
from aclick.configuration import restrict_parse_configuration

from aclick.utils import _fill_signature_defaults_from_dict, _is_class, Literal

from ._common import _call_fn_empty, click_test, click_test_error


def _store_config(cfg, type="json"):
    file = NamedTemporaryFile("w+", suffix=f".{type}")
    assert type in {"json", "yaml"}
    if type == "json":
        json.dump(cfg, file)
    elif type == "yaml":
        import yaml

        yaml.safe_dump(cfg, file)
    file.flush()
    return file


def test_set_signature_defaults_simple_types():
    @_fill_signature_defaults_from_dict(dict(a=6))
    def fn(a: int = 5):
        assert a == 6

    fn()

    @_fill_signature_defaults_from_dict(dict(a=6.1))
    def fn(a: float = 5.1):
        assert a == 6.1

    fn()

    @_fill_signature_defaults_from_dict(dict(a=False))
    def fn(a: bool = True):
        assert a is False

    fn()

    @_fill_signature_defaults_from_dict(dict(a="ok"))
    def fn(a: str = "fail"):
        assert a == "ok"

    fn()

    @_fill_signature_defaults_from_dict(dict(a="ok"))
    def fn(a: Literal["ok", "fail"] = "fail"):
        assert a == "ok"

    fn()


def test_from_dict_simple_types():
    was_called = False

    def fn(a: int = 5):
        nonlocal was_called
        was_called = True
        assert a == 6

    aclick.utils.from_dict(fn, dict(a=6))
    assert was_called

    def fn(a: float = 5.1):
        assert a == 6.1

    aclick.utils.from_dict(fn, dict(a=6.1))

    @_fill_signature_defaults_from_dict(dict(a=False))
    def fn(a: bool = True):
        assert a is False

    aclick.utils.from_dict(fn, dict(a=False))

    @_fill_signature_defaults_from_dict(dict(a="ok"))
    def fn(a: str = "fail"):
        assert a == "ok"

    aclick.utils.from_dict(fn, dict(a="ok"))

    @_fill_signature_defaults_from_dict(dict(a="ok"))
    def fn(a: Literal["ok", "fail"] = "fail"):
        assert a == "ok"

    aclick.utils.from_dict(fn, dict(a="ok"))


def test_set_signature_defaults_class_type():
    @dataclass
    class A:
        a: str = "test"

    @_fill_signature_defaults_from_dict(dict(a=dict(a="ok")))
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

    @_fill_signature_defaults_from_dict(dict(a=dict(a=True)))
    def fn(a: A):
        assert isinstance(a, A)
        assert type(a) == A
        assert a.a is True

    sig = inspect.signature(fn)
    assert _is_class(sig.parameters["a"].annotation)
    assert issubclass(sig.parameters["a"].annotation, A)
    _call_fn_empty(fn)


def test_from_dict_class_type():
    @dataclass
    class A:
        a: str = "test"

    def fn(a: A):
        assert isinstance(a, A)
        assert type(a) == A
        assert a.a == "ok"

    aclick.utils.from_dict(fn, dict(a=dict(a="ok")))

    @dataclass
    class A:
        a: bool = False

    def fn(a: A):
        assert isinstance(a, A)
        assert type(a) == A
        assert a.a is True

    aclick.utils.from_dict(fn, dict(a=dict(a=True)))


def test_set_signature_defaults_optional_type():
    @dataclass
    class A:
        a: t.Optional[str] = None

    @_fill_signature_defaults_from_dict(dict(a=dict(a="ok")))
    def fn(a: A):
        assert isinstance(a, A)
        assert type(a) == A
        assert a.a == "ok"

    sig = inspect.signature(fn)
    assert _is_class(sig.parameters["a"].annotation)
    assert issubclass(sig.parameters["a"].annotation, A)
    _call_fn_empty(fn)

    @_fill_signature_defaults_from_dict(dict(a="ok"))
    def fn(a: t.Optional[str]):
        assert a == "ok"
        assert isinstance(a, str)

    _call_fn_empty(fn)

    @_fill_signature_defaults_from_dict(dict(a=dict()))
    def fn(a: t.Optional[A]):
        assert a is not None
        assert a.a is None

    _call_fn_empty(fn)

    @_fill_signature_defaults_from_dict(dict(a=dict(a="ok")))
    def fn(a: t.Optional[A]):
        assert a is not None
        assert a.a == "ok"

    _call_fn_empty(fn)


def test_from_dict_optional_type():
    @dataclass
    class A:
        a: t.Optional[str] = None

    def fn(a: A):
        assert isinstance(a, A)
        assert type(a) == A
        assert a.a == "ok"

    aclick.utils.from_dict(fn, dict(a=dict(a="ok")))

    def fn(a: t.Optional[str]):
        assert a == "ok"
        assert isinstance(a, str)

    aclick.utils.from_dict(fn, dict(a="ok"))

    def fn(a: t.Optional[A]):
        assert a is not None
        assert a.a is None

    aclick.utils.from_dict(fn, dict(a=dict()))

    def fn(a: t.Optional[A]):
        assert a is not None
        assert a.a == "ok"

    aclick.utils.from_dict(fn, dict(a=dict(a="ok")))


def test_set_signature_defaults_union_type():
    @dataclass
    class A:
        a: t.Union[str, int] = None

    @_fill_signature_defaults_from_dict(dict(a=dict(a="ok")))
    def fn(a: A):
        assert isinstance(a, A)
        assert type(a) == A
        assert a.a == "ok"

    sig = inspect.signature(fn)
    assert _is_class(sig.parameters["a"].annotation)
    assert issubclass(sig.parameters["a"].annotation, A)
    _call_fn_empty(fn)

    @_fill_signature_defaults_from_dict(dict(a=dict(a=3)))
    def fn(a: A):
        assert isinstance(a, A)
        assert type(a) == A
        assert a.a == 3

    sig = inspect.signature(fn)
    assert _is_class(sig.parameters["a"].annotation)
    assert issubclass(sig.parameters["a"].annotation, A)
    _call_fn_empty(fn)

    @_fill_signature_defaults_from_dict(dict(a="ok"))
    def fn(a: t.Union[str, int]):
        assert a == "ok"
        assert isinstance(a, str)

    _call_fn_empty(fn)

    @_fill_signature_defaults_from_dict(dict(a=3))
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

    @_fill_signature_defaults_from_dict(dict(a=dict(__class__="c", a="ok", b=4)))
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

    @_fill_signature_defaults_from_dict(dict(a=dict(a="ok")))
    def fn(a: t.Union[A, B]):
        pass

    sig = inspect.signature(fn)
    assert getattr(sig.parameters["a"].annotation, "__origin__", None) is t.Union
    i = -1
    for i, c in enumerate(sig.parameters["a"].annotation.__args__):
        assert c().a == "ok"
    assert i == 1


def test_from_dict_union_class_type():
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

    def fn(a: t.Union[A, B]):
        assert isinstance(a, B)
        assert type(a) == B
        assert a.a == "ok"
        assert a.b == 4
        assert a.c == 2

    aclick.utils.from_dict(fn, dict(a=dict(__class__="c", a="ok", b=4)))

    with pytest.raises(RuntimeError):

        def fn(a: t.Union[A, B]):
            pass

        aclick.utils.from_dict(fn, dict(a=dict(a="ok")))


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


def test_config_option_list_type(monkeypatch):
    with _store_config(dict(a=["ok", "ok2"])) as cfg:

        @click_test("--config", cfg.name)
        @aclick.configuration_option("--config")
        def main(a: t.List[str]):
            assert isinstance(a, list)
            assert a == ["ok", "ok2"]

        main(monkeypatch)

    with _store_config(dict(a=["ok", '"jonas"'])) as cfg:

        @click_test_error("--config", cfg.name, "--help")
        @aclick.configuration_option("--config")
        def main(err, a: t.List[str] = ["ok", "ok2"]):
            status, msg = err
            assert status == 0
            assert 'ok, "\\"jonas\\""' in msg

        main(monkeypatch)


def test_config_option_from_str_type(monkeypatch):
    @dataclass
    class A:
        x: str

        @classmethod
        def from_str(cls, val):
            return A("ok" + val)

        def __str__(self):
            assert self.x.startswith("ok")
            return self.x[2:]

    with _store_config(dict(a="cls")) as cfg:

        @click_test("--config", cfg.name)
        @aclick.configuration_option("--config")
        def main(a: A):
            assert isinstance(a, A)
            assert a.x == "okcls"

        main(monkeypatch)

    with _store_config(dict(a="cls")) as cfg:

        @click_test("--config", cfg.name)
        @aclick.configuration_option("--config")
        def main(a: t.Optional[A]):
            assert isinstance(a, A)
            assert a.x == "okcls"

        main(monkeypatch)

    with _store_config(dict(a="jclass")) as cfg:

        @click_test_error("--config", cfg.name, "--help")
        @aclick.configuration_option("--config")
        def main(err, a: A = A("cls")):
            status, msg = err
            assert status == 0
            assert "jclass" in msg

        main(monkeypatch)


def test_config_option_yaml(monkeypatch):
    @dataclass
    class A:
        b: str = "ok"

    with _store_config(dict(a=dict(b="passed")), "yaml") as cfg:

        @click_test("--configuration", cfg.name)
        @aclick.configuration_option()
        def main(a: A):
            assert isinstance(a, A)
            assert a.b == "passed"

        main(monkeypatch)

    with _store_config(dict(a=dict(b="passed")), "yaml") as cfg:

        @click_test("--config", cfg.name)
        @aclick.configuration_option("--config")
        def main(a: A):
            assert isinstance(a, A)
            assert a.b == "passed"

        main(monkeypatch)


def test_restrict_parse_configuration(monkeypatch):
    @dataclass
    class A:
        b: str = "ok"

    with _store_config(dict(b="passed")) as cfg:

        @click_test("--configuration", cfg.name)
        @aclick.configuration_option(
            parse_configuration=restrict_parse_configuration("a")
        )
        def main(a: A):
            assert isinstance(a, A)
            assert a.b == "passed"

        main(monkeypatch)

    with _store_config(dict(b="passed"), "yaml") as cfg:

        @click_test("--configuration", cfg.name)
        @aclick.configuration_option(
            parse_configuration=restrict_parse_configuration("a")
        )
        def main(a: A):
            assert isinstance(a, A)
            assert a.b == "passed"

        main(monkeypatch)
