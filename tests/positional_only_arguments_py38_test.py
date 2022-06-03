import traceback
import typing as t
from dataclasses import dataclass
from typing import List

import aclick
import aclick.utils
import pytest

from ._common import click_test, click_test_error  # noqa: F401


@dataclass
class D1:
    test: str


@dataclass
class D2:
    test: str = "ok"
    test2: int = 4


@dataclass
class D3:
    c: t.Union[D2]


@dataclass
class D4:
    c: D2


class D5(D2):
    def __init__(self, *args, c: D2, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c


@pytest.mark.parametrize(
    ("tp", "arg", "val"),
    [
        (str, "test", "test"),
        (int, "42", 42),
        (float, "3.14", 3.14),
        (bool, "true", True),
        (bool, "t", True),
        (bool, "yes", True),
        (bool, "1", True),
        (bool, "false", False),
        (bool, "f", False),
        (bool, "no", False),
        (bool, "0", False),
    ],
)
def test_simple_argument(monkeypatch, tp, arg, val):
    @click_test(arg)
    def main(a: tp, /):
        """

        :param a: no help here
        """
        assert a == val

    main(monkeypatch)

    @click_test(arg)
    def main(a: tp, b: int = 3, /):
        assert a == val
        assert b == 3

    main(monkeypatch)

    @click_test_error("--help")
    def main(err, a: tp, /):
        """

        :param a: no help here
        """
        status, msg = err
        assert status == 0
        assert "no help here" not in msg

    main(monkeypatch)


@pytest.mark.parametrize(
    ("tp", "arg", "val"),
    [
        (str, "test", "test"),
        (int, "42", 42),
        (float, "3.14", 3.14),
        (bool, "true", True),
        (bool, "t", True),
        (bool, "yes", True),
        (bool, "1", True),
        (bool, "false", False),
        (bool, "f", False),
        (bool, "no", False),
        (bool, "0", False),
    ],
)
def test_simple_optional_argument(monkeypatch, tp, arg, val):
    @click_test(arg)
    def main(a: t.Optional[tp], /):
        assert a == val

    main(monkeypatch)


@pytest.mark.parametrize(
    ("tp", "arg", "val"),
    [
        (str, "test", "test"),
        (int, "42", 42),
        (float, "3.14", 3.14),
        (bool, "true", True),
        (bool, "t", True),
        (bool, "yes", True),
        (bool, "1", True),
        (bool, "false", False),
        (bool, "f", False),
        (bool, "no", False),
        (bool, "0", False),
    ],
)
def test_args_kwargs(monkeypatch, tp, arg, val):
    @click_test(arg)
    def main(a: tp, b: int = 3, /, *args, **kwargs):
        assert a == val
        assert b == 3

    main(monkeypatch)


#
# Inline types
#
@click_test('d1("passed")')
def test_dataclass_argument(a: D1, /):
    assert isinstance(a, D1)
    assert a.test == "passed"


@click_test("d1(\"passed\"),d1('passed as, well')")
def test_list_of_dataclasses_argument(a: List[D1], /):
    assert len(a) == 2
    assert isinstance(a[0], D1)
    assert a[0].test == "passed"
    assert isinstance(a[1], D1)
    assert a[1].test == "passed as, well"


@click_test("p1=d1(\"passed\"),p2=d1('passed as, well')")
def test_dict_of_dataclasses_argument(a: t.Dict[str, D1], /):
    assert len(a) == 2
    assert isinstance(a["p1"], D1)
    assert a["p1"].test == "passed"
    assert isinstance(a["p2"], D1)
    assert a["p2"].test == "passed as, well"


@click_test("d1(\"passed\"),d2('passed as, well')")
def test_list_of_union_of_dataclasses_argument(a: List[t.Union[D1, D2]], /):
    assert len(a) == 2
    assert isinstance(a[0], D1)
    assert a[0].test == "passed"
    assert isinstance(a[1], D2)
    assert a[1].test == "passed as, well"


@click_test("d1(\"passed\"),d1('passed as, well')")
def test_list_of_optional_dataclasses_argument(a: List[t.Optional[D1]], /):
    assert len(a) == 2
    assert isinstance(a[0], D1)
    assert a[0].test == "passed"
    assert isinstance(a[1], D1)
    assert a[1].test == "passed as, well"


@click_test("d2()")
def test_dataclass_default_argument(a: D2, /):
    assert isinstance(a, D2)
    assert a.test == "ok"


@click_test("d2()")
def test_union_of_dataclasses_argument(a: t.Union[D1, D2], /):
    assert a is not None
    assert isinstance(a, D2)
    assert a.test == "ok"


@click_test('d3(c=d2(test="ok"))')
def test_hierarchical_dataclasses_argument(a: D3, /):
    assert isinstance(a, D3)
    assert isinstance(a.c, D2)
    assert a.c.test == "ok"


@click_test('d4(c=d2(test="ok"))')
def test_hierarchical_dataclasses2_argument(a: D4, /):
    assert isinstance(a, D4)
    assert isinstance(a.c, D2)
    assert a.c.test == "ok"


@click_test('d5("ok2",c=d2(test="ok"))')
def test_hierarchical_classes_argument(a: D5, /):
    assert isinstance(a, D5)
    assert isinstance(a.c, D2)
    assert a.c.test == "ok"
    assert a.test == "ok2"


#
# Hierarchical parameters
#
@click_test('d1("passed")', hierarchical=True)
def test_dataclass_hierarchical_argument_parsed_inline(a: D1, /):
    assert isinstance(a, D1)
    assert a.test == "passed"


def test_dataclass_positional_argument_only_on_top_level():
    class DPos:
        def __init__(self, a: str, /):
            pass

    with pytest.raises(ValueError) as exinfo:

        @aclick.command(hierarchical=True)
        def test_dataclass_no_default(a: DPos):
            pass

    assert "arguments are allowed only on top level" in str(exinfo.value)


def test_unsupported_types_verified_at_declaration():
    with pytest.raises(ValueError) as excinfo:

        @aclick.command()
        def main(a: t.Callable[[], None], /):
            pass

    assert "Could not build a Click" in "\n".join(
        traceback.format_exception(None, excinfo.value, None)
    )


def test_unsupported_untyped_parameter():
    with pytest.raises(ValueError) as excinfo:

        @aclick.command()
        def main(a, /):
            pass

    assert "does not have it's type specified" in str(excinfo.value)


def test_unsupported_union_of_simple_types_and_classes():
    @dataclass
    class D:
        pass

    with pytest.raises(ValueError) as excinfo:

        @aclick.command()
        def main(a: t.Union[D, str], /):
            pass

    assert "is not supported" in "\n".join(
        traceback.format_exception(None, excinfo.value, None)
    )


def test_custom_class(monkeypatch):
    class ModelA:
        def __init__(self, name: str, /, n_layers: int):
            self.name = name
            self.n_layers = n_layers

    class ModelB(ModelA):
        def __init__(self, *args, n_blocks: int = 5, **kwargs):
            super().__init__(*args, **kwargs)
            self.n_blocks = n_blocks

    @click_test("--model", "model_b(test, n_layers=2)", hierarchical=False)
    def main(model: t.Union[ModelA, ModelB]):
        print(
            "Training model "
            + model.__class__.__name__
            + f" with {model.n_layers} layers."
        )

    main(monkeypatch)
