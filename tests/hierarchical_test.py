import typing as t
from dataclasses import dataclass

import aclick
import click
import pytest

from ._common import click_test, click_test_error


@dataclass
class D1:
    test: str


@click_test("--a-test", "passed", hierarchical=True)
def test_dataclass(a: D1):
    assert isinstance(a, D1)
    assert a.test == "passed"


def test_dataclass_no_default_error():
    with pytest.raises(ValueError) as exinfo:

        @aclick.command(hierarchical=True)
        def test_dataclass_no_default(a: D1 = D1("ok")):
            pass

    assert "Cannot use non-default parameter with hierarchical parsing" in str(
        exinfo.value
    )


def test_dataclass_union_nonuniform_error():
    with pytest.raises(ValueError) as exinfo:

        @aclick.command(hierarchical=True)
        def test_dataclass_no_default(a: t.Union[D1, str]):
            pass

    assert "is not supported" in str(exinfo.value)


@dataclass
class D2:
    test: str = "ok"
    test2: int = 4


@click_test(hierarchical=True)
def test_dataclass_default(a: D2):
    assert isinstance(a, D2)
    assert a.test == "ok"


@click_test(hierarchical=True)
def test_dataclass_optional(a: t.Optional[D2] = None):
    assert a is None


@click_test("--a", "d2", hierarchical=True)
def test_union_of_dataclasses(a: t.Union[D1, D2]):
    assert a is not None
    assert isinstance(a, D2)
    assert a.test == "ok"


@click_test_error("--a", "d5", hierarchical=True)
@click.option("--a", type=str)
def test_union_of_dataclasses_raises_error_for_unknown_class(error, a: t.Union[D1, D2]):
    status, msg = error
    assert status != 0
    assert "was not found in the set of supported classes {d1, d2}" in msg


@click_test_error("--a", "d3", hierarchical=True)
def test_union_of_dataclasses_unknown_class_error(error, a: t.Union[D1, D2]):
    pass


@click_test("--a", "d2", "--a-test2", "1", hierarchical=True)
def test_union_of_dataclasses2(a: t.Union[D1, D2]):
    assert a is not None
    assert isinstance(a, D2)
    assert a.test == "ok"
    assert a.test2 == 1


@dataclass
class D4:
    c: D2


@click_test("--a-c-test", "ok", hierarchical=True)
def test_hierarchical_dataclasses2(a: D4):
    assert isinstance(a, D4)
    assert isinstance(a.c, D2)
    assert a.c.test == "ok"


class D5(D2):
    def __init__(self, *args, c: D2, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c


@click_test("--a-c-test", "ok", "--a-test", "ok2", hierarchical=True)
def test_hierarchical_classes(a: D5):
    assert isinstance(a, D5)
    assert isinstance(a.c, D2)
    assert a.c.test == "ok"
    assert a.test == "ok2"
