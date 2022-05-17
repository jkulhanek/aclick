import typing as t
from dataclasses import dataclass
from typing import List, Optional

from ._common import click_test, click_test_error  # noqa: F401


@dataclass
class D1:
    test: str


@click_test("--a", 'd1("passed")', hierarchical=False)
def test_dataclass(a: D1):
    assert isinstance(a, D1)
    assert a.test == "passed"


def test_dataclass_camelcase(monkeypatch):
    @dataclass
    class CammelCase:
        pass

    @click_test("--a", "cammel-case()", hierarchical=False)
    def main(a: CammelCase):
        assert isinstance(a, CammelCase)

    main(monkeypatch)


@click_test("--a", "d1(\"passed\"),d1('passed as, well')", hierarchical=False)
def test_list_of_dataclasses(a: List[D1]):
    assert len(a) == 2
    assert isinstance(a[0], D1)
    assert a[0].test == "passed"
    assert isinstance(a[1], D1)
    assert a[1].test == "passed as, well"


@dataclass
class D2:
    test: str = "ok"


@click_test("--a", "d1(\"passed\"),d2('passed as, well')", hierarchical=False)
def test_list_of_union_of_dataclasses(a: List[t.Union[D1, D2]]):
    assert len(a) == 2
    assert isinstance(a[0], D1)
    assert a[0].test == "passed"
    assert isinstance(a[1], D2)
    assert a[1].test == "passed as, well"


@click_test("--a", "d1(\"passed\"),d1('passed as, well')", hierarchical=False)
def test_list_of_optional_dataclasses(a: List[t.Optional[D1]]):
    assert len(a) == 2
    assert isinstance(a[0], D1)
    assert a[0].test == "passed"
    assert isinstance(a[1], D1)
    assert a[1].test == "passed as, well"


@click_test("--a", "d2()", hierarchical=False)
def test_dataclass_default(a: D2):
    assert isinstance(a, D2)
    assert a.test == "ok"


@click_test(hierarchical=False)
def test_dataclass_optional(a: Optional[D2] = None):
    assert a is None


@click_test("--a", "d2()", hierarchical=False)
def test_union_of_dataclasses(a: t.Union[D1, D2]):
    assert a is not None
    assert isinstance(a, D2)
    assert a.test == "ok"


@dataclass
class D3:
    c: t.Union[D2]


@click_test("--a", 'd3(c=d2(test="ok"))', hierarchical=False)
def test_hierarchical_dataclasses(a: D3):
    assert isinstance(a, D3)
    assert isinstance(a.c, D2)
    assert a.c.test == "ok"


@dataclass
class D4:
    c: D2


@click_test("--a", 'd4(c=d2(test="ok"))', hierarchical=False)
def test_hierarchical_dataclasses2(a: D4):
    assert isinstance(a, D4)
    assert isinstance(a.c, D2)
    assert a.c.test == "ok"


class D5(D2):
    def __init__(self, *args, c: D2, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c


@click_test("--a", 'd5("ok2",c=d2(test="ok"))', hierarchical=False)
def test_hierarchical_classes(a: D5):
    assert isinstance(a, D5)
    assert isinstance(a.c, D2)
    assert a.c.test == "ok"
    assert a.test == "ok2"


@click_test_error('--a', 'dd()')
def test_error_message_contains_example(err, a: D5):
    status, msg = err
    assert status != 0
    assert '''d5(
  test={optional str},
  c=d2(
    test={optional str}))''' in msg.lower()


def test_parameter_help_contains_example(monkeypatch):
    @click_test_error('--help', '--a', 'dd()')
    def main(err, a: D5):
        status, msg = err
        assert status == 0
        assert '''d5(test={optionalstr},c=d2(test={optionalstr}))''' not in msg.lower().replace(' ', '').replace('\n', '')

    main(monkeypatch)

    @click_test_error('--help', '--a', 'dd()', num_inline_examples_help=-1)
    def main(err, a: D5):
        status, msg = err
        assert status == 0
        assert '''d5(test={optionalstr},c=d2(test={optionalstr}))''' in msg.lower().replace(' ', '').replace('\n', '')

    main(monkeypatch)

    @click_test_error('--help', '--a', 'dd()')
    def main(err, a: t.Union[D5, D2]):
        status, msg = err
        assert status == 0
        assert '''d5(test={optionalstr},c=d2(test={optionalstr}))''' not in msg.lower().replace(' ', '').replace('\n', '')

    main(monkeypatch)

    @click_test_error('--help', '--a', 'dd()', num_inline_examples_help=-1)
    def main(err, a: t.Union[D5, D2]):
        status, msg = err
        assert status == 0
        assert '''d5(test={optionalstr},c=d2(test={optionalstr}))''' in msg.lower().replace(' ', '').replace('\n', '')

    main(monkeypatch)

    @click_test_error('--help', '--a', 'dd()', num_inline_examples_help=1)
    def main(err, a: t.Union[D2, D5]):
        status, msg = err
        assert status == 0
        assert '''d5(test={optionalstr},c=d2(test={optionalstr}))''' not in msg.lower().replace(' ', '').replace('\n', '')

    main(monkeypatch)
