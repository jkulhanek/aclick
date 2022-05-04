from dataclasses import dataclass

import aclick

from ._common import click_test, click_test_error  # noqa: F401


def test_remap_parameter_name(monkeypatch):
    @click_test(
        "--aa",
        "passed",
        map_parameter_name=aclick.RegexParameterRenamer([("b", "aa")]),
    )
    def main(b: str = "test"):
        assert b == "passed"

    main(monkeypatch)


def test_remap_parameter_hierarchical(monkeypatch):
    @dataclass
    class B:
        a: str = "test"

    @click_test(
        "--c",
        "ok",
        "--a",
        "passed",
        hierarchical=True,
        map_parameter_name=aclick.FlattenParameterRenamer(1),
    )
    def main(c: str, b: B):
        assert b.a == "passed"

    main(monkeypatch)


def test_regex_parameter_renamer():
    renamer = aclick.RegexParameterRenamer([("a", "b")])
    assert renamer("cc") == "cc"
    assert renamer("aacc") == "bbcc"

    renamer = aclick.RegexParameterRenamer([("a(.*)", r"b\1")])
    assert renamer("aacc") == "bacc"


def test_regex_parameter_renamer_multiple_patterns():
    renamer = aclick.RegexParameterRenamer(
        [
            ("a", "b"),
            ("c", "d"),
        ]
    )
    assert renamer("cc") == "dd"
    assert renamer("aacc") == "bbcc"

    renamer = aclick.RegexParameterRenamer([("a(.*)", r"b\1"), ("(.*)c", r"\1d")])
    assert renamer("aacc") == "bacc"
    assert renamer("kkcc") == "kkcd"


def test_regex_flatten_parameter_renamer():
    renamer = aclick.FlattenParameterRenamer(1)
    assert renamer("abc.def") == "def"
    assert renamer("abc") == "abc"
    assert renamer("abc.def.ghi") == "def.ghi"

    renamer = aclick.FlattenParameterRenamer(2)
    assert renamer("abc.def") == "def"
    assert renamer("abc") == "abc"
    assert renamer("abc.def.ghi") == "ghi"

    renamer = aclick.FlattenParameterRenamer(-1)
    assert renamer("abc.def") == "def"
    assert renamer("abc") == "abc"
    assert renamer("abc.def.ghi") == "ghi"
