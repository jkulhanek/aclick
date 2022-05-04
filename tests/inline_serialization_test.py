import typing as t
from dataclasses import dataclass

import pytest
from aclick.utils import default_from_str


@pytest.mark.parametrize(
    ("tp", "value"), ((int, 2), (str, '"ok"'), (bool, False), (float, 3.14))
)
def test_serialize_dataclass(tp, value):
    @default_from_str
    @dataclass
    class D:
        a: tp = value

    c = D(tp(value + value))
    inst = D.from_str(str(c))
    assert isinstance(inst, D)
    assert inst.a == value + value


def test_serialize_dataclass_custom_name():
    @default_from_str
    @dataclass
    class D:
        a: int = 1

        @classmethod
        def _get_class_name(cls):
            return "b"

    c = D(2)
    inst = D.from_str(str(c))
    assert isinstance(inst, D)
    assert inst.a == 2
    assert str(c) == "b(a=2)"


def test_serialize_dataclass_list_of_strings_property():
    @default_from_str
    @dataclass
    class D:
        a: t.List[str]
        b: t.Optional[t.List[str]]

    c = D(["ok", '"ok"', 'c="ok"'], None)
    inst = D.from_str(str(c))
    assert isinstance(inst, D)
    assert str(c) == 'd(a=[ok, "\\"ok\\"", "c=\\"ok\\""], b=None)'


@pytest.mark.parametrize(
    ("tp", "value"), ((int, 2), (str, '"ok"'), (bool, False), (float, 3.14))
)
def test_serialize_dataclass_list(tp, value):
    @default_from_str
    @dataclass
    class D:
        a: t.Optional[t.List[tp]] = None

    c = D([value] * 2)
    inst = D.from_str(str(c))
    assert isinstance(inst, D)
    assert inst.a == [value] * 2


@pytest.mark.parametrize(
    ("tp", "value"), ((int, 2), (str, '"ok"'), (bool, False), (float, 3.14))
)
def test_serialize_dataclass_tuple(tp, value):
    @default_from_str
    @dataclass
    class D:
        a: t.Optional[t.Tuple[tp, tp]] = None

    c = D(tuple([value] * 2))
    inst = D.from_str(str(c))
    assert isinstance(inst, D)
    assert inst.a == tuple([value] * 2)
    assert str(c).startswith("d(a=(")
    assert str(c).endswith("))")


@pytest.mark.parametrize(
    ("tp", "value"), ((int, 2), (str, '"ok"'), (bool, False), (float, 3.14))
)
def test_serialize_dataclass_tuple_wrong_length(tp, value):
    @default_from_str
    @dataclass
    class D:
        a: t.Optional[t.Tuple[tp, tp, tp]] = None

    c = D(tuple([value] * 2))
    with pytest.raises(RuntimeError) as einfo:
        D.from_str(str(c))

    assert "Cannot parse" in str(einfo)


def test_serialize_dataclass_list_of_classes():
    @dataclass
    class B:
        c: str = "test"

    @default_from_str
    @dataclass
    class D:
        a: t.List[B]

    c = D([B("ok"), B("ok2")])
    inst = D.from_str(str(c))
    assert isinstance(inst, D)
    assert str(c) == "d(a=[b(c=ok), b(c=ok2)])"


def test_serialize_dataclass_tuple_of_classes():
    @dataclass
    class B:
        c: str = "test"

    @default_from_str
    @dataclass
    class D:
        a: t.Tuple[B, float]

    c = D((B("ok"), 3.14))
    inst = D.from_str(str(c))
    assert isinstance(inst, D)
    assert inst.a[1] == 3.14
    isinstance(inst.a[0], B)
    assert str(c) == "d(a=(b(c=ok), 3.14))"


@pytest.mark.parametrize("tp", [t.Dict, t.OrderedDict])
def test_serialize_dataclass_dict(tp):
    @dataclass
    class B:
        c: str = "test"

    @default_from_str
    @dataclass
    class D:
        a: tp[str, B] = "test"

    c = D({"2": B("ok"), "aa": B("fail")})
    inst = D.from_str(str(c))
    assert isinstance(inst, D)
    assert isinstance(inst.a, tp)
    assert isinstance(next(iter(inst.a.keys())), str)
    assert isinstance(next(iter(inst.a.values())), B)
    assert str(c) == "d(a={2=b(c=ok), aa=b(c=fail)})"


@pytest.mark.parametrize("tp", [t.Dict, t.OrderedDict])
def test_serialize_dataclass_dict_does_not_support_complex_keys(tp):
    @dataclass
    class B:
        c: str = "test"

    with pytest.raises(RuntimeError):

        @default_from_str
        @dataclass
        class D1:
            a: tp[B, B] = "test"

    for tp2 in (float, int, bool):
        with pytest.raises(RuntimeError):

            @default_from_str
            @dataclass
            class D:
                a: tp[tp2, B] = "test"
