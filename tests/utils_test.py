import inspect
import typing as t
from collections import OrderedDict
from dataclasses import dataclass

import aclick.utils
import pytest
from aclick.utils import Literal


def test_is_class_for_ordereddict():
    assert not aclick.utils._is_class(t.OrderedDict[str, str])


def test_from_str_class_not_setting_kwargs():
    @aclick.utils.default_from_str
    class A:
        def __init__(self, a: int):
            pass

    with pytest.raises(RuntimeError) as excinfo:
        x = A.from_str("a(a=1)")
        str(x)

    assert (
        "because the constructor parameter a is missing from class's properties"
        in str(excinfo.value).lower()
    )


def test_from_str_class_no_type_annotation_kwargs():
    with pytest.raises(ValueError) as excinfo:

        @aclick.utils.default_from_str()
        class A:
            def __init__(self, a):
                pass

    assert "does not have a type annotation" in str(excinfo.value).lower()


def test_from_str_class_unsupported_type_kwargs():
    with pytest.raises(ValueError) as excinfo:

        @aclick.utils.default_from_str()
        class A:
            def __init__(self, a: t.Callable):
                pass

    assert "is not supported because it contains type" in str(excinfo.value).lower()


def test_from_str_literal_kwargs():
    @aclick.utils.default_from_str()
    class A:
        def __init__(self, a: Literal["test", "ok"]):
            self.a = a

    x = A(a="ok")
    assert A.from_str(str(x)).a == x.a
    assert str(x) == "a(a=ok)"


def test_from_str_class_from_str_object_kwargs():
    class A:
        def __init__(self, a):
            pass

        @staticmethod
        def from_str(s):
            return 42

    @aclick.utils.default_from_str()
    class B:
        def __init__(self, a: A):
            self.a = a

    x = B.from_str("b(a(5))")
    assert isinstance(x, B)
    assert x.a == 42


def test_from_str_unsupported_literal_kwargs():
    with pytest.raises(ValueError) as excinfo:

        @aclick.utils.default_from_str()
        class A:
            def __init__(self, a: Literal[1, "ok"]):
                self.a = a

    assert (
        "is not supported because it contains a literal property of non-str types"
        in str(excinfo.value).lower()
    )


def test_from_str_invalid_literal_kwargs():
    @aclick.utils.default_from_str()
    class A:
        def __init__(self, a: Literal["test", "ok"]):
            self.a = a

    with pytest.raises(RuntimeError) as excinfo:
        A.from_str("a(a=failed)")

    assert (
        "literal value failed is not in the set of supported values {ok, test}"
        in str(excinfo.value).lower()
    )


#
# Test _ClassArgument raises errors correctly
#
def test_class_argument_error_unfinished_class():
    with pytest.raises(aclick.utils._ClassArgument.ParseError) as excinfo:
        aclick.utils._ClassArgument.from_str("a(")
    assert "unexpected end of argument" in str(excinfo.value).lower()

    with pytest.raises(aclick.utils._ClassArgument.ParseError) as excinfo:
        aclick.utils._ClassArgument.from_str("a(b=x(")
    assert "unexpected end of argument" in str(excinfo.value).lower()

    with pytest.raises(aclick.utils._ClassArgument.ParseError) as excinfo:
        aclick.utils._ClassArgument.from_str('a(b=x(test="ok")')
    assert "unexpected end of argument" in str(excinfo.value).lower()


def test_class_argument_error_unfinished_list():
    with pytest.raises(aclick.utils._ClassArgument.ParseError) as excinfo:
        aclick.utils._ClassArgument.from_str("a([")
    assert "unexpected end of argument" in str(excinfo.value).lower()

    with pytest.raises(aclick.utils._ClassArgument.ParseError) as excinfo:
        aclick.utils._ClassArgument.from_str("a(b=[")
    assert "unexpected end of argument" in str(excinfo.value).lower()

    with pytest.raises(aclick.utils._ClassArgument.ParseError) as excinfo:
        aclick.utils._ClassArgument.from_str('a(b=[(test="ok"), ')
    assert "unexpected end of argument" in str(excinfo.value).lower()


def test_class_argument_error_unfinished_tuple():
    with pytest.raises(aclick.utils._ClassArgument.ParseError) as excinfo:
        aclick.utils._ClassArgument.from_str("a((")
    assert "unexpected end of argument" in str(excinfo.value).lower()

    with pytest.raises(aclick.utils._ClassArgument.ParseError) as excinfo:
        aclick.utils._ClassArgument.from_str("a(b=(")
    assert "unexpected end of argument" in str(excinfo.value).lower()

    with pytest.raises(aclick.utils._ClassArgument.ParseError) as excinfo:
        aclick.utils._ClassArgument.from_str('a(b=((test="ok"), ')
    assert "unexpected end of argument" in str(excinfo.value).lower()


def test_class_argument_error_unfinished_dict():
    with pytest.raises(aclick.utils._ClassArgument.ParseError) as excinfo:
        aclick.utils._ClassArgument.from_str("a({")
    assert "unexpected end of argument" in str(excinfo.value).lower()

    with pytest.raises(aclick.utils._ClassArgument.ParseError) as excinfo:
        aclick.utils._ClassArgument.from_str("a({b=")
    assert "unexpected end of argument" in str(excinfo.value).lower()

    with pytest.raises(aclick.utils._ClassArgument.ParseError) as excinfo:
        aclick.utils._ClassArgument.from_str('a(b={test="ok", ')
    assert "unexpected end of argument" in str(excinfo.value).lower()

    with pytest.raises(aclick.utils._ClassArgument.ParseError) as excinfo:
        aclick.utils._ClassArgument.from_str('{test="ok", "ok"}')
    assert "missing dict entry name" in str(excinfo.value).lower()


def test_class_argument_tuple():
    x = aclick.utils._ClassArgument.from_str("(test)")
    assert isinstance(x, tuple)
    assert x == ("test",)

    x = aclick.utils._ClassArgument.from_str("()")
    assert isinstance(x, tuple)
    assert x == tuple()


def test_class_argument_dict():
    x = aclick.utils._ClassArgument.from_str("{test=ok}")
    assert isinstance(x, OrderedDict)
    assert x == dict(test="ok")

    x = aclick.utils._ClassArgument.from_str("{}")
    assert isinstance(x, dict)
    assert x == dict()


def test_class_argument_list():
    x = aclick.utils._ClassArgument.from_str("[test]")
    assert isinstance(x, list)
    assert x == [
        "test",
    ]

    x = aclick.utils._ClassArgument.from_str('[test="ok"]')
    assert isinstance(x, list)
    assert x == [
        'test="ok"',
    ]

    x = aclick.utils._ClassArgument.from_str("[]")
    assert isinstance(x, list)
    assert x == list()

    A = aclick.utils.default_from_str(makeclass(t.List[str]))
    assert A.from_str("a(p=[])").p == list()


def test_unsupported_type_raises_runtime_error():
    @aclick.utils.default_from_str
    class A:
        def __init__(self, a: str):
            self.a: t.Callable = lambda: "ok"

    with pytest.raises(RuntimeError) as excinfo:
        str(A("ok"))
    assert "is not supported" in str(excinfo.value).lower()


def test_to_str_use_dashes():
    @aclick.utils.default_from_str
    class CammelCase:
        def __init__(self, param_1: str):
            self.param_1 = param_1

    x = CammelCase("ok")
    assert x.__str_with_dashes_option__(True) == "cammel-case(param-1=ok)"
    assert x.__str_with_dashes_option__(False) == "cammel_case(param_1=ok)"
    assert str(x) == "cammel_case(param_1=ok)"

    @aclick.utils.default_from_str
    @dataclass
    class CammelCase2:
        param_1: str

    x = CammelCase2("ok")
    assert x.__str_with_dashes_option__(True) == "cammel-case2(param-1=ok)"
    assert x.__str_with_dashes_option__(False) == "cammel_case2(param_1=ok)"
    assert str(x) == "cammel_case2(param_1=ok)"


def makeclass(tp):
    class A:
        def __init__(self, p: tp):
            self.p = p

    return A


def test_from_str_cannot_parse_type_as_dict():
    with pytest.raises(RuntimeError) as excinfo:
        A = aclick.utils.default_from_str(makeclass(t.Dict[str, str]))
        A.from_str("a(p=[])")

    assert "cannot parse [] as a dict instance" in str(excinfo.value).lower()


def test_to_str_from_str_type():
    class A:
        @staticmethod
        def from_str(x):
            pass

        def __str_with_dashes_option__(self, use_dashes=False):
            return "ok" + ("-" if use_dashes else "_")

    B = aclick.utils.default_from_str(makeclass(A))
    assert B(A()).__str_with_dashes_option__(use_dashes=True) == "a(p=ok-)"
    assert B(A()).__str_with_dashes_option__(use_dashes=False) == "a(p=ok_)"
    assert str(B(A())) == "a(p=ok_)"

    class A:
        @staticmethod
        def from_str(x):
            pass

        def __str__(self):
            return "ok_"

    B = aclick.utils.default_from_str(makeclass(A))
    assert B(A()).__str_with_dashes_option__(use_dashes=True) == "a(p=ok_)"
    assert B(A()).__str_with_dashes_option__(use_dashes=False) == "a(p=ok_)"
    assert str(B(A())) == "a(p=ok_)"


def test_class_argument_bool():
    A = aclick.utils.default_from_str(makeclass(bool))
    x = A.from_str("a(p=t)")
    assert isinstance(x, A)
    assert x.p is True

    x = A.from_str("a(p=f)")
    assert isinstance(x, A)
    assert x.p is False

    with pytest.raises(RuntimeError) as excinfo:
        x = A.from_str("a(p=unknown)")
    assert "cannot parse unknown as bool" in str(excinfo.value).lower()

    with pytest.raises(RuntimeError) as excinfo:
        x = A.from_str("a(p=[])")
    assert "cannot parse [] as bool" in str(excinfo.value).lower()


def test_merge_signatures():
    def a(x: int, *, y: float):
        pass

    s = inspect.signature(a)
    assert aclick.utils._merge_signatures(s) == s


def test_parse_str_on_class_descendents():
    @aclick.utils.default_from_str()
    class A:
        def __init__(self, x: str):
            self.x = x

    class B(A):
        def __init__(self, y: str, **kwargs):
            super().__init__(**kwargs)
            self.y = y

    x = A.from_str("b(x=ok, y=pass)")
    assert isinstance(x, B)
    assert x.x == "ok"
    assert x.y == "pass"

    with pytest.raises(RuntimeError) as excinfo:
        x = B.from_str("b(x=ok, y=pass)")

    assert "method from_str should be called on" in str(excinfo.value).lower()


def test_build_examples():
    @dataclass
    class A:
        a: str

    @dataclass
    class B:
        b: int

    class C:
        pass

    @dataclass
    class D:
        x: t.Union[A, B, C]

    _s = aclick.utils._ClassArgument._escaped_str
    examples = aclick.utils._build_examples(A)
    assert examples == [aclick.utils._ClassArgument("a", [], dict(a=_s("{str}")))]

    examples = aclick.utils._build_examples(t.Union[A, B, C])
    assert examples == [
        aclick.utils._ClassArgument("a", [], dict(a=_s("{str}"))),
        aclick.utils._ClassArgument("b", [], dict(b=_s("{int}"))),
        aclick.utils._ClassArgument("c", [], dict()),
    ]

    examples = aclick.utils._build_examples(D)

    def mkcls(inside):
        return aclick.utils._ClassArgument("d", [], dict(x=inside))

    assert examples == [
        mkcls(aclick.utils._ClassArgument("a", [], dict(a=_s("{str}")))),
        mkcls(aclick.utils._ClassArgument("b", [], dict(b=_s("{int}")))),
        mkcls(aclick.utils._ClassArgument("c", [], dict())),
    ]

    examples = aclick.utils._build_examples(t.Union[D, C])

    def mkcls(inside):
        return aclick.utils._ClassArgument("d", [], dict(x=inside))

    assert examples == [
        mkcls(aclick.utils._ClassArgument("a", [], dict(a=_s("{str}")))),
        aclick.utils._ClassArgument("c", [], dict()),
        mkcls(aclick.utils._ClassArgument("b", [], dict(b=_s("{int}")))),
        mkcls(aclick.utils._ClassArgument("c", [], dict())),
    ]

    examples = aclick.utils._build_examples(t.Union[D, C], 2)

    def mkcls(inside):
        return aclick.utils._ClassArgument("d", [], dict(x=inside))

    assert examples == [
        mkcls(aclick.utils._ClassArgument("a", [], dict(a=_s("{str}")))),
        aclick.utils._ClassArgument("c", [], dict()),
    ]

    examples = aclick.utils._build_examples(t.List[A])
    assert examples == [
        (aclick.utils._ClassArgument("a", [], dict(a=_s("{str}"))),
         aclick.utils._ClassArgument._escaped_str('...')),
    ]

    examples = aclick.utils._build_examples(t.Tuple[A, B])
    assert examples == [
        (aclick.utils._ClassArgument("a", [], dict(a=_s("{str}"))),
         aclick.utils._ClassArgument("b", [], dict(b=_s("{int}"))))
    ]

    examples = aclick.utils._build_examples(t.OrderedDict[str, A])
    assert examples == [
        OrderedDict([
            ('{str}', aclick.utils._ClassArgument("a", [], dict(a=_s("{str}")))),
            (None, aclick.utils._ClassArgument._escaped_str('...')),
        ])
    ]

    examples = aclick.utils._build_examples(t.Dict[str, A])
    assert examples == [
        OrderedDict([
            ('{str}', aclick.utils._ClassArgument("a", [], dict(a=_s("{str}")))),
            (None, aclick.utils._ClassArgument._escaped_str('...')),
        ])
    ]


def test_as_dict_hierarchical_type():
    @dataclass
    class A:
        a: str

    @dataclass
    class B:
        a: A

    out = aclick.utils.as_dict(B(A('test')))
    assert out == OrderedDict([('a', OrderedDict([('a', 'test')]))])
