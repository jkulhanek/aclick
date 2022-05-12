import inspect
import typing as t
from dataclasses import dataclass

import aclick.utils
import pytest


#
# Utils tests
#
def test_is_class_on_class():
    class D:
        def __init__(self, a: int, /, b: str, *, c: float):
            self.a = a
            self.b = b
            self.c = c

    assert aclick.utils._is_class(D)


def test_from_str_class():
    @aclick.utils.default_from_str
    class D:
        def __init__(self, a: int, /, b: str, *, c: float):
            self.a = a
            self.b = b
            self.c = c

    x = D(42, "test", c=3.14)
    y = D.from_str(str(x))
    assert str(x) == "d(42, b=test, c=3.14)"
    assert isinstance(y, D)
    assert x.a == y.a
    assert x.b == y.b
    assert x.c == y.c


def test_from_str_class_not_setting_args():
    @aclick.utils.default_from_str()
    class A:
        def __init__(self, a: int, /):
            pass

    with pytest.raises(RuntimeError) as excinfo:
        x = A.from_str("a(1)")
        str(x)

    assert (
        "because the constructor parameter a is missing from class's properties"
        in str(excinfo.value).lower()
    )


def test_from_str_class_no_type_annotation_args():
    with pytest.raises(ValueError) as excinfo:

        @aclick.utils.default_from_str
        class A:
            def __init__(self, a, /):
                pass

    assert "does not have a type annotation" in str(excinfo.value).lower()


def test_from_str_class_unsupported_type_args():
    with pytest.raises(ValueError) as excinfo:

        @aclick.utils.default_from_str
        class A:
            def __init__(self, a: t.Callable, /):
                pass

    assert "is not supported because it contains type" in str(excinfo.value).lower()


def test_copy_signature():
    Return = type("Return", (), {})

    def z(d: int, /, e=3) -> None:
        pass

    @aclick.utils.copy_signature(z)
    def y(b: int, **kwargs) -> None:
        pass

    @aclick.utils.copy_signature(y)
    def x(a: str, *args, c=2, **kwargs) -> Return:
        pass

    signature = inspect.signature(x)
    assert signature.return_annotation is Return
    assert len(signature.parameters) == 4
    assert list(signature.parameters.keys()) == ["a", "b", "c", "e"]
    assert signature.parameters["a"].to_parameter() == inspect.Parameter(
        "a", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str
    )
    assert signature.parameters["b"].to_parameter() == inspect.Parameter(
        "b", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int
    )
    assert signature.parameters["e"].to_parameter() == inspect.Parameter(
        "e", inspect.Parameter.KEYWORD_ONLY, default=3
    )
    assert signature.parameters["c"].to_parameter() == inspect.Parameter(
        "c", inspect.Parameter.KEYWORD_ONLY, default=2
    )


def test_from_str_invalid_class_argument_errors():
    @aclick.utils.default_from_str
    class A:
        def __init__(self, a: str, /, *, b: str):
            self.a = a
            self.b = b

    with pytest.raises(RuntimeError) as excinfo:
        A.from_str("c()")

    assert (
        'could not find class with name "c" in the list of registered classes: a'
        in str(excinfo.value).lower()
    )

    with pytest.raises(RuntimeError) as excinfo:
        A.from_str("a(b=ok)")

    assert (
        "number of passed positional arguments (0) to class a is lower then the number of expected arguments"
        in str(excinfo.value).lower()
    )

    with pytest.raises(RuntimeError) as excinfo:
        A.from_str("a(13, 5)")

    assert (
        "of passed positional arguments (2) to class a exceeds the number of allowed arguments (1)"
        in str(excinfo.value).lower()
    )

    with pytest.raises(RuntimeError) as excinfo:
        A.from_str('a("", b=ok, d=3, c=fail)')

    assert (
        "there were unknown parameters {c, d} to class a. allowed parameters are: {b}"
        in str(excinfo.value).lower()
    )

    with pytest.raises(RuntimeError) as excinfo:
        A.from_str('a("")')

    assert "parameters {b} to class a are missing" in str(excinfo.value).lower()


def test_wrap_fn_to_allow_kwargs_instead_of_args():
    @aclick.utils._wrap_fn_to_allow_kwargs_instead_of_args
    def a(x: int, y: float = 2, /, z: str = "", *, u: str = "ok"):
        return dict(x=x, y=y, z=z, u=u)

    out = a(x=1, z="pass")
    assert out["x"] == 1
    assert out["y"] == 2
    assert out["z"] == "pass"
    assert out["u"] == "ok"


def test_build_examples_str():
    @dataclass
    class A:
        a: str

    class B:
        def __init__(self, b: int, /):
            pass

    class C:
        pass

    @dataclass
    class D:
        x: t.Union[A, B, C]
        y: float = 3.14

    examples = aclick.utils.build_examples(A)
    assert examples == ['''a(
  a={str})''']

    examples = aclick.utils.build_examples(t.Union[A, B, C])
    assert examples == ['''a(
  a={str})''',
      '''b(
  {int})''',
      'c()'
    ]

    examples = aclick.utils.build_examples(D)

    def mkcls(inside):
        inside = inside.lstrip().replace("\n", "\n  ")
        return f'''d(
  x={inside},
  y={{optional float}})'''

    assert examples == [
        mkcls('''a(
  a={str})'''),
        mkcls('''b(
  {int})'''),
        mkcls('c()'),
    ]

    examples = aclick.utils.build_examples(t.Union[D, C])

    assert examples == [
        mkcls('''a(
  a={str})'''),
        'c()',
        mkcls('''b(
  {int})'''),
        mkcls('c()'),
    ]

    examples = aclick.utils.build_examples(t.Union[D, C], limit=2)

    assert examples == [
        mkcls('''a(
  a={str})'''),
        'c()',
    ]
