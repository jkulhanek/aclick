import typing as t
from dataclasses import dataclass

import pytest

from aclick.configuration import parse_configuration, ParseConfigurationContext
from aclick.utils import from_dict

try:
    import gin
    import gin.config
except ImportError:
    pytest.skip(
        "Gin tests were skipped because the gin-config module is not installed",
        allow_module_level=True,
    )


def test_gin_config(tmp_path):
    import gin

    @gin.configurable
    def fn(a: int, b: str):
        pass

    with open(tmp_path / "conf.gin", "w+") as f:
        f.write(
            """
fn.a = 1
fn.b = "ok"
        """
        )

    with open(tmp_path / "conf.gin") as fp:
        cfg = parse_configuration(fp, context=ParseConfigurationContext(fn))

    assert cfg == dict(a=1, b="ok")


def test_gin_config_class(tmp_path):
    import gin

    allow_call = False

    @gin.configurable
    class A:
        def __init__(self, c: str = "fail"):
            self.c = c
            assert allow_call, "Constructor should not be called yet"

    @gin.configurable
    def fn2(a: int, b: str, cls: A):
        return (a, b, cls)

    with open(tmp_path / "conf.gin", "w+") as f:
        f.write(
            """
A.c = "pass"
fn2.cls = @A()
fn2.a = 1
fn2.b = "ok"
        """
        )

    with open(tmp_path / "conf.gin") as fp:
        cfg = parse_configuration(fp, context=ParseConfigurationContext(fn2))

    assert cfg == dict(a=1, b="ok", cls=dict(c="pass"))

    allow_call = True
    out = from_dict(fn2, cfg)
    assert out[:2] == (1, "ok")
    assert isinstance(out[-1], A)
    assert out[-1].c == "pass"
    gin.clear_config()

    # class has to be an instance
    with open(tmp_path / "conf.gin", "w+") as f:
        f.write(
            """
A.c = "pass"
fn2.cls = @A
fn2.a = 1
fn2.b = "ok"
        """
        )

    with pytest.raises(RuntimeError) as errinfo:
        with open(tmp_path / "conf.gin") as fp:
            cfg = parse_configuration(fp, context=ParseConfigurationContext(fn2))
    gin.clear_config()
    assert "must be an instance" in str(errinfo.value)

    @gin.configurable
    def fn2b(cls: A, clsb: A):
        return (cls, clsb)

    with open(tmp_path / "conf.gin", "w+") as f:
        f.write(
            """
fn2b.cls = @type1/A()
fn2b.clsb = @type2/A()
type1/A.c = "pass"
type2/A.c = "passB"
        """
        )

    with open(tmp_path / "conf.gin") as fp:
        cfg = parse_configuration(fp, context=ParseConfigurationContext(fn2b))

    assert cfg == dict(cls=dict(c="pass"), clsb=dict(c="passB"))

    allow_call = True
    out = from_dict(fn2b, cfg)
    assert isinstance(out[0], A)
    assert isinstance(out[1], A)
    assert out[0].c == "pass"
    assert out[1].c == "passB"
    gin.clear_config()


def test_gin_config_union_class(tmp_path):
    import gin

    allow_call = False

    @gin.configurable
    class B:
        def __init__(self, c: str = "fail"):
            self.c = c
            assert allow_call, "Constructor should not be called yet"

    @dataclass
    class C:
        pass

    @gin.configurable
    def fn3(a: int, b: str, cls: t.Union[B, C]):
        return (a, b, cls)

    with open(tmp_path / "conf.gin", "w+") as f:
        f.write(
            """
B.c = "pass"
fn3.cls = @B()
fn3.a = 1
fn3.b = "ok"
        """
        )

    with open(tmp_path / "conf.gin") as fp:
        cfg = parse_configuration(fp, context=ParseConfigurationContext(fn3))

    assert cfg == dict(a=1, b="ok", cls=dict(__class__="b", c="pass"))

    allow_call = True
    out = from_dict(fn3, cfg)
    assert out[:2] == (1, "ok")
    assert isinstance(out[-1], B)
    assert out[-1].c == "pass"

    @gin.configurable
    def fn4(cls: t.Optional[B]):
        return (cls,)

    with open(tmp_path / "conf.gin", "w+") as f:
        f.write(
            """
B.c = "pass"
fn4.cls = @B()
        """
        )

    with open(tmp_path / "conf.gin") as fp:
        cfg = parse_configuration(fp, context=ParseConfigurationContext(fn4))

    assert cfg == dict(cls=dict(__class__="b", c="pass"))

    allow_call = True
    out = from_dict(fn4, cfg)
    assert isinstance(out[-1], B)
    assert out[-1].c == "pass"
    gin.clear_config()

    with open(tmp_path / "conf.gin", "w+") as f:
        f.write(
            """
B.c = "pass"
fn4.cls = None
        """
        )

    with open(tmp_path / "conf.gin") as fp:
        cfg = parse_configuration(fp, context=ParseConfigurationContext(fn4))

    assert cfg == dict(cls=None)

    allow_call = True
    out = from_dict(fn4, cfg)
    assert out[-1] is None
    gin.clear_config()


def test_gin_external_configurable(tmp_path):
    @dataclass
    class E:
        val1: str

    @gin.configurable
    def fn5(e: E):
        return e

    gin.config.external_configurable(E, module="tst")
    with open(tmp_path / "conf.gin", "w+") as f:
        f.write(
            """
E.val1 = "pass"
fn5.e = @tst.E()
        """
        )

    with open(tmp_path / "conf.gin") as fp:
        cfg = parse_configuration(fp, context=ParseConfigurationContext(fn5))

    assert cfg == dict(e=dict(val1="pass"))

    out = from_dict(fn5, cfg)
    assert isinstance(out, E)
    assert out.val1 == "pass"
