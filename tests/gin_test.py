import typing as t
from dataclasses import dataclass

from aclick.configuration import parse_configuration, ParseConfigurationContext
from aclick.utils import from_dict


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
fn2.cls = @A
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
fn3.cls = @B
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
fn4.cls = @B
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
