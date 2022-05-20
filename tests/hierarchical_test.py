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


@click_test("--a-test", "passed")
def test_hierarchical_is_default(a: D1):
    assert isinstance(a, D1)
    assert a.test == "passed"


def test_dataclass_no_default_error():
    with pytest.raises(ValueError) as exinfo:

        @aclick.command(hierarchical=True)
        def test_dataclass_no_default(a: D1 = D1("ok")):
            pass

    assert "Cannot use a parameter with a default in hierarchical parsing" in str(
        exinfo.value
    )


@dataclass
class D2:
    test: str = "ok"
    test2: int = 4


def test_dataclass_union_validation():
    with pytest.raises(ValueError) as exinfo:

        @aclick.command(hierarchical=True)
        def test_dataclass_no_default(a: t.Union[D1, str]):
            pass

    assert "is not supported" in str(exinfo.value)

    with pytest.raises(ValueError) as exinfo:

        @aclick.command(hierarchical=True)
        def test_dataclass_no_default2(a: t.Union[D1, D2, None]):
            pass

    assert "is not supported" in str(exinfo.value)


def test_dataclass_default(monkeypatch):
    @aclick.command(hierarchical=True)
    def test_dataclass_default(a: D2):
        assert isinstance(a, D2)
        assert a.test == "ok"

    ctx = click.Context(test_dataclass_default)
    test_dataclass_default.parse_args(ctx, [])
    params = test_dataclass_default.get_params(ctx)
    assert params[1].default == "ok"
    assert params[2].default == 4

    @click_test(hierarchical=True)
    def test_dataclass_default(a: D2):
        assert isinstance(a, D2)
        assert a.test == "ok"

    test_dataclass_default(monkeypatch)


def test_dataclass_optional_validation():
    with pytest.raises(ValueError) as excinfo:

        @aclick.command(hierarchical=True)
        def test_dataclass_optional(a: t.Optional[D2] = D2()):
            assert a is None

    assert "is not supported" in str(excinfo.value)


def test_dataclass_optional(monkeypatch):
    @click_test(hierarchical=True)
    def test_dataclass_optional(a: t.Optional[D2] = None):
        assert a is None

    test_dataclass_optional(monkeypatch)

    class Schedule:
        def __init__(self, type: str = "constant", constant: float = 1e-4):
            self.type = type
            self.constant = constant

    @dataclass
    class Model:
        """
        :param learning_rate: Learning rate
        :param num_features: Number of features
        """

        learning_rate: t.Optional[Schedule] = None
        num_features: int = 5

    @click_test("--model-learning-rate", hierarchical=True)
    def train(model: Model, num_epochs: int = 5):
        assert model is not None
        assert isinstance(model.learning_rate, Schedule)

    train(monkeypatch)

    @click_test(hierarchical=True)
    def train(model: Model, num_epochs: int = 5):
        assert model is not None
        assert model.learning_rate is None

    train(monkeypatch)

    @dataclass
    class Model:
        """
        :param learning_rate: Learning rate
        :param num_features: Number of features
        """

        learning_rate: t.Optional[Schedule]
        num_features: int = 5

    @click_test_error(hierarchical=True)
    def train(err, model: Model, num_epochs: int = 5):
        status, msg = err
        assert status != 0
        assert "missing option" in msg.lower()

    train(monkeypatch)

    @click_test("--no-model-learning-rate", hierarchical=True)
    def train(model: Model, num_epochs: int = 5):
        assert model is not None
        assert model.learning_rate is None

    train(monkeypatch)

    @click_test("--model-learning-rate", hierarchical=True)
    def train(model: Model, num_epochs: int = 5):
        assert model is not None
        assert model.learning_rate is not None

    train(monkeypatch)

    @click_test_error("--model-learning-rate", "--help", hierarchical=True)
    def train(err, model: Model, num_epochs: int = 5):
        status, msg = err
        assert status == 0
        print(msg)
        assert "--model-learning-rate-constant" in msg

    train(monkeypatch)


def test_union_of_dataclasses(monkeypatch):
    @click_test("--a", "d2", hierarchical=True)
    def test_union_of_dataclasses(a: t.Union[D1, D2]):
        assert a is not None
        assert isinstance(a, D2)
        assert a.test == "ok"

    test_union_of_dataclasses(monkeypatch)

    @dataclass
    class ModelA:
        """
        :param learning_rate: Learning rate
        :param num_features: Number of features
        """

        learning_rate: float = 0.1
        num_features: int = 5

    @dataclass
    class ModelB:
        """
        :param learning_rate: Learning rate
        :param num_layers: Number of layers
        """

        learning_rate: float = 0.2
        num_layers: int = 10

    @click_test_error("--help", hierarchical=True)
    def train(err, model: t.Union[ModelA, ModelB], num_epochs: int):
        status, msg = err
        assert status == 0
        assert "model-a" in msg
        assert "model-b" in msg
        assert "num-layers" not in msg

    train(monkeypatch)

    @click_test_error("--model", "model-b", "--help", hierarchical=True)
    def train(err, model: t.Union[ModelA, ModelB], num_epochs: int):
        status, msg = err
        assert status == 0
        print(msg)
        assert "num-layers" in msg

    train(monkeypatch)


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


@dataclass
class D6:
    test: str

    @classmethod
    def from_str(cls, val):
        return D6(val)

    def __str__(self):
        return self.test


@click_test("--a", 'd6("passed")', hierarchical=True)
def test_hierarchical_parsing_disabled_for_from_str_classes(a: D6):
    assert isinstance(a, D6)
    assert a.test == "passed"
