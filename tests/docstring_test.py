import typing as t
from dataclasses import dataclass
from functools import partial

import aclick

import aclick.utils
from aclick.utils import (
    ParameterWithDescription,
    signature_with_description,
    SignatureWithDescription,
)

from ._common import click_test_error


def test_parameter_with_description():
    p1 = ParameterWithDescription(
        name="test",
        kind=ParameterWithDescription.POSITIONAL_ONLY,
        description="test",
        annotation=str,
        default=None,
    )
    p2 = ParameterWithDescription(
        name="test",
        kind=ParameterWithDescription.POSITIONAL_ONLY,
        description="notest",
        annotation=str,
        default=None,
    )
    assert p1 != p2
    assert hash(p1) != hash(p2)

    p2 = ParameterWithDescription(
        name="test",
        kind=ParameterWithDescription.POSITIONAL_ONLY,
        description="test",
        annotation=str,
        default="ok",
    )
    assert p1 != p2
    assert hash(p1) != hash(p2)

    p2 = ParameterWithDescription(
        name="test",
        kind=ParameterWithDescription.POSITIONAL_ONLY,
        description="test",
        annotation=str,
        default=None,
    )
    assert p1 == p2
    assert hash(p1) == hash(p2)

    p3 = p2.replace(description="ok")
    assert p2.description == "test"
    assert p3.description == "ok"
    assert p2 == p3.replace(description="test")


def test_signature_with_description():
    parameters = [
        ParameterWithDescription(
            name="test",
            kind=ParameterWithDescription.POSITIONAL_ONLY,
            description="test",
            annotation=str,
            default=None,
        )
    ]
    s1 = SignatureWithDescription(parameters, short_description="test")
    s2 = SignatureWithDescription(parameters, short_description="notest")
    assert s1 != s2
    assert hash(s1) != hash(s2)

    s1 = SignatureWithDescription(
        parameters, short_description="test", long_description="ok"
    )
    s2 = SignatureWithDescription(
        parameters, short_description="test", long_description="fail"
    )
    assert s1 != s2
    assert hash(s1) != hash(s2)

    s1 = SignatureWithDescription(
        parameters, short_description="test", long_description="ok"
    )
    s2 = SignatureWithDescription(
        parameters, short_description="test", long_description="ok"
    )
    assert s1 == s2
    assert hash(s1) == hash(s2)

    parameters2 = [
        ParameterWithDescription(
            name="test",
            kind=ParameterWithDescription.POSITIONAL_ONLY,
            description="fail",
            annotation=str,
            default=None,
        )
    ]
    s1 = SignatureWithDescription(
        parameters, short_description="test", long_description="ok"
    )
    s2 = SignatureWithDescription(
        parameters2, short_description="test", long_description="ok"
    )
    assert s1 != s2
    assert hash(s1) != hash(s2)

    s3 = s2.replace(short_description="ok", long_description="pass")
    assert s2.short_description == "test"
    assert s2.long_description == "ok"
    assert s3.short_description == "ok"
    assert s3.long_description == "pass"
    assert s2 == s3.replace(short_description="test", long_description="ok")


def test_parse_sections_google():
    # Google format parsing
    def a(task, deployment):
        """
        Test function description.

        Description 2

        Args:
            task: Path to the input task file.
            deployment: UUID or name of the deployment

        Returns:
            NIL
        Example:
            this is an ex
            ample

        """

    x = signature_with_description(a)
    assert x.short_description == "Test function description."
    assert x.long_description == "Description 2"
    assert x.parameters["task"].description == "Path to the input task file."
    assert x.parameters["deployment"].description == "UUID or name of the deployment"

    @dataclass
    class A:
        """
        Test function description.

        Description 2

        Args:
            task: Path to the input task file.
            deployment: UUID or name of the deployment
        """

        task: str
        deployment: str

    x = signature_with_description(A)
    assert x.short_description == "Test function description."
    assert x.long_description == "Description 2"
    assert x.parameters["task"].description == "Path to the input task file."
    assert x.parameters["deployment"].description == "UUID or name of the deployment"


def test_parse_sections_rest():
    # ReST format parsing
    def a(task, deployment):
        """
        Test function description.

        Description 2

        :param task: Path to the input task file.
        :param deployment: UUID or name of the deployment

        :returns: NIL
        """

    x = signature_with_description(a)
    assert x.short_description == "Test function description."
    assert x.long_description == "Description 2"
    assert x.parameters["task"].description == "Path to the input task file."
    assert x.parameters["deployment"].description == "UUID or name of the deployment"

    @dataclass
    class A:
        """
        Test function description.

        Description 2

        :param task: Path to the input task file.
        :param deployment: UUID or name of the deployment
        """

        task: str
        deployment: str

    x = signature_with_description(A)
    assert x.short_description == "Test function description."
    assert x.long_description == "Description 2"
    assert x.parameters["task"].description == "Path to the input task file."
    assert x.parameters["deployment"].description == "UUID or name of the deployment"


def test_parse_sections_epydoc():
    # Epydoc format parsing
    def a(task, deployment):
        """
        Test function description.

        Description 2

        @param task: Path to the input task file.
        @param deployment: UUID or name of the deployment

        @return: NIL
        """
        pass

    x = signature_with_description(a)
    assert x.short_description == "Test function description."
    assert x.long_description == "Description 2"
    assert x.parameters["task"].description == "Path to the input task file."
    assert x.parameters["deployment"].description == "UUID or name of the deployment"

    @dataclass
    class A:
        """
        Test function description.

        Description 2

        @param task: Path to the input task file.
        @param deployment: UUID or name of the deployment
        """

        task: str
        deployment: str

    x = signature_with_description(A)
    assert x.short_description == "Test function description."
    assert x.long_description == "Description 2"
    assert x.parameters["task"].description == "Path to the input task file."
    assert x.parameters["deployment"].description == "UUID or name of the deployment"


def test_full_signature_copies_parameter_descriptions():
    def b(deployment):
        """
        @param deployment: UUID or name of the deployment
        """
        pass

    @aclick.utils.copy_signature(b)
    def a(task, **kwargs):
        """
        Test function description.

        Description 2

        @param task: Path to the input task file.
        """

    x = aclick.utils._full_signature(a)
    assert x.short_description == "Test function description."
    assert x.long_description == "Description 2"
    assert x.parameters["task"].description == "Path to the input task file."
    assert x.parameters["deployment"].description == "UUID or name of the deployment"


def test_signature_description_in_click_help(monkeypatch):
    def a(error, a: str, b: int):
        """
        Test function description.

        Description 2

        @param a: param a.
        @param b: param b.
        """

        status, msg = error
        assert status == 0
        if a != "second":
            assert "param a." in msg
            assert "param b." in msg
        assert "Test function description." in msg
        assert "@param" not in msg

    cmd = click_test_error("--help")(a)
    cmd(monkeypatch)

    a_part = partial(a, None)
    setattr(a_part, "__doc__", a.__doc__)
    cmd = aclick.Command(name="a", callback=a_part)
    a((0, cmd.help), t.cast(str, "second"), t.cast(int, None))
