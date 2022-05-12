import typing as t
from dataclasses import dataclass

import aclick.utils
import pytest


@dataclass
class A:
    a: str


class B:
    b: str


@pytest.mark.parametrize(
    "tp",
    [
        # str,
        # int,
        # float,
        # bool,
        # type(None)
        t.Union[A, B]
    ],
)
def test_type(tp):
    assert len(aclick.utils.build_examples(tp, use_dashes=True)) > 0
    assert len(aclick.utils.build_examples(tp, use_dashes=False)) > 0
