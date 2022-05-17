import typing as t
from dataclasses import dataclass
from aclick.types import Tuple

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
        str,
        int,
        float,
        bool,
        type(None),
        t.Union[A, B],
        t.List[A],
        t.Tuple[A],
        t.Dict[str, A],
        t.OrderedDict[str, A],
    ],
)
def test_type(tp):
    assert len(aclick.utils.build_examples(tp, use_dashes=True)) > 0
    assert len(aclick.utils.build_examples(tp, use_dashes=False)) > 0
