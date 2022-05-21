import copy
import inspect
import random
import typing as t
from dataclasses import dataclass
from collections import OrderedDict
from aclick.types import Tuple
import itertools

from ._common import _call_fn_empty
import aclick.utils
import pytest


def _construct_type_examples(tp):
    if tp in (type(None),):
        return [None]
    if tp == str:
        return ['str' + str(random.randint(0, 1000))]
    elif tp == int:
        return [random.randint(0, 1000)]
    elif tp is bool:
        return [True, False]
    elif tp is float:
        return [random.random()]
    elif getattr(tp, '__origin__', None) is aclick.utils.Literal:
        return list(tp.__args__)
    elif getattr(tp, '__origin__', None) is list:
        return list(map(list, zip(*[_construct_type_examples(tp.__args__[0]) for _ in range(random.randint(2, 5))])))
    elif getattr(tp, '__origin__', None) is tuple:
        return list(itertools.product(*tuple(_construct_type_examples(local_tp) for local_tp in tp.__args__)))
    elif getattr(tp, '__origin__', None) in (dict, OrderedDict):
        num_vals = random.randint(15, 20)
        keys = [x for _ in range(num_vals) for x in _construct_type_examples(tp.__args__[0])][:num_vals]
        values = [x for _ in range(num_vals) for x in _construct_type_examples(tp.__args__[1])][:num_vals]
        return [tp.__origin__(zip(keys, values))]
    elif getattr(tp, '__origin__', None) is t.Union:
        return list(x for y in map(_construct_type_examples, tp.__args__) for x in y)
    elif aclick.utils._is_class(tp):
        signature = aclick.utils._full_signature(tp)
        values = list(itertools.product(*tuple(_construct_type_examples(x.annotation) for x in signature.parameters.values())))
        out = []
        for local_values in values:
            args, kwargs = [], dict()
            for p, val in zip(signature.parameters.values(), local_values):
                if p.kind == inspect.Parameter.POSITIONAL_ONLY:
                    args.append(val)
                else:
                    kwargs[p.name] = val
            out.append(tp(*args, **kwargs))
        return out
    raise RuntimeError(f'Type {tp} not supported')


@dataclass
class A:
    a: str


class B:
    def __init__(self, b: str):
        self.b = b

    def __eq__(self, other):
        return isinstance(other, B) and self.b == other.b


@pytest.mark.parametrize(
    "tp",
    [
        str,
        int,
        float,
        bool,
        type(None),
        aclick.utils.Literal['ok', 'fail'],
        A,
        B,
        t.Union[A, B],
        t.List[A],
        t.Tuple[A],
        t.Dict[str, A],
        t.OrderedDict[str, A],
        t.Optional[A],
        t.Optional[B],
    ],
)
def test_type(tp):
    assert len(aclick.utils.build_examples(tp, use_dashes=True)) > 0
    assert len(aclick.utils.build_examples(tp, use_dashes=False)) > 0

    default_instance = next(iter(_construct_type_examples(tp)))

    for example_instance in _construct_type_examples(tp):
        example_dict = aclick.utils.as_dict(example_instance, tp)
        was_called = False

        @aclick.utils._fill_signature_defaults_from_dict(dict(a=copy.deepcopy(example_dict)))
        def fn(a: tp = default_instance):
            nonlocal was_called
            assert a == example_instance
            was_called = True

        _call_fn_empty(fn)
        example_instance2 = aclick.utils.from_dict(tp, example_dict)
        assert example_instance2 == example_instance
        assert was_called
