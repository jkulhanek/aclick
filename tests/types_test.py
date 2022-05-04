import typing as t

import aclick


def test_class_union_repr():
    class A:
        pass

    class B:
        pass

    x = aclick.types.convert_type(t.Union[A, B])
    assert "a|b" in repr(x).lower()


def test_list_repr():
    class A:
        pass

    x = aclick.types.convert_type(t.List[A])
    assert "list[a]" in repr(x).lower()
