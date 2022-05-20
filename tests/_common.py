import inspect
import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from functools import partial

import aclick as click
from aclick.utils import _full_signature, _get_signature_args_kwargs, _is_class

SUPPORTS_POSITIONAL_ONLY_ARGUMENTS = sys.version_info >= (3, 8, 0)


def click_test(*args, **kwargs):
    def wrap(f):
        def fn(monkeypatch):
            monkeypatch.setattr(sys, "argv", ["prg.py"] + list(args))
            monkeypatch.setattr(sys, "exit", lambda *args, **kwargs: None)
            was_called = False

            def f2(*args, **kwargs):
                nonlocal was_called
                was_called = True

                f(*args, **kwargs)

            setattr(f2, "__signature__", inspect.signature(f))
            if hasattr(f, "__click_params__"):
                setattr(f2, "__click_params__", f.__click_params__)
            setattr(f2, "__doc__", f.__doc__)
            f_wrapped = click.command(**kwargs)(f2)
            f_wrapped()
            assert was_called, f"Function {f.__name__} was not called"

        return fn

    return wrap


def _call_fn_empty(fn, *args, **kwargs):
    def construct(tp):
        if _is_class(tp):
            return _call_fn_empty(tp)
        return None

    fn = partial(fn, *args, **kwargs)
    signature = _full_signature(fn)
    args_par, kwargs_par = _get_signature_args_kwargs(signature, exclusive=True)
    new_args = [
        construct(x.annotation) for x in args_par if x.default is inspect._empty
    ]
    new_kwargs = {
        x.name: construct(x.annotation)
        for x in kwargs_par
        if x.default is inspect._empty
    }
    return fn(*new_args, **new_kwargs)


def click_test_error(*args, **kwargs):
    def wrap(f):
        def fn(monkeypatch):
            status = 0

            def lexit(s=0):
                nonlocal status
                status = s

            monkeypatch.setattr(sys, "argv", ["prg.py"] + list(args))
            monkeypatch.setattr(sys, "exit", lexit)
            was_called = False

            def f2(*args, **kwargs):
                nonlocal was_called
                was_called = True

            error_text = "failed"
            try:
                signature = inspect.signature(partial(f, None))
                setattr(f2, "__signature__", signature)
                if hasattr(f, "__click_params__"):
                    setattr(f2, "__click_params__", f.__click_params__)
                setattr(f2, "__doc__", f.__doc__)
                with io.StringIO() as fcapt, redirect_stderr(fcapt), redirect_stdout(
                    fcapt
                ):
                    f_wrapped = click.command(**kwargs)(f2)
                    f_wrapped()
                    error_text = fcapt.getvalue()
            except Exception as e:
                _call_fn_empty(f, e)
            else:
                _call_fn_empty(f, (status, error_text))
            assert not was_called, f"Function {f.__name__} should not be called"

        return fn

    return wrap
