import os
import re
import typing as t
from collections import OrderedDict
from dataclasses import dataclass

from .utils import _full_signature, get_class_name


_CONFIGURATION_PROVIDERS: t.OrderedDict[str, t.Callable] = OrderedDict()
TCallable = t.TypeVar("TCallable", bound=t.Callable)


@dataclass
class ParseConfigurationContext:
    """
    :param fn: Callback that will use the parsed parameters.
    """

    fn: t.Callable


def register_configuration_provider(regex: str) -> t.Callable[[TCallable], TCallable]:
    """
    A decorator that registers a new configuration provider.

    :param regex: Regex expression that matches the supported filenames.
    """

    def wrap(fn: TCallable) -> TCallable:
        _CONFIGURATION_PROVIDERS[regex] = fn
        return fn

    return wrap


def parse_configuration(fp, *, context: ParseConfigurationContext):
    '''
    Parses the configuration file using one of the supported configuration providers.

    :param fp: opened configuration file stream
    :param context: context providing additional information used for parsing
    """
    '''
    name = os.path.split(fp.name)[-1]
    regex_match = re.match(
        "(?:"
        + "|".join(
            f"(?P<pp{i}>{x})" for i, x in enumerate(_CONFIGURATION_PROVIDERS.keys())
        )
        + ")",
        name,
    )
    if not regex_match:
        raise RuntimeError(
            f"Cannot find a configuration provider to parse file {fp.name}"
        )

    found_pattern_dict = regex_match.groupdict()
    found_provider = next(
        provider
        for i, provider in enumerate(_CONFIGURATION_PROVIDERS.values())
        if found_pattern_dict[f"pp{i}"]
    )
    return found_provider(fp, context=context)


@register_configuration_provider(r".*\.json")
def parse_json_configuration(fp, *, context: ParseConfigurationContext):
    """
    Loads a configuration stored in a json file

    :param fp: opened json file stream
    :param context: context providing additional information used for parsing
    """
    import json

    return json.load(fp)


@register_configuration_provider(r".*\.gin")
def parse_gin_configuration(fp, *, context: ParseConfigurationContext):
    """
    Loads a configuration stored as a gin config file

    :param fp: opened gin file stream
    :param context: context providing additional information used for parsing
    """
    import gin.config

    def fix_config(config, tp):
        if isinstance(config, gin.config.ConfigurableReference):
            props = gin.config.get_bindings(
                config.scoped_selector, resolve_references=False
            )
            assert tp is not None
            if getattr(tp, "__origin__", None) is t.Union:
                props["__class__"] = get_class_name(config.configurable.wrapped)
            if not config.evaluate:
                raise RuntimeError(
                    f"For argument of type {tp} the value must be an instance, not a factory. Replace {config.selector} with {config.selector}()"
                )
            return props
        elif isinstance(config, dict):
            signature = _full_signature(tp)
            return OrderedDict(
                (k, fix_config(v, signature.parameters[k].annotation))
                for k, v in config.items()
            )
        return config

    skip_unknown = False
    includes, imports = gin.config.parse_config(fp, skip_unknown=skip_unknown)
    results = gin.config.ParsedConfigFileIncludesAndImports(
        filename=fp.name, imports=imports, includes=includes
    )
    gin.config.log_includes_and_imports(results)
    gin_callback: t.Optional[t.Callable] = context.fn
    while gin_callback is not None:
        if gin_callback in gin.config._INVERSE_REGISTRY:
            break
        gin_callback = getattr(gin_callback, "__wrapped__", None)
    else:
        raise RuntimeError(
            "Class of function {tp.__name__} was not registered with gin"
        )
    config = gin.config.get_bindings(gin_callback, resolve_references=False)
    return fix_config(config, context.fn)
