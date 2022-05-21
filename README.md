# aclick

[![pypi](https://img.shields.io/pypi/v/aclick.svg)](https://pypi.org/project/aclick/)
[![tests](https://img.shields.io/github/workflow/status/jkulhanek/aclick/run-tests?label=tests)](https://github.com/jkulhanek/aclick/actions/workflows/run-tests.yml)
[![coverage](https://img.shields.io/codecov/c/gh/jkulhanek/aclick)](https://app.codecov.io/gh/jkulhanek/aclick)
![python](https://img.shields.io/badge/python-3.7%2C3.8%2C3.9%2C3.10-blue)

**aclick** is a python library extending `click` with the support for typing. It uses function signatures to
automatically register options to parsers. Please refer to the [documentation](https://jkulhanek.github.io/aclick).

The following features are currently supported:

- Positional-only parameters are added as click Arguments, other parameters become click Options.
- Docstring is automatically parsed and used to generate command and parameter descriptions.
- Arguments with `int`, `float`, `str`, `bool` values both with and without default value.
- Complex structures of classes and dataclasses that are automatically inlined as a single string, e.g.,
  `class1("arg1", arg2=class2())`.
- Complex structures of classes and dataclasses that are expanded as individual options with the `hierarchical=True`
  option enabled.
- Type `Union` of complex classes both inlined and hierarchical.
- Type `Optional` of inlined complex classes.
- Type `Literal` of strings.
- Lists and tuples of both the primitive and inlined complex types.
- Parameters can be renamed.
- Parameter values can be loaded from a json or other file.
- For other features please refer to the [documentation](https://jkulhanek.github.io/aclick).

## Installation

Install the library from pip:

```
$ pip install aclick
```

## Getting started

Import `aclick` instead of `click`:

```python
# python main.py test --arg2 4

import aclick

@aclick.command()
def example(arg1: str, /, arg2: int = 5):
    pass

example()
```

When using `click.groups`:

```python
# python main.py example test --arg2 4

import aclick

@aclick.group()
def main():
    pass

@main.command('example')
def example(arg1: str, /, arg2: int = 5):
    pass

main()
```

For further details please look at the [documentation](https://jkulhanek.github.io/aclick).

## License

[MIT](/LICENSE)
