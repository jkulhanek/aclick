import os

from setuptools import find_packages, setup

__version__ = None
basepath = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(basepath, "aclick", "_version.py")).read())
assert __version__ is not None

setup(
    name="aclick",
    version=__version__,
    packages=find_packages(include=("aclick", "aclick.*")),
    author="Jonáš Kulhánek",
    author_email="jonas.kulhanek@live.com",
    license="MIT License",
    long_description=open(os.path.join(basepath, "README.md")).read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    install_requires=[
        "click>=8.0.3",
        "docstring-parser>=0.14.1",
        "typing-extensions>=3.6.4; python_version < '3.8'",
    ],
)
