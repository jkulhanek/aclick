Getting started
===============

.. currentmodule:: aclick

Install AClick from PyPI by running::

    pip install aclick
    


From Click to AClick
--------------------

AClick is a wrapper library around click and provides the :func:`command` and
:func:`group` decorators as well as :class:`Command` and :class:`Group` classes.
The decorators will automatically register `aclick` types and if you supply your
own :class:`Command` or :class:`Group` implementation, they will wrap it as an `aclick`
type.

Similarly to Click, decorating a function with :func:`click.command`
will make it into a callable script. However, with `aclick` we do not 
need to specify parameter types, defaults and help, but everything
is parsed automatically from type annotations and docstring:

.. container:: pair-code

    .. click:example::

       # Click example 
       import click

       @click.command()
       @click.argument('name')
       @click.option('--age', 
           default=18, type=int,
           help='Your age')
       def hello(name, age):
           '''
           Prints greeting
           '''


           click.echo(f'Hello {name}!')

    .. click:example::

       # AClick example 
       import aclick, click





       @aclick.command()
       def hello(name: str, /, age: int = 18):
           '''
           Prints greeting

           :param age: Your age
           '''
           click.echo(f'Hello {name} with age {age}!')


In both cases the resulting `Command` can be invoked::

    if __name__ == '__main__':
        hello()


And what it looks like:


.. click:run::

    invoke(hello, args=['Jonas', '--age', '25'], prog_name='python hello.py')

And the corresponding help page:

.. click:run::

    invoke(hello, args=['--help'], prog_name='python hello.py')


Supported types
---------------

By default, ``aclick`` supports the following types:

* str, bool, int, float

* Optional, List, Tuple

* Custom classes and dataclasses using :doc:`inline parsing <inline-types>` or :doc:`hierarchical parsing <hierarchical-parsing>`

In :doc:`hierarchical parsing <hierarchical-parsing>`, there are some restrictions on supported types. Please
refer to :doc:`hierarchical parsing section <hierarchical-parsing>`.

Arguments and options
---------------------

All positional arguments are expanded as Click's arguments and keyword arguments
are expanded as options. In the following example, ``name`` will become an argument
and age will became an option.

.. click:example::

   @aclick.command()
   def hello(name: str, /, age: int = 18):
       click.echo(f'Hello {name} with age {age}!')

.. click:run::

    invoke(hello, args=['--help'], prog_name='python hello.py')

Command groups
--------------

Similarly to the `Command` example, by importing the group decorator
from `aclick` all commands will be of class :class:`Command` and they
will automatically parse function's parameters.

.. container:: pair-code

    .. click:example::

        # Click example
        @click.group()
        def cli():
            pass

        @cli.command()
        @click.option('--user', type=str)
        def initdb(user: str):
            click.echo(f'{user} initialized the database')

        @cli.command()
        @click.option('--database', type=str)
        def dropdb(database):
            click.echo(f'Dropped the database: {database}')


    .. click:example::
    
        # AClick example
        @aclick.group()
        def cli():
            pass

            
        @cli.command()
        def initdb(user: str):
            click.echo(f'{user} initialized the database')


        @cli.command()
        def dropdb(database: str):
            click.echo(f'Dropped the database: {database}')

The `cli` group is then used as the entrypoint::

    if __name__ == '__main__':
        cli()


Lists and Dictionaries
----------------------

Lists and dictionaries can be uses as parameters in aclick.
For the dictionaries, the key has to be a string. The values
contained in the list or dictionary can be any basic type like
`str`, `float`, `bool`, `int`, but it can also be a complex structure
of classes, see :doc:`inline types <inline-types>`. We will demonstrate
the usage in the following example.

.. click:example::

    @dataclass
    class Person:
        name: str
    

    @aclick.command
    def main(list: t.List[Person], age: t.Dict[str, int]):
        for p in list:
            p_age = age[p.name]
            print(f'    {p.name} is {p_age} years old')


Which we can call as follows:

.. click:run::

    invoke(main, args=['--list', 'person(Alice),person(Bob)', '--age', 'Alice=21,Bob=53'], prog_name='python main.py')
