AClick
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


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


Documentation
-------------
.. toctree::
   :maxdepth: 2
    
   Introduction <self>
   basic
   hierarchical-parsing
   inline-types
   parameter-renaming
   loading-configuration


API Reference
-------------

The API reference lists the full API reference of all public
classes and functions.

.. toctree::
   :maxdepth: 2

   api
