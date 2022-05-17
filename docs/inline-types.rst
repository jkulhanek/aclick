Inline types
============

.. currentmodule:: aclick


Inline types allows complex class structures to be parsed as string.
First, we will motivate the reader by a simple example:

.. click:example::

    import aclick, click, typing as t

    class ModelA:
        def __init__(self, name: str, n_layers: int):
            self.name = name
            self.n_layers = n_layers

        def train(self):
            click.echo(f'Training model A with {self.n_layers} layers.')

    class ModelB(ModelA):
        def __init__(self, *args, n_blocks: int = 5, **kwargs):
            super().__init__(*args, **kwargs)
            self.n_blocks = n_blocks

        def train(self):
            click.echo(f'Training model B with {self.n_blocks} blocks.')


    @aclick.command(hierarchical=False)
    def main(model: t.Union[ModelA, ModelB]):
        model.train()

    if __name__ == '__main__':
        main()

We can invoke the script as follows:

.. click:run::

    invoke(main, args=['--model', 'model-b(test, n-layers=2)'], prog_name='python hello.py')


Note, that by default complex classes will be parsed using :doc:`hierarchical parsing <hierarchical-parsing>`.
If you want all complex class types to be parsed using inline parsing,
simply turn off hierarchical parsing by passing ``hierarchical = False`` to the 
class constructor. If you want to keep using hierarchical parsing, but
use inline parsing for one type only, use the :func:`aclick.default_from_str`
decorator (or implement :func:`from_str` and :func:`__str__` methods).


Custom classes
--------------

In inline parsing, user-defined classes can be naturally parsed with the following structure:

    ``class-name(arg1, ..., argM, named-arg-1=value1, ..., named-arg-n=valueN)``

Where the class ``ClassName`` accepts ``M`` positional arguments and all named
arguments as well. Each argument value can itself be a complex structure of classes:

.. click:example::
    class Schedule:
        def __init__(self, type: str, constants: t.Optional[t.List[float]] = None):
            self.type = type
            self.constants = constants
        
        def __repr__(self):
            return f'Schedule("{self.type}", constants={self.constants})'
   
    @dataclass
    class Model:
        '''
        :param learning_rate: Learning rate
        :param num_features: Number of features
        '''
        learning_rate: Schedule
        num_features: int = 5

    
    @aclick.command(hierarchical=False)
    def train(model: Model):
        print(model)


This can be invoked as follows:

.. click:run::

    invoke(train, args=['--model', 'model(schedule("schedule 1", constants=[1.5, 2.0]))'], prog_name='python train.py')


Union of classes
----------------

We can specify multiple possible types for a parameter by using
`t.Union` type. The specified type will then be constructed from the passed
arguments.
This scenario is illustrated in the following example:

.. click:example::
    @dataclass
    class ModelA:
        learning_rate: float = 0.1
        num_features: int = 5

    @dataclass
    class ModelB:
        learning_rate: float = 0.2
        num_layers: int = 10

    
    @aclick.command(hierarchical=False)
    def train(model: t.Union[ModelA, ModelB]):
        print(f'Training {model}')


The output will look as follows:

.. click:run::

    invoke(train, args=['--model', 'model-b(num-layers=3)'], prog_name='python train.py')


Container types
---------------

The inline class representation natively supports `list`, `tuple`, `dict`, and `OrderedDict`.

* `list` or 'tuple' are represented as a comma-separated list of items inside of square brackets or parentheses::

     ["string val1", 'string val2', ..., string-val-N]

  or::

     ("string val1", 'string val2', ..., string-val-N)

* `dict` or `OrderedDict` are represented as a comma-separated list of pairs ``key=value`` inside of curly brackets::

  {key_1="string val1", key_2='string val2', ..., key_n=string-val-N}

We demonstrate all types in the following example:

.. click:example::
    
    @dataclass
    class Example:
        prop_list: t.List[str]
        prop_tuple: t.Tuple[int, float]
        prop_dict: t.Dict[str, t.Optional[int]]

    @aclick.command(hierarchical=False)
    def main(x: Example):
        print(x)


With the corresponding output:

.. click:run::

    invoke(main, args=['--x', 'example(["str 1", "str 2"], (10, 3.14), {p1=1, p2=2})'], prog_name='python main.py')
