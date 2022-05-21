Hierarchical parsing
====================

Hierarchical parsing allows complex class types to be expaded into individual parameters.
For each property of a class type a parameter is generated with name being the path to the
parameter. This is illustrated in the following example:

.. click:example::
   
    @dataclass
    class Model:
        '''
        :param learning_rate: Learning rate
        :param num_features: Number of features
        '''
        learning_rate: float = 1e-4
        num_features: int = 5

    
    @aclick.command
    def train(model: Model, num_epochs: int):
        print(f'''lr: {learning_rate},
    num_features: {model.num_features},
    num_epochs: {num_epochs}''')


The corresponding help page looks as follows:

.. click:run::

    invoke(train, args=['--help'], prog_name='python train.py')


Note, that in some cases hierarchical parsing may not support your type,
default value, etc. In particular, positional arguments cannot be expanded
in hierarchical parsing. If your type in not supported, you can either inline
it by wrapping it with :func:`aclick.default_from_str` wrapper or turn of hierarchical
parsing by passing ``hierarchical = False`` to the command constructor.

With a deep class structure the parameter names can grow long. Parameters
can be renamed to prevent long names. For more information see :doc:`hierarchical parsing <hierarchical-parsing>` and :class:`FlattenParameterRenamer`.

Custom classes
--------------

User defined classes are naturally expand to individual parameters if ``hierarchical = True`` (default).
We can have a class hierarchy and parameters will be expanded to the lowest level. This
can be seen in the following example:

.. click:example::
    class Schedule:
        def __init__(self, type: str, constant: float = 1e-4):
            self.type = type
            self.constant = constant
   
    @dataclass
    class Model:
        '''
        :param learning_rate: Learning rate
        :param num_features: Number of features
        '''
        learning_rate: Schedule
        num_features: int = 5

    
    @aclick.command
    def train(model: Model, num_epochs: int):
        pass


The corresponding help page looks as follows:

.. click:run::

    invoke(train, args=['--help'], prog_name='python train.py')


Optional values
---------------

If a property of a class type is optional, there will be a boolean
flag with the property name indicating whether the class is actually present.
This is illustrated in the following example:

.. click:example::
    class Schedule:
        def __init__(self, type: str, constant: float = 1e-4):
            self.type = type
            self.constant = constant
   
    @dataclass
    class Model:
        '''
        :param learning_rate: Learning rate
        :param num_features: Number of features
        '''
        learning_rate: t.Optional[Schedule] = None
        num_features: int = 5

    
    @aclick.command
    def train(model: Model, num_epochs: int):
        pass


The corresponding help page looks as follows:

.. click:run::

    invoke(train, args=['--help'], prog_name='python train.py')


And after specifying that we want to instantiate the ``learning_rate`` instance:

.. click:run::

    invoke(train, args=['--model-learning-rate', '--help'], prog_name='python train.py')



Union of classes
----------------

We can also specify multiple types for a parameter or property and
the concrete type will be specified when invoking the command.
This scenario is illustrated in the following example:

.. click:example::
    @dataclass
    class ModelA:
        '''
        :param learning_rate: Learning rate
        :param num_features: Number of features
        '''
        learning_rate: float = 0.1
        num_features: int = 5

    @dataclass
    class ModelB:
        '''
        :param learning_rate: Learning rate
        :param num_layers: Number of layers
        '''
        learning_rate: float = 0.2
        num_layers: int = 10

    
    @aclick.command
    def train(model: t.Union[ModelA, ModelB], num_epochs: int):
        pass


The corresponding help page looks as follows:

.. click:run::

    invoke(train, args=['--help'], prog_name='python train.py')


And after specifying that we want to use ``ModelB`` class:

.. click:run::

    invoke(train, args=['--model', 'model-b', '--help'], prog_name='python train.py')
