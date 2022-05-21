Parameter renaming
==================

All generated parameters of a command can be renamed by passing a callback
function to the :class:`Command` constructor. This function is given the original
name and returns a new name of the parameter. In the `aclick` library, there
are two renamer classes implemented by default: :class:`RegexParameterRenamer` and :class:`FlattenParameterRenamer`


Renaming parameters with a regex expression
-------------------------------------------

The class :class:`RegexParameterRenamer` renames the parameter using regex expressions.
The constuctor of the class accepts a list of pairs of regex expression and the corresponding
replacement value. First, the list is searched until a match is found. If no expression matches
the parameter name the name is left unchanged. After the matching expression is found, the matching
part of the parameter name is replaced with the replacement value. Note, that the replacement value
can reference captured groups using `\1`, `\2`, ... syntax.

We will demonstrate regex parameter renaming in the following example:

.. click:example::
   
    @dataclass
    class Model:
        '''
        :param learning_rate: Learning rate
        :param num_features: Number of features
        '''
        learning_rate: float = 1e-4
        num_features: int = 5

    
    @aclick.command(map_parameter_name=aclick.RegexParameterRenamer([("model\.(.*)_", r"\1_param_")]))
    def train(model: Model, num_epochs: int):
        print(f'''lr: {learning_rate},
    num_features: {model.num_features},
    num_epochs: {num_epochs}''')


The corresponding help page looks as follows:

.. click:run::

    invoke(train, args=['--help'], prog_name='python train.py')

In the example the regex expression matched prefix `model.`, captured the following text until 
an underscore and replaced this prefix with the matched text after `model.` followed by `_param_`.
Therefore, we have the following replacements:

::

    model.learning_rate ->  learning_param_rate
    model.num_features ->  num_param_features
    num_epochs -> num_epochs


Flattening nested parameter structure
-------------------------------------

With a deep class structure the parameter names can grow long. 
The class :class:`FlattenParameterRenamer` can be used to prevent such long names.
In hierarchical parsing the name of a parameter is the full path to the parameter.
For example if a function `train` accepts parameter `model` of class `Model`, the
`Model`'s parameters will start with the prefix `model.` (all dots and underscores are
replaced with dashes at the end). By using the :class:`FlattenParameterRenamer`, we 
can remove one or more parts of the parameter path from the parameter name.

This can be seen in the following example:

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

    
    @aclick.command(map_parameter_name=aclick.FlattenParameterRenamer(1))
    def train(model: Model, num_epochs: int):
        pass


The corresponding help page looks as follows:

.. click:run::

    invoke(train, args=['--help'], prog_name='python train.py')

In this example, one level is removed from the parameter paths. Therefore
we have the following replacements:

::

    model.learning_rate.type ->  learning_rate.type
    model.learning_rate.constant ->  learning_rate.constant
    model.num_features ->  num_features
    num_epochs -> num_epochs
