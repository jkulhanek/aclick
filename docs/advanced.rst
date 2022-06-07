Advanced features
=================

Getting values of all parameters
--------------------------------

All parameter values can be injected to the function
call using a special :class:`AllParameters` type.
If a parameter has this annotation,
a dictionary containing all parameters will
be passed as the parameter value. The :class:`AllParameters`
class supports additional string argument specifying the path
to the configuration. We show the usage in the following example:

.. click:example::
   
    @dataclass
    class Model:
        learning_rate: float = 1e-4
        num_features: int = 5

    
    @aclick.command()
    def train(config: aclick.AllParameters, model: Model, num_epochs: int = 1):
        import json
        print(json.dumps(config))


We get the following output:

.. click:run::

    invoke(train, args=["--model-num-features", "3"], prog_name='python train.py')

We can also specify the config path as follows:

.. click:example::
   
    @aclick.command()
    def train(config: aclick.AllParameters["model"], model: Model, num_epochs: int = 1):
        import json
        print(json.dumps(config))


With the following output:

.. click:run::

    invoke(train, args=["--model-num-features", "3"], prog_name='python train.py')
