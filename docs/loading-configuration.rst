Loading configuration
=====================

With `aclick` you can load a configuration file containing default values for all parameters.
This can be especially useful if you want to save the parameters and load them later. In order
to enable this feature, there is the :func:`configuration_option` decorator. This decorator
adds a new parameter called `--configuration` (althought the name can be customized) which
reads a configuration file before the actual parsing happens and sets all default values and
types based on this configuration.

By default the :func:`configuration_option` loads a `json` configuration file, however, custom
parsing function can also be specified. 

We show this feature in the following example:
.. click:example::
   
    @dataclass
    class Model:
        '''
        :param learning_rate: Learning rate
        :param num_features: Number of features
        '''
        learning_rate: float = 1e-4
        num_features: int = 5

    
    @aclick.command()
    @aclick.configuration_option('--config')
    def train(model: Model, num_epochs: int):
        print(f'''lr: {learning_rate},
    num_features: {model.num_features},
    num_epochs: {num_epochs}''')


We have the following help page:
.. click:run::

    invoke(train, args=['--help'], prog_name='python train.py')


Now, we create a file `config.json` with the following content:

::

    {
        "num_epochs": 3,
        "model": {
            "num_features": 4
        }
    }

If we pass our configuration file, the help page looks as follows:
.. click:run::

    invoke(train, args=['--config', 'config.json', '--help'], prog_name='python train.py')
