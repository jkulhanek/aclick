Loading configuration
=====================

With `aclick` you can load a configuration file containing default values for all parameters.
This can be especially useful if you want to save the parameters and load them later. In order
to enable this feature, there is the :func:`configuration_option` decorator. This decorator
adds a new parameter called `--configuration` (althought the name can be customized) which
reads a configuration file before the actual parsing happens and sets all default values and
types based on this configuration.

By default the :func:`configuration_option` loads one of the registered configuration provider
based on the filename of the config file and parses the file using the configuration provider.
The following configuration providers are supported in the default :func:`parse_configuration`
function:

- `*.json`: a JSON parser is used
- `*.yml|*.yaml`: a YAML parser is used
- `*.gin`: the file is parsed using the `gin-config` library. In this case, all used classes have to be decorated with `@gin.configurable`.

A custom configuration parser can also be registered using the :func:`register_configuration_parser`
function which takes a regex matching the supported filenames and a parse function as its arguments.
Alternatively, the :func:`parse_configuration` function can be overrided in the :func:`configuration_option`
decorator.


JSON and YAML
-------------

JSON and YAML files are supported in the default :func:`parse_configuration` function.
We show this feature for a JSON configuration file in the following example:

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

.. click:file::
    config.json

    {
        "num_epochs": 3,
        "model": {
            "num_features": 4
        }
    }

If we pass our configuration file, the help page looks as follows:

.. click:run::

    invoke(train, args=['--config', 'config.json', '--help'], prog_name='python train.py')

Gin
---

The `gin-config library <https://github.com/google/gin-config>`_ can be used to configure the parameters.
In this case, the library must be installed and the `.gin` configuration file must be compatible with
the classes and functions, e.g., all required classes must be decorated with `@gin.configurable`.
We show the parsing of a gin-config configuration file in the following example:

.. click:example::
   
    @gin.configurable
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
    @gin.configurable
    def train(model: Model, num_epochs: int):
        print(f'''lr: {model.learning_rate},
    num_features: {model.num_features},
    num_epochs: {num_epochs}''')


Now, we create a file `config.gin` with the following content:

.. click:file::
    config.gin

    Model.learning_rate = 0.1
    Model.num_features = 4
    train.model = @Model()
    train.num_epochs = 1

If we pass our configuration file, the help page looks as follows:

.. click:run::

    invoke(train, args=['--config', 'config.gin', '--help'], prog_name='python train.py')


Restricting configuration to a custom parameter
-----------------------------------------------

Sometimes you want to restrict the configuration to a single parameter.
In AClick, this can easily be achieved using the :func:`restrict_parse_configuration`
function. This function takes as its first parameter the path to the parameter
for which we will parse the configuration. The (optional) second parameter
is the function that will be used to parse the configuration.

We show this feature in the following example:

.. click:example::

    @aclick.command()
    @aclick.configuration_option(
        '--config', 
        parse_configuration=aclick.configuration.restrict_parse_configuration("model"))
    def train(model: Model, num_epochs: int):
        print(f'''lr: {model.learning_rate},
    num_features: {model.num_features},
    num_epochs: {num_epochs}''')


Now, we create a file `config.json` with the following content:

.. click:file::
    config.json

    {
        "num_features": 4
    }

If we pass our configuration file, the help page looks as follows:

.. click:run::

    invoke(train, args=['--config', 'config.json', '--help'], prog_name='python train.py')


Custom configuration provider
-----------------------------

A custom configuration provider can be registered using the :func:`register_configuration_provider` decorator.
We demonstrate this in a simple TXT configuration provider.
First, we create a file `config.txt` with the following content:

.. click:file::
    config.txt

    model.num_features=4
    num_epochs=3

We then register our configuration provider:

.. click:example::
   
    @aclick.register_configuration_provider(r".*\.txt")
    def parse_txt_configuration(fp, ctx):
        cfg = dict()
        for line in fp:
            line = line.rstrip("\n")
            key, val = line.split("=")
            path = key.split(".")
            local_cfg = cfg
            for part in path[:-1]:
                local_cfg[part] = local_cfg[part] if part in local_cfg else dict()
                local_cfg = local_cfg[part]
            local_cfg[path[-1]] = int(val)
        return cfg

    
If we pass our configuration file, the help page looks as follows:

.. click:run::

    invoke(train, args=['--config', 'config.txt', '--help'], prog_name='python train.py')
