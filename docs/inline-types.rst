Inline types
============

.. currentmodule:: aclick


Inline types allows complex class structures to be parsed as string.

Example
------------

First, we will motivate the reader by a simple example:

.. click:example::

    import aclick, click, typing as t

    class ModelA:
        def __init__(self, name: str, n_layers: int):
            self.name = name
            self.n_layers = n_layers

        def train(self):
            click.echo('Training model A with {model.n_layers} layers.')

    class ModelB(ModelA):
        def __init__(self, *args, n_blocks: int = 5, **kwargs):
            super().__init__(*args, **kwargs)
            self.n_blocks = n_blocks

        def train(self):
            click.echo('Training model B with {model.n_blocks} blocks.')


    @aclick.command()
    def main(model: t.Union[ModelA, ModelB]):
        model.train()

    if __name__ == '__main__':
        main()

We can invoke the script as follows:

.. click:run::

    invoke(main, args=['--model', 'model_b(test, n_layers=2)'], prog_name='python hello.py')
