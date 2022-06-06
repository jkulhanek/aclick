import gin
from dataclasses import dataclass
import aclick

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


train()
