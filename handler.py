import yaml
from hedger import Hedger
from nn import NeuralNetwork, EuropeanBlackScholes, NoHedge
from instruments import BrownianStock, HestonStock
from instruments import EuropeanOption, LookbackOption

def remove_keys(dictionary, *keys) :
    for key in keys:
        if key in dictionary:
            del dictionary[key]
        else:
            raise KeyError(f'{key} not found in dictionary')
    return dictionary

class Handler:
    def __init__(self, config):
        file = open(config, 'r')
        self.config = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
        self._setup()

    def _setup(self):
        self.model = self._setup_model()
        self.underlier = self._setup_underlier()
        self.derivative = self._setup_derivative()
        self.hedger = self._setup_hedger()


    def _setup_model(self):
        models = {'NeuralNetwork': NeuralNetwork,
                  'EuropeanBlackScholes': EuropeanBlackScholes,
                  'NoHedge': NoHedge}
        model = models[self.config['model']['type']]
        kwargs = remove_keys(self.config['model'], 'type')
        model = model(**kwargs)
        return model
    
    def _setup_underlier(self):
        underliers = {'BrownianStock': BrownianStock,
                      'HestonStock': HestonStock}
        underlier = underliers[self.config['underlier']['type']]
        kwargs = remove_keys(self.config['underlier'], 'type')
        underlier = underlier(**kwargs)
        return underlier
    
    def _setup_derivative(self):
        derivatives = {'EuropeanOption': EuropeanOption,
                       'LookbackOption': LookbackOption}
        derivative = derivatives[self.config['derivative']['type']]
        kwargs = remove_keys(self.config['derivative'], 'type')
        derivative = derivative(self.underlier, **kwargs)
        return derivative
    
    def _setup_hedger(self):
        hedger = Hedger(self.model, self.underlier, self.derivative)
        return hedger
    
    def _setup_training(self):
        n_epochs = self.config['training']['n_epochs']
        n_paths = self.config['training']['n_paths']
        return n_epochs, n_paths
    
    def run(self):
        n_epochs, n_paths = self._setup_training()
        history = self.hedger.fit(n_epochs=n_epochs, n_paths=n_paths)
        results = self.hedger.test(n_paths=5000)
        return history, results