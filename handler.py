import yaml
from hedger import Hedger
from nn import NeuralNetwork, EuropeanBlackScholes, NoHedge
from instruments import BrownianStock, CIRStock
from instruments import EuropeanOption, LookbackOption

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
        in_features = self.config['model']['in_features']
        out_features = self.config['model']['out_features']
        model = models[self.config['model']['type']](in_features, out_features)
        return model
    
    def _setup_underlier(self):
        underliers = {'BrownianStock': BrownianStock,
                      'CIRStock': CIRStock}
        mu = self.config['underlier']['mu']
        sigma = self.config['underlier']['sigma']
        underlier = underliers[self.config['underlier']['type']](mu, sigma)
        return underlier
    
    def _setup_derivative(self):
        derivatives = {'EuropeanOption': EuropeanOption,
                       'LookbackOption': LookbackOption}
        call = self.config['derivative']['call']
        strike = self.config['derivative']['strike']
        maturity = self.config['derivative']['maturity']
        derivative = derivatives[self.config['derivative']['type']](self.underlier, call, strike, maturity)
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