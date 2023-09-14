import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch import relu
from torch.distributions import Normal


def cdf(input):
    return Normal(0, 1).cdf(input)


class NeuralNetwork(Module):
    def __init__(self, in_features, out_features, n_layers=4, n_neurons=16):
        super(NeuralNetwork, self).__init__()
        layers = []
        layers.append(Linear(in_features, n_neurons))
        for _ in range(n_layers - 2):
            layers.append(Linear(n_neurons, n_neurons))
        layers.append(Linear(n_neurons, out_features))
        self.layers = ModuleList(layers)
        
    def forward(self, X):
        for layer in self.layers[:-1]:
            X = F.relu(layer(X))
        return self.layers[-1](X)
    

class EuropeanBlackScholes(Module):
    def __init__(self, derivative):
        self.derivative = derivative

    def compute_delta(self, r):
        sigma = self.derivative.underlier.sigma
        ttm = self.derivative.maturity-self.derivative.underlier.time
        log_m = self.derivative.log_moneyness
        d1 = 1/(sigma*torch.sqrt(ttm))*(log_m+(r+sigma**2/2)*ttm)
        return cdf(d1)
    

class NoHedge(Module):
    def __init__(self, derivative):
        self.derivative = derivative

    def compute_hedge(self):
        n, m = self.derivative.underlier.spot.shape
        return torch.zeros(n, m)
    
