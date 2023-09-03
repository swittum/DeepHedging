import torch
from torch.nn import Module
from torch.nn import Linear
from torch import relu
from torch.distributions import Normal


def cdf(input):
    return Normal(0, 1).cdf(input)


class NeuralNetwork(Module):
    def __init__(self, in_features, out_features):
        super(NeuralNetwork, self).__init__()
        self.fc1 = Linear(in_features, 16)
        self.fc2 = Linear(16, 16)
        self.fc3 = Linear(16, 16)
        self.fc4 = Linear(16, out_features)

    def forward(self, X):
        X = relu(self.fc1(X))
        X = relu(self.fc2(X))
        X = relu(self.fc3(X))
        return self.fc4(X)
    

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
