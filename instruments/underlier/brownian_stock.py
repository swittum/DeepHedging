import torch
import numpy as np
from .abstract_stock import AbstractStock


class BrownianStock(AbstractStock):
    def __init__(self, mu=1.0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def simulate(self, n_paths, timescale=100/250, dt=1/250):
        self.time = torch.arange(0, timescale, dt)
        n = round(timescale/dt)
        self.spot = []
        for i in range(n_paths):
            dW = np.sqrt(dt)*torch.randn(n)
            W = torch.cumsum(dW, dim=0)
            self.spot.append(torch.exp((self.mu-self.sigma**2/2)*self.time+self.sigma*W))
        self.spot = torch.stack(self.spot)
        self.volatility = self.sigma*torch.ones(self.spot.shape)