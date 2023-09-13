import torch
import numpy as np
from .abstract_stock import AbstractStock


class CIRStock(AbstractStock):
    def __init__(self, kappa, theta, sigma, r0):
        self.kappa = kappa  # Speed of mean reversion
        self.theta = theta  # Long-term mean
        self.sigma = sigma  # Volatility
        self.r0 = r0        # Initial interest rate

    def simulate(self, n_paths, timescale=100/250, dt=1/250):
        timescale = int(timescale*250)
        rates = torch.zeros((n_paths, timescale+1))
        rates[:, 0] = self.r0
        
        for i in range(1, timescale+1):
            dW = torch.randn(n_paths)*np.sqrt(dt)
            rates[:, i] = (
                rates[:, i - 1] +
                self.kappa * (self.theta - rates[:, i-1]) * dt +
                self.sigma * np.sqrt(np.maximum(rates[:, i-1], 0)) * dW
            )
            rates[:, i] = np.maximum(rates[:, i], 0)  # Ensure rates are non-negative
        self.spot = rates