import torch
import numpy as np
from .abstract_stock import AbstractStock


class HestonStock(AbstractStock):
    def __init__(self, mu=0.05, kappa=1.5, theta=0.04, xi=0.3, nu0=0.04):
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.nu0 = nu0

    def simulate(self, n_paths, timescale=100/250, dt=1/250):
        n = round(timescale/dt)
        self.time = torch.arange(0, timescale, dt)
        self.spot = torch.zeros(n_paths, n)
        
        dW1 = torch.normal(0, torch.sqrt(torch.tensor(dt)), size=(n_paths, n))
        dW2 = torch.normal(0, torch.sqrt(torch.tensor(dt)), size=(n_paths, n))
        
        nu = torch.zeros(n_paths, n)
        S = torch.zeros(n_paths, n)
        
        nu[:, 0] = self.nu0
        S[:, 0] = 1.0
    
        for j in range(1, n):
            dnu = self.kappa * (self.theta - nu[:, j-1]) * dt + self.xi * torch.sqrt(nu[:, j-1]) * dW1[:, j-1]
            nu[:, j] = torch.max(nu[:, j-1] + dnu, torch.tensor(0.0))
            dS = self.mu * S[:, j-1] * dt + torch.sqrt(nu[:, j-1]) * S[:, j-1] * dW2[:, j-1]
            S[:, j] = torch.max(S[:, j-1] + dS, torch.tensor(0.0))
        
        self.volatility = torch.sqrt(nu)
        self.spot = S


