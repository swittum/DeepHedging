import torch
from .abstract_option import AbstractOption


class LookbackOption(AbstractOption):
    def __init__(self, underlier, call=True, strike=1.0, maturity=0.4):
        super().__init__(underlier, call, strike, maturity)

    def simulate(self, n_paths, dt=1/250):
        self.underlier.simulate(n_paths, self.maturity, dt)
    
    @property
    def payoff(self):
        maxvals = self.underlier.spot.max(axis=1).values
        return(-1)**(self.call+1)*torch.tensor([maxval-self.strike if maxval > self.strike else 0 for maxval in maxvals])
