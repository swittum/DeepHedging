import torch
from .abstract_option import AbstractOption


class LookbackOption(AbstractOption):
    def __init__(self, underlier, short=True, call=True, strike=1.0, maturity=0.4):
        super().__init__(underlier, short, call, strike, maturity)

    def simulate(self, n_paths, dt=1/250):
        self.underlier.simulate(n_paths, self.maturity, dt)
    
    @property
    def payoff(self):
        maxvals = self.underlier.spot.max(axis=1).values
        return(-1)**self.short*torch.tensor([maxval-self.strike if maxval > self.strike else 0 for maxval in maxvals])

    # @property
    # def log_moneyness(self):
    #     if self.call:
    #         _log_moneyness = torch.log(self.underlier.spot/self.strike)
    #     else:
    #         _log_moneyness = torch.log(self.strike/self.underlier.spot)
    #     return _log_moneyness
    
    # def reformat(self):
    #     time = self.underlier.time
    #     log_moneyness = self.log_moneyness
    #     time_reshaped = time.unsqueeze(0)
    #     time_expanded = time_reshaped.expand(log_moneyness.shape[0], -1)
    #     data = torch.stack((log_moneyness, time_expanded), dim=-1)
    #     return data