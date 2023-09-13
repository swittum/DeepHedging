import torch
from .abstract_option import AbstractOption


class EuropeanOption(AbstractOption):
    def __init__(self, underlier, call=True, strike=1.0, maturity=0.4):
        super().__init__(underlier, call, strike, maturity)

    def simulate(self, n_paths, dt=1/250):
        self.underlier.simulate(n_paths, self.maturity, dt=dt)

    @property
    def payoff(self):
        return (-1)**(self.call+1)*torch.tensor([max(spot[-1]-self.strike, 0) for spot in self.underlier.spot])
    
    @property
    def log_moneyness(self):
        _log_moneyness = torch.log(self.underlier.spot/self.strike)
        return _log_moneyness
    
    def reformat(self):
        time = self.underlier.time
        log_moneyness = self.log_moneyness
        time_reshaped = time.unsqueeze(0)
        time_expanded = time_reshaped.expand(log_moneyness.shape[0], -1)
        data = torch.stack((log_moneyness, time_expanded), dim=-1)
        return data