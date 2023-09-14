from abc import ABC, abstractmethod
import torch


class AbstractOption:
    def __init__(self, underlier, short, call, strike, maturity):
        self.underlier = underlier
        self.short = short
        self.call = call
        self.strike = strike
        self.maturity = maturity

    @abstractmethod
    def simulate(self, n_paths, dt):
        pass

    @property
    @abstractmethod
    def payoff(self):
        pass

    @property
    def log_moneyness(self):
        if self.call:
            _log_moneyness = torch.log(self.underlier.spot/self.strike)
        else:
            _log_moneyness = torch.log(self.strike/self.underlier.spot)
        return _log_moneyness

    @property
    def volatility(self):
        return self.underlier.volatility

    def reformat(self, in_features):
        features = {
            'log_moneyness': self.log_moneyness,
            'volatility': self.volatility,
            'time': self.underlier.time.unsqueeze(0).expand(self.log_moneyness.shape[0], -1)
        }
        features = [features[feature] for feature in in_features]
        data = torch.stack(features, dim=-1) 
        return data