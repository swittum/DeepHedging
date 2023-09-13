from abc import ABC, abstractmethod


class AbstractOption:
    def __init__(self, underlier, call, strike, maturity):
        self.underlier = underlier
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
    @abstractmethod
    def log_moneyness(self):
        pass

    @abstractmethod
    def reformat(self):
        pass