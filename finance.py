import numpy as np
import torch


class BrownianStock:
    def __init__(self, mu=1, sigma=0.1):
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


class EuropeanOption:
    def __init__(self, underlier, call, strike, maturity):
        self.underlier = underlier
        self.call = call
        self.strike = strike
        self.maturity = maturity

    def simulate(self, n_paths, dt=1/250):
        self.underlier.simulate(n_paths, self.maturity, dt=dt)

    @property
    def payoff(self):
        _payoff = []
        for spot in self.underlier.spot:
            thres = spot[-1]-self.strike
            _payoff.append(thres if thres > 0 else 0)
        return torch.tensor(_payoff)
    
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
    

def pl(derivative, underlier, ratios):
    payoffs = derivative.payoff
    stocks = underlier.spot
    n_paths = len(stocks[:, 0])
    n_time = len(stocks[0, :])
    portfolio = [0] * n_paths
    for i, (stock, ratio) in enumerate(zip(stocks, ratios)):
        portfolio[i] -= payoffs[i]
        # portfolio[i] += payoffs[i]
        for j in range(n_time-1):
            portfolio[i] += ratio[j]*(stock[j+1]-stock[j])
    return torch.stack(portfolio)


def main():
    import matplotlib.pyplot as plt
    underlier = BrownianStock(.4, .2)
    derivative = EuropeanOption(underlier, call=True, strike=1.1, maturity=50/250)
    derivative.simulate(n_paths=5_000)
    stocks = derivative.underlier.spot
    payoffs = derivative.payoff
    units = torch.ones(5_000, 50)
    profit = pl(derivative, underlier, units)


if __name__ == '__main__':
    main()
    exit(0)
