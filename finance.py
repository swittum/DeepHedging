import torch

   
def pl(derivative, underlier, ratios):
    payoffs = derivative.payoff
    stocks = underlier.spot
    stock_diff = stocks[:, 1:] - stocks[:, :-1]
    weighted_diff = torch.mul(ratios[:, :-1], stock_diff)
    portfolio = torch.sum(weighted_diff, dim=1)+payoffs
    return portfolio
