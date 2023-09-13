import torch
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from nn import NeuralNetwork
from finance import pl


class Hedger:
    def __init__(self, model, underlier, derivative):
        self.model = model
        self.underlier = underlier
        self.derivative = derivative

    def test(self, n_paths):
        self.derivative.simulate(n_paths=n_paths)
        input = self.derivative.reformat()
        output = torch.squeeze(self.model(input), dim=-1)
        hist = pl(self.derivative, self.underlier, output)
        return hist.detach().numpy()

    def compute_loss(self, output, p=0.1):
        output = torch.squeeze(output, dim=-1)
        hist = pl(self.derivative, self.underlier, output)
        hist_sorted, ind_sorted = torch.sort(hist)
        n, = hist_sorted.shape
        expected_shortfall = -torch.mean(hist_sorted[:round(p*n)])
        gradients = torch.autograd.grad(outputs=expected_shortfall, inputs=self.model.parameters(),
                                        retain_graph=True, create_graph=True)
        return expected_shortfall, gradients

    def fit(self, n_epochs=1, n_paths=200):
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        history = []

        for epoch in range(n_epochs):
            self.derivative.simulate(n_paths)
            input = self.derivative.reformat()
            output = self.model(input)

            loss, gradients = self.compute_loss(output)
            optimizer.zero_grad()
            for param, grad in zip(self.model.parameters(), gradients):
                param.grad = grad  # Assign the computed gradients to the model's parameters
            optimizer.step()
            history.append(loss.detach().numpy())
            if epoch % 10 == 0:
                print(loss)

        return history
    

def main():
    from finance import BrownianStock
    from finance import EuropeanOption
    n_paths = 200
    n_features = 2
    model = NeuralNetwork(n_features, 1)
    underlier = BrownianStock(mu=0, sigma=.2)
    derivative = EuropeanOption(underlier, call=True, strike=1.1, maturity=100/250)
    hedger = Hedger(model, underlier, derivative)
    history = hedger.fit(n_epochs=100, n_paths=50)
    data = hedger.test(5000)
    plt.hist(data, bins=100)
    plt.show()


    # plt.plot(history)
    # plt.show()


if __name__ == '__main__':
    main()