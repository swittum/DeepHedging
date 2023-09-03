from abc import abstractmethod, ABC


class BaseStock(ABC):
    @abstractmethod
    def simulate(self, n_paths, timescale, dt):
        """ Implementation of stock's time evolution """
