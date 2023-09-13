from abc import abstractmethod, ABC


class AbstractStock(ABC):
    @abstractmethod
    def simulate(self, n_paths, timescale, dt):
        """ Implementation of stock's time evolution """
