from abc import ABC, abstractmethod


class TScheduler(ABC):

    @abstractmethod
    def sample(self, batch_size, device):
        pass
