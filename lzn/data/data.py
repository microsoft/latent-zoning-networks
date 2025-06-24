from abc import ABC, abstractmethod


class Data(ABC):
    @abstractmethod
    def get_data_loader_and_sampler(
        self, batch_size, num_workers, train, rank, world_size
    ):
        pass

    @abstractmethod
    def num_samples(self, train):
        pass

    @property
    def num_classes(self):
        return self._num_classes
