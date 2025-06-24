from torch.utils.data import Dataset


class WithIndex(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return *self._dataset[idx], idx
