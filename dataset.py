from torch.utils.data import TensorDataset, Dataset, DataLoader

class PairDataset(Dataset):
    def __init__(self, raw_dset, aug_dset):
        self.raw_dset = raw_dset
        self.aug_dset = aug_dset

    def __getitem__(self, index):
        raw_data = self.raw_dset[index]
        aug_data = self.aug_dset[index]
        
        # comb_x = [raw_data, aug_data]
        # comb_x = torch.stack(comb_x)
        
        return [raw_data, aug_data]

    def __len__(self):
        return len(self.raw_dset)