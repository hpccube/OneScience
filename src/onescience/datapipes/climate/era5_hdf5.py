import h5py
import numpy as np
import torch
import glob

from ..datapipe import Datapipe
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class ERA5HDF5Datapipe(Datapipe):
    def __init__(self, params, distributed):
        self.params = params
        self.distributed = distributed
    def train_dataloader(self):
        data = ERA5Dataset(params=self.params, data_paths=self.params.train_data_dir)
        sampler = DistributedSampler(data, shuffle=True) if self.distributed else None
        data_loader = DataLoader(data,
                                 batch_size=self.params.batch_size,
                                 drop_last=True,
                                 num_workers=self.params.num_workers,
                                 pin_memory=True,
                                 shuffle=False if self.distributed else True,
                                 sampler=sampler)
        return data_loader, sampler

    def val_dataloader(self):
        data = ERA5Dataset(params=self.params, data_paths=self.params.val_data_dir)
        sampler = DistributedSampler(data, shuffle=False) if self.distributed else None
        data_loader = DataLoader(data,
                                 batch_size=self.params.batch_size,
                                 drop_last=True,
                                 num_workers=self.params.num_workers,
                                 pin_memory=True,
                                 shuffle=False if self.distributed else True,
                                 sampler=sampler)
        return data_loader, sampler

    def test_dataloader(self):
        data = ERA5Dataset(params=self.params, data_paths=self.params.test_data_dir)
        data_loader = DataLoader(data,
                                 batch_size=self.params.batch_size,
                                 drop_last=False,
                                 num_workers=self.params.num_workers,
                                 pin_memory=True,
                                 shuffle=False)
        return data_loader

    def __len__(self):
        return self.length


class ERA5Dataset(Dataset):
    def __init__(self, params, data_paths, patch_size=[1, 1]):
        self.params = params
        self.data_files = None
        self.data_dir = data_paths

        self.mu = np.load(f'{self.params.stats_dir}/global_means.npy')
        self.sd = np.load(f'{self.params.stats_dir}/global_stds.npy')
        self.patch_size = patch_size
        self.parse_dataset_files()
        
    def parse_dataset_files(self):
        self.data_paths = glob.glob(f'{self.data_dir}/*.h5')
        self.data_paths.sort()
        self.n_years = len(self.data_paths)
        with h5py.File(self.data_paths[0], "r") as f:
            data_samples_per_year = f["fields"].shape[0]
            self.img_shape = f["fields"].shape[2:]
            self.channels = [i for i in range(f["fields"].shape[1])]
            self.num_samples_per_year = data_samples_per_year
            self.img_shape = [s - s % self.patch_size[i] for i, s in enumerate(self.img_shape)]
            self.total_length = self.n_years * self.num_samples_per_year

    
    def __getitem__(self, idx):
        if self.data_files is None:
            self.data_files = [h5py.File(path, "r") for path in self.data_paths]

        if idx > self.total_length - self.params.num_steps - 1:
            idx = self.total_length - self.params.num_steps - 1
        invar_idx = idx
        outvar_idx = idx + self.params.num_steps

        if outvar_idx == self.num_samples_per_year- 1:
            outvar_idx = self.num_samples_per_year - 1
            invar_idx = self.num_samples_per_year - 1 - self.params.num_steps

        invar_year_idx = invar_idx // self.num_samples_per_year
        invar_in_idx = invar_idx % self.num_samples_per_year
        outvar_year_idx = outvar_idx // self.num_samples_per_year
        outvar_in_idx = outvar_idx % self.num_samples_per_year


        invar_data = self.data_files[invar_year_idx]["fields"]
        invar = invar_data[invar_in_idx: invar_in_idx+1]

        outvar_data = self.data_files[outvar_year_idx]["fields"]
        outvar = outvar_data[outvar_in_idx: outvar_in_idx+1]

        # numpy数组转化为tensor
        invar = torch.as_tensor(invar)
        outvar = torch.as_tensor(outvar)

        h, w = self.img_shape
        invar = invar[:, :, :h, :w]
        outvar = outvar[:, :, :h, :w]

        invar = (invar - self.mu) / self.sd
        outvar = (outvar - self.mu) / self.sd

        return invar.squeeze(0), outvar.squeeze(0)
    
    def __len__(self):
        return self.total_length # // self.batch_size
