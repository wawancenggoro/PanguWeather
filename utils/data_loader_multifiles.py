import glob
import logging

import h5py
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import os

def pad_data(t1, t2, patch_size):
    """
    Perform padding for outermost patching step.

    t1: Tensor
        pressure-level tensors
    t2: Tensor
        surface-level tensors
    """
    # perform padding for patch embedding step
    input_shape = t1.shape  # shape is (5 variables x 13 pressure levels x 721 latitude x 1440 longitude)

    x1_pad    = (patch_size[0] - (input_shape[1] % patch_size[0])) % patch_size[0] // 2
    x2_pad    = (patch_size[0] - (input_shape[1] % patch_size[0])) % patch_size[0] - x1_pad
    y1_pad    = (patch_size[1] - (input_shape[2] % patch_size[1])) % patch_size[1] // 2
    y2_pad    = (patch_size[1] - (input_shape[2] % patch_size[1])) % patch_size[1] - y1_pad
    z1_pad    = (patch_size[2] - (input_shape[3] % patch_size[2])) % patch_size[2] // 2
    z2_pad    = (patch_size[2] - (input_shape[3] % patch_size[2])) % patch_size[2] - z1_pad

    # pad pressure fields input and output
    t1 = torch.nn.functional.pad(t1, pad=(z1_pad, z2_pad, y1_pad, y2_pad, x1_pad, x2_pad), mode='constant', value=0)

    # pad 
    t2  = torch.nn.functional.pad(t2, pad=(z1_pad, z2_pad, y1_pad, y2_pad), mode='constant', value=0)

    return t1, t2
  
def get_data_loader_synthetic(params, data_path):
    #TODO: save generated random numbers to files, read from those files
    with open(os.path.join(data_path, 'input_upper.npy'), 'rb') as f:
        input_upper = np.load(f)

    with open(os.path.join(data_path, 'input_surface.npy'), 'rb') as f:
        input_surface = np.load(f)

    with open(os.path.join(data_path, 'target_upper.npy'), 'rb') as f:
        target_upper = np.load(f)

    with open(os.path.join(data_path, 'target_surface.npy'), 'rb') as f:
        target_surface = np.load(f)

    # input_upper = torch.randn((params['batch_size'], 5, 13, 721, 1440)).to(torch.float32)
    # input_surface = torch.randn((params['batch_size'], 4, 721, 1440)).to(torch.float32)
    input_upper = torch.tensor(input_upper).to(torch.float32)
    input_surface = torch.tensor(input_surface).to(torch.float32)
    
    input_upper, input_surface = pad_data(input_upper, input_surface, params['patch_size'])

    #TODO: Why is the model output padded on lat (721->724) and vert (13->14)? Need to find out
    # target_upper = torch.randn((params['batch_size'], 5, 14, 724, 1440)).to(torch.float32)
    # target_surface = torch.randn((params['batch_size'], 4, 724, 1440)).to(torch.float32)
    target_upper = torch.tensor(target_upper).to(torch.float32)
    target_surface = torch.tensor(target_surface).to(torch.float32)

    dataset = TensorDataset(input_upper, input_surface, target_upper, target_surface)

    sampler = DistributedSampler(dataset, shuffle=True)
    # dataloader = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=True)
    dataloader = DataLoader(dataset,
                            batch_size=int(params['batch_size']),
                            shuffle=(sampler is None),
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=torch.cuda.is_available())

    return dataloader

def get_data_loader(params, file_path, distributed, mode, patch_size, subset_size=None, forecast_length=1, two_dimensional=False):
    """
    Return data loader for 2 or 3D dataset.
    
    params: Dict
        configuration file
    file_path: String
        path to data directory
    distributed: bool
        flag for DDP
    mode: String
        of value 'training', 'testing', 'validation'
    device: String
        device that the code is running/offloaded on
    patch_size: Tuple(int, int, Optional[int])
        Number of pixels in ([vert], lat, lon) dimensions per patch
    forecast_length: int
        For training, always 1. For validation, defines the number of autoregressive steps to roll-out to.
    two_dimensional: bool
        Flag for 2D vs 3D model.

    """
    if not two_dimensional:
        dataset = PanguDataset(params, file_path, mode, patch_size, forecast_length=forecast_length)
    else:
        dataset = Pangu2DDataset(params, file_path, mode, patch_size, forecast_length=forecast_length)

    # If we are setting a subset
    if subset_size is not None:
        subset_indices = torch.randperm(len(dataset))[:subset_size]
        dataset = Subset(dataset, subset_indices)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    
    dataloader = DataLoader(dataset,
                            batch_size=int(params['batch_size']),
                            shuffle=(sampler is None),
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=torch.cuda.is_available())

    return dataloader

class PanguDataset(Dataset):
    """Define 3D dataset."""
    
    def __init__(self, params, file_path, mode, patch_size, forecast_length=1):
        """
        initialize.

        params: Dict
            configuration file
        file_path: String
            path to data directory
        distributed: bool
            flag for DDP
        mode: String
            of value 'training', 'testing', 'validation'
        device: String
            device that the code is running/offloaded on
        patch_size: Tuple(int, int, Optional[int])
            Number of pixels in ([vert], lat, lon) dimensions per patch
        forecast_length: int
            For training, always 1. For validation, defines the number of autoregressive steps to roll out to.
        """
        self.params = params
        self.file_path = file_path
        self.mode = mode
        self.dt = params['dt']
        self.filetype = params['filetype']
        self.deltaTDivisor = params['delta_T_divisor']
        self.forecast_length = forecast_length
        self._get_files_stats(file_path, lite=params['Lite'], mode=self.mode)
        
        # If data is downloaded from weatherbench, the 6-hourly subsampled data is stored in the reverse pressure level order.
        # Need to reverse so that the 0th presssure level is 1000 hPa, 1st is 925 hPa, etc.
        if params['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr':
            self.level_ordering = range(13-1, -1, -1)
        elif params['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/era5.zarr':
            self.level_ordering = range(0, 13)
        else: # baseline assuption is subsampled WB2 data
            self.level_ordering = range(13-1, -1, -1)


        self.p_mean = np.load(params['pressure_static_data_path'], allow_pickle=True)[0].reshape(5, 13, 1, 1)
        self.p_std  = np.load(params['pressure_static_data_path'], allow_pickle=True)[1].reshape(5, 13, 1, 1)
        self.s_mean = np.load(params['surface_static_data_path'], allow_pickle=True)[0].reshape(4, 1, 1)
        self.s_std  = np.load(params['surface_static_data_path'], allow_pickle=True)[1].reshape(4, 1, 1)
        self.patch_size = patch_size
        
        if "normalize" in params.keys():
            self.normalize = params.normalize
        else:
            self.normalize = True #by default turn on normalization if not specified in config

    def _get_files_stats(self, file_path, lite=False, mode='train'):
        """Filter desired time points based on parameters and return file statistics."""
        if self.filetype == 'hdf5':
            self.files_paths_pressure = glob.glob(file_path + "/????.h5") # indicates file paths for pressure levels
            self.files_paths_surface = glob.glob(file_path + "/single_????.h5") # indicates file paths for pressure levels
            self.n_years = len(self.files_paths_pressure)
        elif self.filetype == 'netcdf':
            self.files_paths_pressure = glob.glob(file_path + '/*_pl.nc')
            self.files_paths_surface = glob.glob(file_path + '/*_sfc.nc')
            self.n_years = len(self.files_paths_pressure)
        elif self.filetype == 'zarr':
            self.files_paths_pressure = []
            self.files_paths_surface = []
        else:
            raise ValueError("Input data must be in either the hdf5 or netcdf format")
        
        self.files_paths_pressure.sort()
        self.files_paths_surface.sort()
        
        assert len(self.files_paths_pressure) == len(self.files_paths_surface), "Number of years not identical in pressure vs. surface level data."
        
        if self.filetype == 'hdf5':
            with h5py.File(self.files_paths_pressure[0], 'r') as _f:
                logging.info("Getting file stats from {}".format(self.files_paths_pressure[0]))
                self.n_samples_per_year = _f['fields'].shape[0]
                #original image shape (before padding)
                self.img_shape_x = _f['fields'].shape[2]
                self.img_shape_y = _f['fields'].shape[3]
                self.n_in_channels = 13 
                self.n_samples_total = self.n_years * self.n_samples_per_year
            self.files_pressure = [None for _ in range(self.n_years)]
            self.files_surface  = [None for _ in range(self.n_years)]
        elif self.filetype == 'netcdf':
            ds = xr.open_dataset(self.files_paths_surface[0])
            #original image shape (before padding)
            self.img_shape_x = len(ds.lon) # 1440
            self.img_shape_y = len(ds.lat) # 721
            self.n_in_channels = len(ds.data_vars)
            n_points = 0
            self.cumulative_points_per_file = []
            self.points_per_file = []
            # Need to iterate through files to find the length because of leap years, incorrectly organized data, etc.
            for path in self.files_paths_surface:
                ds = xr.open_dataset(path)
                n_points = int(len(ds.time))
                self.points_per_file.append(n_points)
            self.cumulative_points_per_file = np.cumsum(np.array(self.points_per_file))
            self.n_samples_total = self.cumulative_points_per_file[-1]
            self.files_pressure = [None for _ in range(self.n_years)]
            self.files_surface  = [None for _ in range(self.n_years)]
        elif self.filetype == 'zarr':
            self.zarr_data = xr.open_dataset(self.file_path, engine='zarr')
            times = pd.to_datetime(self.zarr_data['time'].values)
            if mode == 'train' and lite: # training case, lite
                train_years = times[(times.year == 2017) & (times.month == 1) ] #USER: change to match the training and validation of your dataset
                self.zarr_data = self.zarr_data.sel(time=train_years)
            elif mode == 'validation':           # validation
                validation_years = times[(times.year == 2017) & (times.month == 2)]
                self.zarr_data = self.zarr_data.sel(time=validation_years)
            elif mode == 'testing':           # validation
                validation_years = times[(times.year == 2020) | (times.year == 2021)] 
                self.zarr_data = self.zarr_data.sel(time=validation_years)
            elif mode == 'train':
                train_years = times[(times.year < 2018) & (times.year > 1978)] # 1990
                self.zarr_data = self.zarr_data.sel(time=train_years)
            
            self.n_samples_total = len(self.zarr_data['time'])
            self.img_shape_x     = len(self.zarr_data['latitude'])
            self.img_shape_y     = len(self.zarr_data['longitude'])
            # N channels
            self.n_in_channels   = len(self.zarr_data.data_vars)
        else:
            raise ValueError("File type doesn't match one of hdf5, netcdf, or zarr")
        
    def _open_pressure_file(self, file_idx):
        """Open HDF5 or NetCDF pressure file."""
        if self.filetype == 'hdf5':
            _file = h5py.File(self.files_paths_pressure[file_idx], 'r')
            self.files_pressure[file_idx] = _file
        else:
            _file = xr.open_dataset(self.files_paths_pressure[file_idx])
            self.files_pressure[file_idx] = _file
            
    def _open_surface_file(self, file_idx):
        """Open HDF5 or NetCDF surface file."""
        if self.filetype == 'hdf5':
            _file = h5py.File(self.files_paths_surface[file_idx], 'r')
            self.files_surface[file_idx] = _file
        else:
            _file = xr.open_dataset(self.files_paths_surface[file_idx])
            self.files_surface[file_idx] = _file
      
    def __len__(self):
        """Return total number of samples."""
        return self.n_samples_total - self.forecast_length * self.dt // self.deltaTDivisor  # -1 to avoid last data point
    
    def __getitem__(self, global_idx, normalize=True, two_dimensional=False):
        """
        Return single input, output.

        global_idx: int
                global index of item
        normalize: bool
                flag for whether the data should be normalized
        two_dimensional: bool
                flag for whether the data should be processed for a 2D model.
                if not, 3D data is returned.
        """
        # TODO: not yet safety checked for edge cases or other errors
        if self.filetype == 'hdf5':
            year_idx  = int(global_idx/self.n_samples_per_year) #which year we are on
            local_idx = int(global_idx%self.n_samples_per_year) #which sample in that year we are on - determines indices for centering

            # open image file
            if self.files_pressure[year_idx] is None:
                self._open_pressure_file(year_idx)
        
            if self.files_surface[year_idx] is None:
                self._open_surface_file(year_idx)
            
            step = self.dt
            target_step = local_idx + step
            if target_step == self.n_samples_per_year:
                target_step = local_idx

            if normalize:
                t1 = torch.as_tensor((self.files_pressure[year_idx]['fields'][local_idx] - self.p_means['mean']) / self.p_means['std_dev'])
                t2 = torch.as_tensor((self.files_surface[year_idx]['fields'][local_idx] - self.s_means['mean']) / self.s_means['std_dev'])
                t3 = torch.as_tensor((self.files_pressure[year_idx]['fields'][target_step] - self.p_means['mean']) / self.p_means['std_dev'])
                t4 = torch.as_tensor((self.files_surface[year_idx]['fields'][target_step] - self.s_means['mean']) / self.s_means['std_dev'])
            else:
                t1 = torch.as_tensor(self.files_pressure[year_idx]['fields'][local_idx])
                t2 = torch.as_tensor(self.files_surface[year_idx]['fields'][local_idx])
                t3 = torch.as_tensor(self.files_pressure[year_idx]['fields'][target_step])
                t4 = torch.as_tensor(self.files_surface[year_idx]['fields'][target_step])
            
            t1, t2, t3, t4 = self._pad_data(t1, t2, t3, t4)
            return t1, t2, t3, t4
        
        elif self.filetype == 'netcdf': # dealing directly with netcdf files
            step = self.dt
            target_step = global_idx + step
            
            # Get file and local_idx for the input steps
            input_file_idx  = np.argmax(self.cumulative_points_per_file > global_idx) #which year we are on
            if input_file_idx > 0:
                input_local_idx = global_idx - self.cumulative_points_per_file[input_file_idx - 1] #which sample in that year we are on - determines indices for centering
            else:
                input_local_idx = global_idx
                # If we are on the last value, output_file_idx = 0 because self.cumulative_points_per_file is never > target_step
                if global_idx == self.n_samples_total:
                    output_file_idx =  -1 # last file value
                    output_local_idx = -2 # last value in file # will break if there is only 1 data point in the last file
            
            # Get file and local_idx for the output step
            output_file_idx  = np.argmax(self.cumulative_points_per_file > target_step) #which year we are on
            # case for which output_file_idx = last 
            if output_file_idx > 0:
                output_local_idx = target_step - self.cumulative_points_per_file[output_file_idx - 1] #which sample in that year we are on - determines indices for centering
            else: # = 0
                output_local_idx = target_step
                # If we are on the last value, output_file_idx = 0 because self.cumulative_points_per_file is never > target_step
                if target_step == self.n_samples_total:
                    output_file_idx =  -1 # last file value
                    output_local_idx = -1 # last value in file
            
            # open files
            if self.files_pressure[input_file_idx] is None:
                self._open_pressure_file(input_file_idx)
        
            if self.files_surface[input_file_idx] is None:
                self._open_surface_file(input_file_idx)
            
            if self.files_pressure[output_file_idx] is None:
                self._open_pressure_file(output_file_idx)

            if self.files_surface[output_file_idx] is None:
                self._open_surface_file(output_file_idx)
            
            if target_step >= self.n_samples_total: # If we are at the very last step
                target_step = input_local_idx

            # Isolate data from time point and convert to numpy array
            input_time_pressure = self.files_pressure[input_file_idx].isel(time=input_local_idx)
            input_time_surface  = self.files_surface[input_file_idx].isel(time=input_local_idx)

            input_time_pressure = np.stack([input_time_pressure['Z'].values, input_time_pressure['Q'].values, input_time_pressure['T'].values, input_time_pressure['U'].values, input_time_pressure['V'].values], axis=0)
            input_time_surface  = np.stack([input_time_surface['MSL'].values, input_time_surface['U10M'].values, input_time_surface['V10M'].values, input_time_surface['T2M'].values], axis=0)

            output_time_pressure = self.files_pressure[output_file_idx].isel(time=output_local_idx)
            output_time_surface  = self.files_surface[output_file_idx].isel(time=output_local_idx)

            output_time_pressure = np.stack([output_time_pressure['Z'].values,  output_time_pressure['Q'].values,   output_time_pressure['T'].values,   output_time_pressure['U'].values, output_time_pressure['V'].values], axis=0)
            output_time_surface  = np.stack([output_time_surface['MSL'].values, output_time_surface['U10M'].values, output_time_surface['V10M'].values, output_time_surface['T2M'].values], axis=0)
            
            if normalize:
                # p_ and s_means is a stack of the mean and standard deviation values
                t1 = torch.as_tensor((input_time_pressure - self.p_mean) / self.p_std)
                t2 = torch.as_tensor((input_time_surface - self.s_mean)  / self.s_std)
                t3 = torch.as_tensor((output_time_pressure - self.p_mean) / self.p_std)
                t4 = torch.as_tensor((output_time_surface - self.s_mean)  / self.s_std)
            else:
                t1 = torch.as_tensor(input_time_pressure)
                t2 = torch.as_tensor(input_time_surface)
                t3 = torch.as_tensor(output_time_pressure)
                t4 = torch.as_tensor(output_time_surface)
            
            t1, t2, t3, t4 = self._pad_data(t1, t2, t3, t4)
            return t1, t2, t3, t4
        
        elif self.filetype == 'zarr':
            output_pressure = []
            output_surface  = []
            step = self.dt
            # if forecast_length = 3, output_file_idx = global_idx + range(1, forecast_length+1)*step // 6
            output_file_idxs = global_idx + torch.arange(1, self.forecast_length+1)*step // self.deltaTDivisor 
            
            for output_idx in output_file_idxs:
                if output_idx >= self.__len__(): 
                    output_idx = self.__len__() - 1
            
            # Isolate data from time point and convert to numpy array
            # WeatherBench data stores from low --> high pressure levels
            # We convert to high --> low
            input_pressure = self.zarr_data.isel(time=global_idx, level=self.level_ordering)[['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']]
            input_surface  = self.zarr_data.isel(time=global_idx, level=self.level_ordering)[['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']]
            output_pressure_ds = self.zarr_data.isel(time=output_file_idxs, level=self.level_ordering)[['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']]
            output_surface_ds  = self.zarr_data.isel(time=output_file_idxs, level=self.level_ordering)[['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']]
            
            # Stack and convert to numpy array
            input_pressure = np.stack([input_pressure['geopotential'].values, input_pressure['specific_humidity'].values, input_pressure['temperature'].values, input_pressure['u_component_of_wind'].values, input_pressure['v_component_of_wind'].values], axis=0)
            input_surface  = np.stack([input_surface['mean_sea_level_pressure'].values, input_surface['10m_u_component_of_wind'].values, input_surface['10m_v_component_of_wind'].values, input_surface['2m_temperature'].values], axis=0)
            for i in range(self.forecast_length):
                pressure_ds = output_pressure_ds.isel(time=i)
                surface_ds  = output_surface_ds.isel(time=i)
                output_pressure.append(np.stack([pressure_ds['geopotential'].values, pressure_ds['specific_humidity'].values, pressure_ds['temperature'].values, pressure_ds['u_component_of_wind'].values, pressure_ds['v_component_of_wind'].values], axis=0))
                output_surface.append(np.stack([surface_ds['mean_sea_level_pressure'].values, surface_ds['10m_u_component_of_wind'].values, surface_ds['10m_v_component_of_wind'].values, surface_ds['2m_temperature'].values], axis=0))
                output_pressure[i] = torch.as_tensor((output_pressure[i] - self.p_mean) / self.p_std)
                output_surface[i]  = torch.as_tensor((output_surface[i] - self.s_mean) / self.s_std)
                output_pressure[i], output_surface[i] = self._pad_data(output_pressure[i], output_surface[i])

            # p_ and s_means is a stack of the mean and standard deviation values
            input_pressure = torch.as_tensor((input_pressure - self.p_mean) / self.p_std)
            input_surface = torch.as_tensor((input_surface - self.s_mean)  / self.s_std)
            
            input_pressure, input_surface = self._pad_data(input_pressure, input_surface)

        
            return input_pressure, input_surface, output_pressure, output_surface
        
    def _pad_data(self, t1, t2):
        """
        Perform padding for outermost patching step.

        t1: Tensor
                pressure-level tensors
        t2: Tensor
                surface-level tensors
        """
        # perform padding for patch embedding step
        input_shape = t1.shape  # shape is (5 variables x 13 pressure levels x 721 latitude x 1440 longitude)
        
        x1_pad    = (self.patch_size[0] - (input_shape[1] % self.patch_size[0])) % self.patch_size[0] // 2
        x2_pad    = (self.patch_size[0] - (input_shape[1] % self.patch_size[0])) % self.patch_size[0] - x1_pad
        y1_pad    = (self.patch_size[1] - (input_shape[2] % self.patch_size[1])) % self.patch_size[1] // 2
        y2_pad    = (self.patch_size[1] - (input_shape[2] % self.patch_size[1])) % self.patch_size[1] - y1_pad
        z1_pad    = (self.patch_size[2] - (input_shape[3] % self.patch_size[2])) % self.patch_size[2] // 2
        z2_pad    = (self.patch_size[2] - (input_shape[3] % self.patch_size[2])) % self.patch_size[2] - z1_pad

        # pad pressure fields input and output
        t1 = torch.nn.functional.pad(t1, pad=(z1_pad, z2_pad, y1_pad, y2_pad, x1_pad, x2_pad), mode='constant', value=0)
        
        # pad 
        t2  = torch.nn.functional.pad(t2, pad=(z1_pad, z2_pad, y1_pad, y2_pad), mode='constant', value=0)

        return t1, t2
    
class Pangu2DDataset(PanguDataset):
    """Dataloader for 2D model."""

    def __init__(self, params, file_path, mode, patch_size, forecast_length=1):
        super().__init__(params, file_path, mode, patch_size, forecast_length=1)
        self.patch_size = self.patch_size[-2:]
        self.p_mean = np.load(params['pressure_static_data_path'])[0].reshape(5, 13, 1, 1)
        self.p_std  = np.load(params['pressure_static_data_path'])[1].reshape(5, 13, 1, 1)
        self.s_mean = np.load(params['surface_static_data_path'])[0].reshape(4, 1, 1)
        self.s_std  = np.load(params['surface_static_data_path'])[1].reshape(4, 1, 1)
        
    def __getitem__(self, global_idx, normalize=True, two_dimensional=False):
        """
        Return an input output pair.

        global_idx: int
                global index of item
        normalize: bool
                flag for whether the data should be normalized
        two_dimensional: bool
                flag for whether the data should be processed for a 2D model.
                if not, 3D data is returned.
        """
        # TODO: not yet safety checked for edge cases or other errors
        output_pressure = []
        output_surface  = []
        step = self.dt
        # if forecast_length = 3, output_file_idx = global_idx + range(1, forecast_length+1)*step // 6
        output_file_idxs = global_idx + torch.arange(1, self.forecast_length+1)*step // self.deltaTDivisor 
            
        for output_idx in output_file_idxs:
            if output_idx >= self.__len__(): 
                output_idx = self.__len__() - 1
            
        # Isolate data from time point and convert to numpy array
        # WeatherBench data stores from low --> high pressure levels
        # We convert to high --> low
        input_pressure = self.zarr_data.isel(time=global_idx, level=self.level_ordering)[['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']]
        input_surface  = self.zarr_data.isel(time=global_idx, level=self.level_ordering)[['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']]
        output_pressure_ds = self.zarr_data.isel(time=output_file_idxs, level=self.level_ordering)[['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']]
        output_surface_ds  = self.zarr_data.isel(time=output_file_idxs, level=self.level_ordering)[['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']]
            
        # Stack and convert to numpy array
        input_pressure = np.stack([input_pressure['geopotential'].values, input_pressure['specific_humidity'].values, input_pressure['temperature'].values, input_pressure['u_component_of_wind'].values, input_pressure['v_component_of_wind'].values], axis=0)
        input_surface  = np.stack([input_surface['mean_sea_level_pressure'].values, input_surface['10m_u_component_of_wind'].values, input_surface['10m_v_component_of_wind'].values, input_surface['2m_temperature'].values], axis=0)
        for i in range(self.forecast_length):
            pressure_ds = output_pressure_ds.isel(time=i)
            surface_ds  = output_surface_ds.isel(time=i)
            output_pressure.append(np.stack([pressure_ds['geopotential'].values, pressure_ds['specific_humidity'].values, pressure_ds['temperature'].values, pressure_ds['u_component_of_wind'].values, pressure_ds['v_component_of_wind'].values], axis=0))
            output_surface.append(np.stack([surface_ds['mean_sea_level_pressure'].values, surface_ds['10m_u_component_of_wind'].values, surface_ds['10m_v_component_of_wind'].values, surface_ds['2m_temperature'].values], axis=0))

            if normalize:
                output_pressure[i] = torch.as_tensor((output_pressure[i] - self.p_mean) / self.p_std)
                output_surface[i]  = torch.as_tensor((output_surface[i] - self.s_mean) / self.s_std)
            else:
                output_pressure[i] = torch.as_tensor(output_pressure[i])
                output_surface[i]  = torch.as_tensor(output_surface[i])
            output_pressure[i], output_surface[i] = self._pad_data(output_pressure[i], output_surface[i])

        # p_ and s_means is a stack of the mean and standard deviation values
        if normalize:
            input_pressure = torch.as_tensor((input_pressure - self.p_mean) / self.p_std)
            input_surface = torch.as_tensor((input_surface - self.s_mean)  / self.s_std)
        else:
            input_pressure = torch.as_tensor(input_pressure)
            input_surface  = torch.as_tensor(input_surface)

        # shape of input_pressure
        # (5, 13, 721, 1440)
        input_pressure, input_surface = self._pad_data(input_pressure, input_surface)
        output_pressure[0], output_surface[0] = self._pad_data(output_pressure[0], output_surface[0])
        # Reshape into 2D
        shape_pressure  = input_pressure.shape
        input_pressure  = input_pressure.reshape(-1, shape_pressure[2], shape_pressure[3])
        output_pressure[0] = output_pressure[0].reshape( -1, shape_pressure[2], shape_pressure[3])
        
        return input_pressure, input_surface, output_pressure, output_surface
        
    def _pad_data(self, t1, t2):
        # perform padding for patch embedding step
        input_shape = t1.shape  # shape is (5 variables x 13 pressure levels x 721 latitude x 1440 longitude)
        
        y1_pad    = (self.patch_size[0] - (input_shape[2] % self.patch_size[0])) % self.patch_size[0] // 2
        y2_pad    = (self.patch_size[0] - (input_shape[2] % self.patch_size[0])) % self.patch_size[0] - y1_pad
        z1_pad    = (self.patch_size[1] - (input_shape[3] % self.patch_size[1])) % self.patch_size[1] // 2
        z2_pad    = (self.patch_size[1] - (input_shape[3] % self.patch_size[1])) % self.patch_size[1] - z1_pad

        # pad pressure fields input and output
        t1 = torch.nn.functional.pad(t1, pad=(z1_pad, z2_pad, y1_pad, y2_pad), mode='constant', value=0)
        
        # pad 
        t2 = torch.nn.functional.pad(t2, pad=(z1_pad, z2_pad, y1_pad, y2_pad), mode='constant', value=0)

        return t1, t2
    
