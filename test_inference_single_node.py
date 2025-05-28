from networks.pangu import PanguModel as PanguModel
import torch
from utils.data_loader_multifiles import get_data_loader_synthetic
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# params = {'dt': 24, 'data_distributed': True, 'num_epochs': 5, 'C': 192, 'subset_size': 16, 'validation_subset_size': 6, 'restart': False, 'hash': '20250528_111563285', 'Lite': False, 'daily': False, 'save_counter': 51, 'model': 'pangu', 'save_dir': {'pangu': 'trained_models/test/panguMoreData/20250528_111563285/', 'panguLite': 'trained_models/test/panguLite/20250528_111563285/', 'relativeBias': 'trained_models/test/relativeBiasLite/20250528_111563285/', 'noBiasLite': 'trained_models/test/noBiasLite/20250528_111563285/', '2D': 'trained_models/test/twoDimensionalLite/20250528_111563285/', 'threeLayer': 'trained_models/test/threeLayerLite/20250528_111563285/', 'positionEmbedding': 'trained_models/test/posEmbeddingLite/20250528_111563285/', '2Dim192': 'trained_models/test/twoDim192Lite/20250528_111563285/', '2DPosEmb': 'trained_models/test/twoDimPosEmb/20250528_111563285/', '2DPosEmbLite': 'trained_models/test/twoDimPosEmbLite/20250528_111563285/'}, 'patch_size': (2, 4, 4), 'batch_size': 2, 'lat_crop': (1, 2), 'lon_crop': (0, 0)}
# train_data_loader = get_data_loader_synthetic(params)
# for i, data in enumerate(train_data_loader):
#   input, input_surface, target, target_surface = data[0], data[1], data[2][0], data[3][0]
#   breakpoint()

"""
    input_upper: Tensor of shape 
      (n_batch,  n_fields, n_vert, n_lat, n_lon)
      (n_batch,  5, 13, 721, 1440) based on the number in the paper
    input_surface: Tensor of shape 
      (n_batch, 4, 721, 1440) based on the number in the paper
"""
num_samples = 1

input_upper = np.random.rand(num_samples, 5, 13, 721, 1440).astype(np.float32)
input_surface = np.random.rand(num_samples, 4, 721, 1440).astype(np.float32)

input_upper = torch.tensor(input_upper).to(device)
input_upper = torch.tensor(input_upper).to(device)

# input_upper = torch.randn((num_samples, 5, 13, 721, 1440)).to(torch.float32).to(device)
# input_surface = torch.randn((num_samples, 4, 721, 1440)).to(torch.float32).to(device)

patch_size = (2, 4, 4)

def pad_data(t1, t2):
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

input_upper, input_surface = pad_data(input_upper, input_surface)

model = PanguModel(device=device, dim=192, patch_size=patch_size).to(device)

output_upper, output_surface = model(input_upper, input_surface)
print(output_upper.shape)
print(output_surface.shape)
