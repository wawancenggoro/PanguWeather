from networks.pangu import PanguModel as PanguModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
    input_upper: Tensor of shape 
      (n_batch,  n_fields, n_vert, n_lat, n_lon)
      (n_batch,  5, 13, 721, 1440) based on the number in the paper
    input_surface: Tensor of shape 
      (n_batch, 4, 721, 1440) based on the number in the paper
"""
num_samples = 1
input_upper = torch.randn((num_samples, 5, 13, 721, 1440)).to(torch.float32).to(device)
input_surface = torch.randn((num_samples, 4, 721, 1440)).to(torch.float32).to(device)

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
