import numpy as np
import os

directory = 'synthetic_data'

if not os.path.exists(directory):
    os.makedirs(directory)

num_samples = {
    'train': 192,
    'valid': 48
}

for dsname in ['train', 'valid']:
    if not os.path.exists(os.path.join(directory, dsname)):
        os.makedirs(os.path.join(directory, dsname))

    print(f'Generate input upper {dsname}...')
    with open(os.path.join(directory, dsname,'input_upper.npy'), 'wb') as f:
        input_upper = np.random.rand(num_samples[dsname], 5, 13, 721, 1440).astype(np.float32)
        print(input_upper.shape)
        np.save(f, input_upper)

    print(f'Generate input surface {dsname}...')
    with open(os.path.join(directory, dsname,'input_surface.npy'), 'wb') as f:
        input_surface = np.random.rand(num_samples[dsname], 4, 721, 1440).astype(np.float32)
        print(input_surface.shape)
        np.save(f, input_surface)

    print(f'Generate target upper {dsname}...')
    with open(os.path.join(directory, dsname,'target_upper.npy'), 'wb') as f:
        target_upper = np.random.rand(num_samples[dsname], 5, 14, 724, 1440).astype(np.float32)
        print(target_upper.shape)
        np.save(f, target_upper)

    print(f'Generate target surface {dsname}...')
    with open(os.path.join(directory, dsname,'target_surface.npy'), 'wb') as f:
        target_surface = np.random.rand(num_samples[dsname], 4, 724, 1440).astype(np.float32)
        print(target_surface.shape)
        np.save(f, target_surface)