from diffusion.diff_wrapper import DynaSTI_NASCE
from utils.utils import train
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import matplotlib
from datasets.dataset_nasce import get_dataloader
from json import JSONEncoder
import json
import sys
from utils.utils import evaluate_imputation_all, get_num_params
from models.ema import EMA
from utils.ignnk_util import train_ignnk
from models.ignnk import IGNNK
import pandas as pd
from tqdm import tqdm

matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
# torch.manual_seed(42)
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(precision=10)
plt.rcParams.update({'font.size': 22})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"torch cuda: {torch.cuda.is_available()}")
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

data = 'temps'
is_neighbor = False
is_separate = True
is_multi = True
n_steps = 30
n_features = 2 if data == 'temps' else 775
# 3 -> None, 6 -> add-delta, 3 -> sole-delta
if is_neighbor:
    n_spatial = 50 # 50
else:
    n_spatial = 143
total_stations = 179
spatial_choice = None # 'sole-delta'
spatial_context_dim = 6 if spatial_choice == 'add-delta' else 3
miss_type = 'random'
seed = 400 # np.random.randint(10,100)
simple = False

is_ema = True

# filename: Any, is_year: bool = True, n_steps: int = 366
data_type = 'year'
dataset_name = 'nasce'
data_file_train = f'./data/nacse/X_OR_{data}_train.npy'
data_file_train_loc = f'./data/nacse/X_OR_{data}_loc.npy'
mean_std_file = f'./data/nacse/X_OR_{data}'
data_file_test = f'./data/nacse/X_OR_{data}_test.npy'
data_file_test_loc = f'./data/nacse/X_OR_{data}_loc.npy'
nsample = 50
print("################### Start ###################")
 #352 #len(given_features)

train_loader, test_loader = get_dataloader(total_stations, mean_std_file, n_features, batch_size=8, missing_ratio=0.02, type=data_type, data=data, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, is_multi=is_multi)

print(f"################### Data loading done ###################")

config_file = 'configs/config_nacse.json'

try:
    with open(config_file, 'r') as f:
        config = json.load(f)
except Exception as e:
    print(f"Error reading the configuration file '{config_file}': {e}")
    sys.exit(1)

model_diff_saits = DynaSTI_NASCE(config, device, n_spatial=n_spatial).to(device)

filename = f"model_dynasti_nacse_multi.pth"
print(f"\nDynaSTI training starts.....\n")
model_folder = 'saved_models_nacse'

model_diff_saits.load_state_dict(torch.load(f"{model_folder}/{filename}"))
#

if not os.path.isdir(model_folder):
    os.makedirs(model_folder)

# train(
#     model_diff_saits,
#     config["train"],
#     train_loader,
#     valid_loader=test_loader,
#     foldername=model_folder,
#     filename=f"{filename}",
#     is_dit=config['is_dit_ca2'],
#     d_spatial=config['model']['d_spatial'],
#     d_time=config['model']['d_time'],
#     is_spat=False,
#     is_ema=is_ema,
#     name=f"nacse_multi_1"
# )

# print(f"DynaSTI params: {get_num_params(model_diff_saits)}")
# Create EMA handler with the main model
# ema = EMA(model_diff_saits)

# # Define the file path where the EMA model is saved
# ema_model_filepath = f"{model_folder}/ema_model_nacse_multi.pth"

# # Load the saved EMA model
# ema.load(ema_model_filepath)
# model_diff_saits = ema.ema_model

models = {
    'SPAT-SADI': model_diff_saits,
    # 'SPAT-SADI': model_diff_saits_fft,
    # 'IGNNK': model_ignnk,
    # # 'GP': None,
    # 'DK': dk_model,
    # 'MEAN': None,
    # 'PriSTI': model_pristi
}
mse_folder = f"results_nacse/metric"
data_folder = f"results_nacse/data"
filename = (data_file_test, data_file_test_loc, mean_std_file)
# evaluate_imputation_all(models=models, trials=3, mse_folder=mse_folder, n_features=n_features, dataset_name='nasce', batch_size=16, filename=filename, spatial=True, simple=simple, unnormalize=False, n_stations=n_spatial, n_steps=n_steps, total_locations=total_stations, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, data=False, missing_dims=10 if config['is_multi'] else -1, is_multi=is_multi) #, latent_size=(latent_seq_dim, 2, n_iters, lr, random))




def generate_uniform_points_around_targets(targets: torch.Tensor, N: int, expand_ratio=0.1):
    """
    Generate N new 3D coordinates uniformly distributed in and around the bounding box 
    of the given target points.

    Args:
        targets (torch.Tensor): Shape (M, 3), existing coordinates.
        N (int): Number of new points to generate.
        expand_ratio (float): How much to expand the bounding box beyond the min/max.
                             e.g., 0.1 means extend by 10% on each side.

    Returns:
        new_points (torch.Tensor): Shape (N, 3), uniformly distributed points.
    """

    # Compute bounding box
    print(f"targets: {targets.shape}")
    coord_min = targets.min(dim=0).values
    coord_max = targets.max(dim=0).values

    # Expand the bounding box
    box_size = coord_max - coord_min
    expand = expand_ratio * box_size
    coord_min = coord_min - expand
    coord_max = coord_max + expand

    # Sample uniformly
    print(f"coord max: {coord_max}, coord min: {coord_min}")
    new_points = torch.rand((N, 3)) * (coord_max - coord_min) + coord_min

    return new_points

def compute_global_uncertainty_mean(samples, eps=1e-8):
    """
    Computes a single scalar uncertainty score from MC samples.
    
    Args:
        samples: Tensor of shape (T, N_sensors, L, K)
                 T = MC samples
                 N_sensors = number of virtual sensors
                 L = time steps
                 K = features/channels

    Returns:
        global_entropy_mean: scalar, average joint entropy over time & sensors
    """
    T, N_sensors, L, K = samples.shape

    # Center samples across T
    mean = samples.mean(dim=0, keepdim=True)           # (1, N_sensors, L, K)
    centered = samples - mean                          # (T, N_sensors, L, K)

    # Reshape to (N_sensors*L, T, K) for covariance computation
    reshaped = centered.permute(1, 2, 0, 3).reshape(-1, T, K)  # (N_sensors*L, T, K)

    # Covariance per sensor & time: Σ = (1/T) X^T X
    cov = torch.matmul(reshaped.transpose(1, 2), reshaped) / (T + eps)  # (N_sensors*L, K, K)

    # Add small ε*I for numerical stability
    identity = eps * torch.eye(K, device=samples.device).unsqueeze(0)    # (1, K, K)
    cov_stable = cov + identity

    # Determinant of covariance matrices
    det = torch.linalg.det(cov_stable)  # (N_sensors*L,)

    # Joint entropy at each sensor & timestep
    joint_entropy = 0.5 * torch.log(((2 * torch.pi * torch.exp(torch.tensor(1.0)))**K) * det)  # (N_sensors*L,)

    # Reshape back to (N_sensors, L)
    joint_entropy = joint_entropy.view(N_sensors, L)

    # Mean over time and sensors → single scalar
    global_entropy_mean = joint_entropy.mean()

    return global_entropy_mean

class NewLocationCoordsAndUncertainty:
    def __init__(self, coords, uncertainty):
        self.coords = coords  # Tensor of shape (N, 3)
        self.uncertainty = uncertainty  # Scalar uncertainty score
    
    def __lt__(self, other):
        return self.uncertainty < other.uncertainty

    def __repr__(self):
        return f"Coords: {self.coords.numpy()}, Uncertainty: {self.uncertainty}"

train_loader, test_loader = get_dataloader(total_stations, mean_std_file, n_features, batch_size=8, missing_ratio=0.02, type=data_type, data=data, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, is_multi=is_multi, is_test=True, southeast=True)

N = 5
M = 10
lr = 0.01
total_points = 100
iters = 10 #00
folder = 'results_map_random'
if not os.path.isdir(folder):
    os.makedirs(folder)

model_diff_saits.eval()
# model_diff_saits.requires_grad_(True)
# for p in model_diff_saits.parameters():
#     p.requires_grad_(False)
#     print(f"p: {p.requires_grad}")

for i, test_batch in enumerate(test_loader):
    input_locations = test_batch['spatial_info'][0]
    missing_locations = test_batch['missing_data_loc'][0]
    df_targets = pd.DataFrame(missing_locations, columns=['longitude', 'latitude', 'elevation'])

    if not os.path.isdir(f"{folder}/{i}"):
        os.makedirs(f"{folder}/{i}")

    df_targets.to_csv(f'{folder}/{i}/target_locations.csv', index=False)
    new_locations = generate_uniform_points_around_targets(missing_locations, total_points)
    # new_locations = torch.tensor(pd.read_csv(f'{folder}/{i}/decided_locations.csv').to_numpy(), dtype=torch.float32)

    df_inputs = pd.DataFrame(input_locations, columns=['longitude', 'latitude', 'elevation'])
    df_inputs.to_csv(f'{folder}/{i}/input_locations.csv', index=False)


    df_targets = pd.DataFrame(new_locations, columns=['longitude', 'latitude', 'elevation'])

    df_targets.to_csv(f'{folder}/{i}/random_locations.csv', index=False)
    
    decided_locations = []
    for k in range(N):
        print(f"Missing sensor {k+1}/{N} processing...")
        locations_and_uncertainty = []
        if len(decided_locations) > 0:
            decided_coords = torch.stack([loc_unc.coords for loc_unc in decided_locations], dim=0)
            test_batch_copy = test_batch.copy()
            # test_batch_copy['spatial_info'] = torch.cat([test_batch_copy['spatial_info'], decided_coords], dim=0)
            test_batch_copy['missing_data_loc'] = decided_coords.reshape((1, -1, 3))
            test_batch_copy['missing_data'] = None
            test_batch_copy['gt_mask'] = torch.zeros((1, n_steps, decided_coords.shape[0], 2))
            test_batch_copy['missing_data_mask'] = torch.ones((1, n_steps, decided_coords.shape[0], 2))
            with torch.no_grad():
                outputs_test_copy = model_diff_saits.evaluate(test_batch_copy, nsample, missing_dims=decided_coords.shape[0])
                samples_test_copy, _, _, _, _, _, _, _, _, _ = outputs_test_copy
                samples_test_copy = samples_test_copy.permute(0, 1, 3, 2)
                samples_test_copy_mean = samples_test_copy.mean(dim=1)  # (B,L,N*K)
            samples_test_copy_mean = samples_test_copy_mean.reshape(1, samples_test_copy_mean.shape[1], decided_coords.shape[0], 2)
            
            test_batch_copy = test_batch.copy()
            test_batch_copy['observed_data'] = torch.cat([test_batch_copy['observed_data'], samples_test_copy_mean.cpu()], dim=-2) #.detach()
            new_obs_mask = torch.ones((1, n_steps, decided_coords.shape[0], 2))
            test_batch_copy['observed_mask'] = torch.cat([test_batch_copy['observed_mask'], new_obs_mask], dim=-2) #.detach()
            test_batch_copy['spatial_info'] = torch.cat([test_batch_copy['spatial_info'], decided_coords.unsqueeze(0)], dim=1)
        else:
            test_batch_copy = test_batch
        with tqdm(range(new_locations.shape[0]), desc=f"Processing test batch {i+1}/{len(test_loader)}") as pbar:
            for j in pbar:
                init_test_batch = test_batch_copy.copy()
                init_test_batch['missing_data_loc'] = new_locations[j].reshape(1, -1, 3)
                init_test_batch['missing_data'] = None
                init_test_batch['gt_mask'] = torch.zeros((1, n_steps, 1, 2))
                init_test_batch['missing_data_mask'] = torch.ones((1, n_steps, 1, 2))
                with torch.no_grad():
                    outputs_init = model_diff_saits.evaluate(init_test_batch, nsample, missing_dims=1)
                    samples_init, _, _, _, _, _, _, _, _, _ = outputs_init
                    samples_init = samples_init.permute(0, 1, 3, 2)
                    samples_init_mean = samples_init.mean(dim=1)  # (B,L,N*K)

                samples_init_mean = samples_init_mean.reshape(1, samples_init_mean.shape[1], 1, 2)
                
                temp_test_batch = test_batch_copy.copy()

                temp_test_batch['observed_data'] = torch.cat([temp_test_batch['observed_data'].to(device), samples_init_mean], dim=-2) #.detach()
                new_obs_mask = torch.ones((1, n_steps, 1, 2)).to(device)
                temp_test_batch['observed_mask'] = torch.cat([temp_test_batch['observed_mask'].to(device), new_obs_mask], dim=-2) #.detach()
                
                temp_test_batch['spatial_info'] = torch.cat([temp_test_batch['spatial_info'].to(device), new_locations[j].reshape((1, -1, 3)).to(device)], dim=1)
                temp_test_batch['missing_data_loc'] = temp_test_batch['missing_data_loc'].to(device) #.detach()

                with torch.no_grad():
                    outputs = model_diff_saits.evaluate(temp_test_batch, nsample, missing_dims=M)
                    samples, _, _, _, _, _, _, _, _, _ = outputs
                    samples = samples.permute(0, 1, 3, 2)  # (B, T, N_sensors, L, K)
                B, T, L, D = samples.shape
                # print(f"sample temp: {samples_temp.shape}")
                samples = samples.reshape(T, L, M, 2).permute(0, 2, 1, 3) # T, M, L, K
                uncertainty_score = compute_global_uncertainty_mean(samples, eps=1e-8)

                loc_and_uncertainty = NewLocationCoordsAndUncertainty(new_locations[j], uncertainty_score.item())
                locations_and_uncertainty.append(loc_and_uncertainty)
                pbar.set_postfix({"Current location": j, "Uncertainty": uncertainty_score.item()})
                print(f"Uncertainty for location {j+1}/{total_points}: {uncertainty_score.item()}")
                print(f"Location {j+1}/{total_points} processed.")
        locations_and_uncertainty.sort()
        
        new_decided_locations_uncertainty = [np.concatenate([loc_unc.coords.numpy(),np.array([loc_unc.uncertainty])], axis=0) for loc_unc in locations_and_uncertainty]

        if not os.path.isdir(f'{folder}/{i}/{k}'):
            os.makedirs(f'{folder}/{i}/{k}')
        df_new_locations = pd.DataFrame(new_decided_locations_uncertainty, columns=['longitude', 'latitude', 'elevation', 'uncertainty'])
        df_new_locations.to_csv(f'{folder}/{i}/{k}/decided_locations_uncertainty.csv', index=False)

        decided_locations.append(locations_and_uncertainty[0])
        selected_coords = locations_and_uncertainty[0].coords
        new_locations_copy = []
        for loc in new_locations:
            if not torch.all(loc == selected_coords):
                new_locations_copy.append(loc)
        new_locations = torch.stack(new_locations_copy, dim=0)
        print(f"Decided location {k+1}/{N}: {locations_and_uncertainty[0].coords.numpy()} with uncertainty {locations_and_uncertainty[0].uncertainty}")
    new_decided_locations = [np.concatenate([loc_unc.coords.numpy(), np.array([loc_unc.uncertainty])], axis=0) for loc_unc in decided_locations]
    df_new_locations = pd.DataFrame(new_decided_locations, columns=['longitude', 'latitude', 'elevation', 'uncertainty'])
    df_new_locations.to_csv(f'{folder}/{i}/decided_locations.csv', index=False)
    print(f"Decided locations for test batch {i+1}/{len(test_loader)} saved.")
    exit()