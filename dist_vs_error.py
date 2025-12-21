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
import math
import random

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
is_multi = False
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

filename = f"model_dynasti_nacse_sparse.pth"
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
ema = EMA(model_diff_saits)

# Define the file path where the EMA model is saved
ema_model_filepath = f"{model_folder}/ema_model_nacse.pth"

# Load the saved EMA model
ema.load(ema_model_filepath)
model_diff_saits = ema.ema_model

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


def sample_location(lat, lon, radius_m, fix_lat=False, fix_lon=False, direction="any"):
    """
    Sample a new location around (lat, lon) within a given radius, with options
    to fix latitude or longitude and to force a sampling direction.

    Parameters
    ----------
    lat, lon : float
        Original coordinates in degrees.
    radius_m : float
        Sampling radius in meters.
    fix_lat : bool
        If True, keep latitude fixed.
    fix_lon : bool
        If True, keep longitude fixed.
    direction : str
        One of {"north", "south", "east", "west", "any"}.

    Returns
    -------
    (lat_new, lon_new) : tuple
        Sampled coordinates in degrees.
    """

    # Earth radius in meters
    R = 6371000

    # Convert to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # Random distance within radius
    distance = radius_m # random.random() * radius_m

    # Map direction to angle
    dir_map = {
        "east": 0,
        "north": math.pi / 2,
        "west": math.pi,
        "south": 3 * math.pi / 2
    }

    if direction in dir_map:
        # print(f"Sampling direction: {dir_map[direction]}")
        angle = dir_map[direction]
    else:
        angle = random.random() * 2 * math.pi

    # -------------------------
    # Case 1: Fix latitude only
    # -------------------------
    if fix_lat and not fix_lon:
        if direction in ["north", "south"]:
            raise ValueError("Cannot move north/south when fix_lat=True.")
        lon_new = lon + math.degrees(distance * math.cos(angle) /
                                     (R * math.cos(lat_rad)))
        # print(f"rad: {math.degrees(distance * math.cos(angle) / (R * math.cos(lat_rad)))}")
        # print(f"lon_new: {lon_new}")
        return lat, lon_new

    # -------------------------
    # Case 2: Fix longitude only
    # -------------------------
    if fix_lon and not fix_lat:
        if direction in ["east", "west"]:
            raise ValueError("Cannot move east/west when fix_lon=True.")
        lat_new = lat + math.degrees(distance * math.sin(angle) / R)
        return lat_new, lon

    # -------------------------
    # Case 3: Fixing both is invalid
    # -------------------------
    if fix_lat and fix_lon:
        raise ValueError("Cannot fix both latitude and longitude.")

    # -------------------------
    # Case 4: Standard spherical sampling
    # -------------------------
    lat_new = math.asin(
        math.sin(lat_rad) * math.cos(distance / R) +
        math.cos(lat_rad) * math.sin(distance / R) * math.sin(angle)
    )

    lon_new = lon_rad + math.atan2(
        math.cos(angle) * math.sin(distance / R) * math.cos(lat_rad),
        math.cos(distance / R) - math.sin(lat_rad) * math.sin(lat_new)
    )

    return math.degrees(lat_new), math.degrees(lon_new)

def haversine(lat1, lon1, lat2, lon2):
    """
    Compute distance in meters between two lat/lon points.
    """
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (math.sin(dphi/2)**2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2)

    return 2 * R * math.asin(math.sqrt(a))


def inverse_distance_sum(point, stations, eps=1e-6):
    """
    Compute sum(1/distance) for a point to all input stations.
    """
    lat, lon = point
    total = 0
    for s_lat, s_lon in stations:
        d = haversine(lat, lon, s_lat, s_lon)
        total += 1.0 / max(d, eps)
    return total


def sample_new_point(center, radius_m):
    """
    Sample random point within a given radius.
    """
    lat0, lon0 = center
    R = 6371000

    dist = random.random() * radius_m
    angle = random.random() * 2 * math.pi

    lat_new = lat0 + (dist * math.sin(angle)) * 1.0 / R * 180 / math.pi
    lon_new = lon0 + (dist * math.cos(angle)) * 1.0 / (R * math.cos(math.radians(lat0))) * 180 / math.pi

    return lat_new, lon_new



N = 3
M = 1 # Number of virtual sensors to evaluate uncertainty on
old = False # True

test_coords = [[-121.1333333, 45.45, 406.0]] # [[-123.6833333, 44.9166667, 1095.0]] # [[-123.5833333, 44.9166667, 1095.0]] #[[-122.5063889, 45.1783333, 197.0]] # # [[-121.1333333, 44.6333333, 702.0]] #   # [[-121.06508, 44.86472, 426.0]]
radius_range = (10000, 60000)  # 10 km to 60 km
quantity = 11
train_loader, test_loader = get_dataloader(total_stations, mean_std_file, n_features, batch_size=8, missing_ratio=0.02, type=data_type, data=data, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, is_multi=is_multi, is_test=True, southeast=False, sparse=False, missing_dims=M, parts=False, test_loc=test_coords, exclude_train_coords=None, old=old) #, radius_range=radius_range, quantity=quantity)

model_diff_saits.eval()
folder = "input_locations_distance_left_1"
if not os.path.isdir(folder):
    os.makedirs(folder)
with torch.no_grad():
    total_batch = 0
    avg_rmse = 0.0
    for i, test_batch in enumerate(test_loader):
        if i <= 7: #== 0:
            input_locations = test_batch['spatial_info'][0]
            df_inputs = pd.DataFrame(input_locations, columns=['longitude', 'latitude', 'elevation'])
            df_inputs.to_csv(f'{folder}/input_locations_10_60.csv', index=False)
            continue
        missing_data_mask = test_batch['missing_data_mask'].squeeze(2).to(device)
        missing_data = test_batch['missing_data'].squeeze(2).to(device) if test_batch['missing_data'] is not None else None
        input_locations = test_batch['spatial_info']
        missing_locations = test_batch['missing_data_loc']
        outputs = model_diff_saits.evaluate(test_batch, nsample, missing_dims=M)
        samples, _, _, _, _, _, _, _, attn_mean, attn_std = outputs
        samples = samples.permute(0, 1, 3, 2)
        sample_mean = samples.mean(dim=1)
        print(f"attn_mean shape: {attn_mean.shape}, spatial_info shape: {test_batch['spatial_info'].shape}")
        spatial_positions = test_batch['spatial_info'].squeeze(0)
        attn_mean = attn_mean.squeeze(0).unsqueeze(-1).cpu().numpy()
        df_array = np.concatenate([spatial_positions, attn_mean], axis=1)
        df_spat_attn = pd.DataFrame(df_array, columns=['longitude', 'latitude', 'elevation', 'attn'])
        df_spat_attn.to_csv(f"{folder}/attn_map.csv")

        rmse = ((sample_mean - missing_data) * missing_data_mask) ** 2
        rmse = rmse.sum().item() / missing_data_mask.sum().item()
        avg_rmse += math.sqrt(rmse)
        total_batch += 1
        break
    print(f"Radius range: {radius_range}, Test RMSE: {avg_rmse/total_batch}")

exit()


radius_range = (60000, 110000)  # 60 km to 110 km

train_loader, test_loader = get_dataloader(total_stations, mean_std_file, n_features, batch_size=8, missing_ratio=0.02, type=data_type, data=data, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, is_multi=is_multi, is_test=True, southeast=False, sparse=False, missing_dims=M, parts=False, test_loc=test_coords, exclude_train_coords=None, old=old, radius_range=radius_range, quantity=quantity)

with torch.no_grad():
    total_batch = 0
    avg_rmse = 0.0
    for i, test_batch in enumerate(test_loader):
        if i == 0:
            input_locations = test_batch['spatial_info'][0]
            df_inputs = pd.DataFrame(input_locations, columns=['longitude', 'latitude', 'elevation'])
            df_inputs.to_csv(f'{folder}/input_locations_60_110.csv', index=False)
        missing_data_mask = test_batch['missing_data_mask'].squeeze(2).to(device)
        missing_data = test_batch['missing_data'].squeeze(2).to(device)
        input_locations = test_batch['spatial_info']
        missing_locations = test_batch['missing_data_loc']
        outputs = model_diff_saits.evaluate(test_batch, nsample, missing_dims=M)
        samples, _, _, _, _, _, _, _, _, _ = outputs
        samples = samples.permute(0, 1, 3, 2)
        sample_mean = samples.mean(dim=1)

        rmse = ((sample_mean - missing_data) * missing_data_mask) ** 2
        rmse = rmse.sum().item() / missing_data_mask.sum().item()
        avg_rmse += math.sqrt(rmse)
        total_batch += 1
    print(f"Radius range: {radius_range}, Test RMSE: {avg_rmse/total_batch}")

radius_range = (110000, 160000)  # 110 km to 160 km

train_loader, test_loader = get_dataloader(total_stations, mean_std_file, n_features, batch_size=8, missing_ratio=0.02, type=data_type, data=data, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, is_multi=is_multi, is_test=True, southeast=False, sparse=False, missing_dims=M, parts=False, test_loc=test_coords, exclude_train_coords=None, old=old, radius_range=radius_range, quantity=quantity)

with torch.no_grad():
    total_batch = 0
    avg_rmse = 0.0
    for i, test_batch in enumerate(test_loader):
        if i == 0:
            input_locations = test_batch['spatial_info'][0]
            df_inputs = pd.DataFrame(input_locations, columns=['longitude', 'latitude', 'elevation'])
            df_inputs.to_csv(f'{folder}/input_locations_110_160.csv', index=False)
        missing_data_mask = test_batch['missing_data_mask'].squeeze(2).to(device)
        missing_data = test_batch['missing_data'].squeeze(2).to(device)
        input_locations = test_batch['spatial_info']
        missing_locations = test_batch['missing_data_loc']
        outputs = model_diff_saits.evaluate(test_batch, nsample, missing_dims=M)
        samples, _, _, _, _, _, _, _, _, _ = outputs
        samples = samples.permute(0, 1, 3, 2)
        sample_mean = samples.mean(dim=1)

        rmse = ((sample_mean - missing_data) * missing_data_mask) ** 2
        rmse = rmse.sum().item() / missing_data_mask.sum().item()
        avg_rmse += math.sqrt(rmse)
        total_batch += 1
    print(f"Radius range: {radius_range}, Test RMSE: {avg_rmse/total_batch}")

radius_range = (160000, 210000)  # 160 km to 210 km

train_loader, test_loader = get_dataloader(total_stations, mean_std_file, n_features, batch_size=8, missing_ratio=0.02, type=data_type, data=data, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, is_multi=is_multi, is_test=True, southeast=False, sparse=False, missing_dims=M, parts=False, test_loc=test_coords, exclude_train_coords=None, old=old, radius_range=radius_range, quantity=quantity)

with torch.no_grad():
    total_batch = 0
    avg_rmse = 0.0
    for i, test_batch in enumerate(test_loader):
        if i == 0:
            input_locations = test_batch['spatial_info'][0]
            df_inputs = pd.DataFrame(input_locations, columns=['longitude', 'latitude', 'elevation'])
            df_inputs.to_csv(f'{folder}/input_locations_160_210.csv', index=False)
        missing_data_mask = test_batch['missing_data_mask'].squeeze(2).to(device)
        missing_data = test_batch['missing_data'].squeeze(2).to(device)
        input_locations = test_batch['spatial_info']
        missing_locations = test_batch['missing_data_loc']
        outputs = model_diff_saits.evaluate(test_batch, nsample, missing_dims=M)
        samples, _, _, _, _, _, _, _, _, _ = outputs
        samples = samples.permute(0, 1, 3, 2)
        sample_mean = samples.mean(dim=1)

        rmse = ((sample_mean - missing_data) * missing_data_mask) ** 2
        rmse = rmse.sum().item() / missing_data_mask.sum().item()
        avg_rmse += math.sqrt(rmse)
        total_batch += 1
    print(f"Radius range: {radius_range}, Test RMSE: {avg_rmse/total_batch}")

radius_range = (210000, 260000)  # 210 km to 260 km

train_loader, test_loader = get_dataloader(total_stations, mean_std_file, n_features, batch_size=8, missing_ratio=0.02, type=data_type, data=data, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, is_multi=is_multi, is_test=True, southeast=False, sparse=False, missing_dims=M, parts=False, test_loc=test_coords, exclude_train_coords=None, old=old, radius_range=radius_range, quantity=quantity)

with torch.no_grad():
    total_batch = 0
    avg_rmse = 0.0
    for i, test_batch in enumerate(test_loader):
        if i == 0:
            input_locations = test_batch['spatial_info'][0]
            df_inputs = pd.DataFrame(input_locations, columns=['longitude', 'latitude', 'elevation'])
            df_inputs.to_csv(f'{folder}/input_locations_210_260.csv', index=False)
        missing_data_mask = test_batch['missing_data_mask'].squeeze(2).to(device)
        missing_data = test_batch['missing_data'].squeeze(2).to(device)
        input_locations = test_batch['spatial_info']
        missing_locations = test_batch['missing_data_loc']
        outputs = model_diff_saits.evaluate(test_batch, nsample, missing_dims=M)
        samples, _, _, _, _, _, _, _, _, _ = outputs
        samples = samples.permute(0, 1, 3, 2)
        sample_mean = samples.mean(dim=1)

        rmse = ((sample_mean - missing_data) * missing_data_mask) ** 2
        rmse = rmse.sum().item() / missing_data_mask.sum().item()
        avg_rmse += math.sqrt(rmse)
        total_batch += 1
    print(f"Radius range: {radius_range}, Test RMSE: {avg_rmse/total_batch}")