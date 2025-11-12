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
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_scipy
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import botorch
botorch.settings.debug(True)

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

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))
 
def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

data = 'temps'
is_neighbor = False
is_separate = True
is_multi = True
n_steps = 30
n_features = 2 
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
M = 10
target_location_file = 'results_map_bayes_parts/0/target_locations.csv' #'results_map_random_parts/0/target_locations.csv'

train_loader, test_loader = get_dataloader(total_stations, mean_std_file, n_features, batch_size=8, missing_ratio=0.02, type=data_type, data=data, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, is_multi=is_multi, southeast=False, sparse=False, missing_dims=M, parts=False, is_test=True, target_loc_filename=target_location_file)

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

model_diff_saits.eval()
folder_2 = f"results_nacse/attention_map"
if not os.path.isdir(folder_2):
    os.makedirs(folder_2)
# with torch.no_grad():
#     for i, test_batch in enumerate(test_loader):
#         outputs = model_diff_saits.evaluate(test_batch, nsample, missing_dims=M)
#         samples, _, _, _, _, _, _, _, attn_spat_mean, sttn_spat_std = outputs
#         samples = samples.permute(0, 1, 3, 2)
#         samples_mean = samples.mean(dim=1)  # (B,L,N*K)
#         spatial_locs = test_batch['spatial_info'].to(device)
#         print(f"attn_spat_mean shape: {attn_spat_mean.shape}, spatial_locs shape: {spatial_locs.shape}")
#         exit()
#         attn_spat_mean = attn_spat_mean.squeeze(0).unsqueeze(-1).cpu().numpy()  # (N, 1)
#         spatial_positions = spatial_locs.squeeze(0).cpu().numpy()  # (N, 3)

#         df_array = np.concatenate([spatial_positions, attn_spat_mean], axis=1)
#         df_spat_attn = pd.DataFrame(df_array, columns=['longitude', 'latitude', 'elevation', 'attn'])
#         df_spat_attn.to_csv(f"{folder_2}/attn_map.csv")

#         break

# exit()
with torch.no_grad():
    for i, test_batch in enumerate(test_loader):
        outputs = model_diff_saits.evaluate(test_batch, nsample, missing_dims=M)
        samples, _, _, _, _, _, _, _, _, _ = outputs
        samples = samples.permute(0, 1, 3, 2)
        samples_mean = samples.mean(dim=1)  # (B,L,N*K)
        missing_data = test_batch['missing_data'].to(device)
        missing_data_mask = test_batch['missing_data_mask'].to(device)

        missing_data = missing_data.reshape(-1, n_steps, M * n_features)
        missing_data_mask = missing_data_mask.reshape(-1, n_steps, M * n_features)


        rmse_without_new_loc = ((samples_mean - missing_data) * missing_data_mask) ** 2
        rmse_without_new_loc = rmse_without_new_loc.sum().item() / missing_data_mask.sum().item()

        crps_without_new_loc = calc_quantile_CRPS(missing_data, samples, missing_data_mask, 0, 1)

        print(f"Batch {i}: RMSE without new loc: {rmse_without_new_loc}, CRPS without new loc: {crps_without_new_loc}")

        decided_locations_file = 'results_map_bayes_parts/0/decided_locations.csv' #'results_map_random_parts/0/decided_locations.csv'
        df_decided = pd.read_csv(decided_locations_file)
        decided_coords = df_decided[['longitude', 'latitude', 'elevation']].values
        decided_coords = torch.tensor(decided_coords, dtype=torch.float32)

        test_batch_copy = test_batch.copy()
        # test_batch_copy['spatial_info'] = torch.cat([test_batch_copy['spatial_info'], decided_coords], dim=0)
        test_batch_copy['missing_data_loc'] = decided_coords.reshape((1, -1, 3))
        test_batch_copy['missing_data'] = None
        test_batch_copy['gt_mask'] = torch.zeros((1, n_steps, decided_coords.shape[0], 2))
        test_batch_copy['missing_data_mask'] = torch.ones((1, n_steps, decided_coords.shape[0], 2))
        
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


        outputs_with_new_loc = model_diff_saits.evaluate(test_batch_copy, nsample, missing_dims=M)
        samples_with_new_loc, _, _, _, _, _, _, _, _, _ = outputs_with_new_loc
        samples_with_new_loc = samples_with_new_loc.permute(0, 1, 3, 2)
        samples_mean_with_new_loc = samples_with_new_loc.mean(dim=1)  # (B,L,N*K)

        rmse_with_new_loc = ((samples_mean_with_new_loc - missing_data) * missing_data_mask) ** 2
        rmse_with_new_loc = rmse_with_new_loc.sum().item() / missing_data_mask.sum().item() 

        crps_with_new_loc = calc_quantile_CRPS(missing_data, samples_with_new_loc, missing_data_mask, 0, 1)

        print(f"Batch {i}: RMSE with new loc: {rmse_with_new_loc}, CRPS with new loc: {crps_with_new_loc}")
        break