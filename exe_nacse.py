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
from models.deep_kriging import prepare_data, train_deep_kriging, get_model

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

train_loader, test_loader = get_dataloader(total_stations, mean_std_file, n_features, batch_size=8, missing_ratio=0.02, type=data_type, data=data, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate)

print(f"################### Data loading done ###################")

config_file = sys.argv[1]

try:
    with open(config_file, 'r') as f:
        config = json.load(f)
except Exception as e:
    print(f"Error reading the configuration file '{config_file}': {e}")
    sys.exit(1)

model_diff_saits = DynaSTI_NASCE(config, device, n_spatial=n_spatial).to(device)

filename = f"model_dynasti_nacse.pth"
print(f"\nDynaSTI training starts.....\n")
model_folder = 'saved_models_nacse'

# model_diff_saits.load_state_dict(torch.load(f"{model_folder}/{filename}"))
#

if not os.path.isdir(model_folder):
    os.makedirs(model_folder)

train(
    model_diff_saits,
    config["train"],
    train_loader,
    valid_loader=test_loader,
    foldername=model_folder,
    filename=f"{filename}",
    is_dit=config['is_dit_ca2'],
    d_spatial=config['model']['d_spatial'],
    d_time=config['model']['d_time'],
    is_spat=False,
    is_ema=is_ema,
    name=f"nacse"
)

print(f"DynaSTI params: {get_num_params(model_diff_saits)}")
# Create EMA handler with the main model
ema = EMA(model_diff_saits)

# Define the file path where the EMA model is saved
ema_model_filepath = f"{model_folder}/ema_model_nacse.pth"

# Load the saved EMA model
ema.load(ema_model_filepath)
model_diff_saits = ema.ema_model




########################## IGNNK ##############################
# model_ignnk = IGNNK(h=n_steps * n_features, z=128, k=1).to(device=device)
# lr = 1e-04
# max_iter = 5000
# train_ignnk(model_ignnk, lr, max_iter, train_loader, test_loader, f"{model_folder}/model_ignnk{'' if not is_neighbor else '_neighbor'}.model")

# model_ignnk.load_state_dict(torch.load(f"{model_folder}/model_ignnk.model"))

########################## DK ##############################
# coords_tensor, times_tensor, values_tensor, num_features = prepare_data(train_loader)
# dk_model = train_deep_kriging(1e-3, 500, coords_tensor[:, :2], times_tensor, values_tensor, num_features, f"{model_folder}/deep_kriging.model")
# dk_model = get_model(n_features)
# dk_model.load_state_dict(torch.load(f"{model_folder}/deep_kriging.model"))

models = {
    'SPAT-SADI': model_diff_saits,
    # 'IGNNK': model_ignnk,
    # 'GP': None,
    # 'DK': dk_model,
    'MEAN': None,
}

name = 'nacse_spatial_dit'
mse_folder = f"results_nacse/metric"
data_folder = f"results_nacse/data"
print(f"data folder: {data_folder}")

filename = (data_file_test, data_file_test_loc, mean_std_file)
evaluate_imputation_all(models=models, trials=1, mse_folder=data_folder, n_features=n_features, dataset_name='nasce', batch_size=2, filename=filename, spatial=True, simple=simple, unnormalize=True, n_stations=n_spatial, n_steps=n_steps, total_locations=total_stations, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, data=True)

# d_rates = [0.1, 0.3, 0.5, 0.7, 0.9]


# is_subset = False

# for dynamic_rate in d_rates:
#     print(f"dynamic rate: {dynamic_rate}")
#     evaluate_imputation_all(models=models, trials=20, mse_folder=mse_folder, n_features=n_features, dataset_name='nasce', batch_size=1, filename=filename, spatial=True, simple=simple, unnormalize=False, n_stations=n_spatial, n_steps=n_steps, total_locations=total_stations, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, dynamic_rate=dynamic_rate, is_subset=is_subset)

# is_subset = True
# evaluate_imputation_all(models=models, trials=20, mse_folder=mse_folder, n_features=n_features, dataset_name='nasce', batch_size=1, filename=filename, spatial=True, simple=simple, unnormalize=False, n_stations=n_spatial, n_steps=n_steps, total_locations=total_stations, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, is_subset=is_subset)
