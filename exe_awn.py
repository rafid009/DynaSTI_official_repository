from diffusion.diff_wrapper import DynaSTI_AWN
from utils.utils import train
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import matplotlib
from datasets.dataset_awn import get_dataloader
from json import JSONEncoder
from utils.utils import evaluate_imputation_all, get_num_params
from models.ema import EMA
from utils.ignnk_util import train_ignnk
from models.ignnk import IGNNK
import torch.nn as nn
import warnings
import sys
import json
from models.deep_kriging import prepare_data, train_deep_kriging, get_model

# Ignore all warnings
warnings.filterwarnings("ignore")

matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
# torch.manual_seed(42)
np.set_printoptions(threshold=np.inf)
plt.rcParams.update({'font.size': 22})

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

given_features = [

    'AIR_TEMP_F',
    'SECOND_AIR_TEMP_F',
    'RELATIVE_HUMIDITY_%',
    'DEWPOINT_F',
    'WIND_SPEED_2M_MPH',
    'WIND_GUST_2M_MPH',
    'SOLAR_RAD_WM2',
]

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


n_steps = 288 # 672
n_features = len(given_features)
# 3 -> None, 6 -> add-delta, 3 -> sole-delta
n_spatial = 54 # 11
total_stations = 67 # 13
spatial_choice = None # 'sole-delta'
spatial_context_dim = 6 if spatial_choice == 'add-delta' else 3
miss_type = 'random'
seed = 400 # np.random.randint(10,100)
simple = False
is_neighbor = False
is_separate = True
is_ema = True

# filename: Any, is_year: bool = True, n_steps: int = 366
zone = 8 # 7
dataset_name = 'awn'
data_file_train = f'./data/awn/zone_{zone}_X_train.npy'
data_file_train_loc = f'./data/awn/zone_{zone}_train_locs.npy'
mean_std_file = f'./data/awn/zone_{zone}'
data_file_test = f'./data/awn/zone_{zone}_X_test.npy'
data_file_test_loc = f'./data/awn/zone_{zone}_test_locs.npy'
nsample = 50
print("################### Start ###################")
 #352 #len(given_features)

# train_loader, test_loader = get_dataloader(total_stations, mean_std_file, n_features, batch_size=8, missing_ratio=0.02, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, zone=zone)
train_loader, test_loader = get_dataloader(total_stations, mean_std_file, n_features, batch_size=4, missing_ratio=0.02, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, zone=zone)

print(f"################### Data loading done ###################")

config_file = 'configs/config_awn.json'

try:
    with open(config_file, 'r') as f:
        config = json.load(f)
except Exception as e:
    print(f"Error reading the configuration file '{config_file}': {e}")
    sys.exit(1)
no_se = not config['ablation']['se']
no_te = not config['ablation']['te']
no_fe = not config['ablation']['fe']

model_diff_saits = DynaSTI_AWN(config, device, n_spatial=n_spatial) #.to(device)
if torch.cuda.device_count() > 1:
    # print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model_diff_saits = nn.DataParallel(model_diff_saits, device_ids=list(range(torch.cuda.device_count())), dim=0)

# model_diff_saits.module.set_device()
# model_diff_saits.to(device)

filename = f"model_dynasti_awn.pth"
print(f"\nDynaSTI training starts.....\n")
model_folder = 'saved_models_awn'

# model_diff_saits.load_state_dict(torch.load(f"{model_folder}/{filename}"))
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
#     name=f"awn"
# )
# model_diff_saits.load_state_dict(torch.load(f"{model_folder}/{filename}"))
print(f"DynaSTI params: {get_num_params(model_diff_saits)}")
# Create EMA handler with the main model
ema = EMA(model_diff_saits)

# Define the file path where the EMA model is saved
ema_model_filepath = f"{model_folder}/ema_model_awn.pth"

# Load the saved EMA model
ema.load(ema_model_filepath)
model_diff_saits = ema.ema_model

##################### FFT DynaSTI #######################
latent_seq_dim = 16

config['model']['d_time'] = 2 * latent_seq_dim + 2


config['train']['epoch'] = 600
config['fft'] = True
n_iters = 100
lr = 0.01
random = False

model_diff_saits_fft = DynaSTI_AWN(config, device, n_spatial=n_spatial) #.to(device)

filename = f"model_dynasti_fft_awn{'_no_se' if no_se else ''}{'_no_te' if no_te else ''}{'_no_fe' if no_fe else ''}{'_random' if random else ''}.pth"
print(f"\nDynaSTI FFT training starts.....\n")

# train(
#     model_diff_saits_fft,
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
#     name=f"fft_awn{'_no_se' if no_se else ''}{'_no_te' if no_te else ''}{'_no_fe' if no_fe else ''}{'_random' if random else ''}",
#     latent_size=(latent_seq_dim, len(given_features), n_iters, lr, random)
# )

ema = EMA(model_diff_saits_fft)

# Define the file path where the EMA model is saved
ema_model_filepath = f"{model_folder}/ema_model_fft_awn{'_no_se' if no_se else ''}{'_no_te' if no_te else ''}{'_no_fe' if no_fe else ''}{'_random' if random else ''}.pth"

# Load the saved EMA model
ema.load(ema_model_filepath)
model_diff_saits_fft = ema.ema_model

############################## PriSTI ##############################
train_loader_pristi, test_loader_pristi = get_dataloader(total_stations, mean_std_file, n_features, batch_size=2, missing_ratio=0.02, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=False, is_pristi=True, zone=zone)
config['is_pristi'] = True
config['is_dit_ca2'] = False
config['is_separate'] = False
config['adj_file'] = 'awn'
config['train_stations'] = 54
config['model']['d_spatial'] = 67
config['model']['use_guide'] = True
config['model']['mask_sensor'] = []
config['train']['lr'] = 1e-04
config['train']['epochs'] = 1000
config['model']['d_time'] = 288
config['fft'] = False
is_ema = False
print(f"PriSTI config: {config}")
model_pristi = DynaSTI_AWN(config, device, n_spatial=n_spatial).to(device)

filename = f"model_pristi_awn.pth"
print(f"\nDynaSTI training starts.....\n")

# train(
#     model_pristi,
#     config["train"],
#     train_loader_pristi,
#     valid_loader=test_loader_pristi,
#     foldername=model_folder,
#     filename=f"{filename}",
#     is_dit=config['is_dit_ca2'],
#     d_spatial=config['model']['d_spatial'],
#     d_time=config['model']['d_time'],
#     is_spat=False,
#     is_ema=is_ema,
#     name=f"awn"
# )

# model_pristi.load_state_dict(torch.load(f"{model_folder}/{filename}"))


########################## IGNNK ##############################
model_ignnk = IGNNK(h=n_steps * n_features, z=256, k=3).to(device=device)
lr = 1e-05 # 1e-06
max_iter = 2000
# train_ignnk(model_ignnk, lr, max_iter, train_loader, test_loader, f"{model_folder}/model_ignnk_{zone}.model")

# model_ignnk.load_state_dict(torch.load(f"{model_folder}/model_ignnk.model"))

########################## DK ##############################
coords_tensor, times_tensor, values_tensor, num_features = prepare_data(train_loader)
# dk_model = train_deep_kriging(1e-3, 600, coords_tensor[:, :2], times_tensor, values_tensor, num_features, f"{model_folder}/deep_kriging.model")
# dk_model = get_model(n_features)
# dk_model.load_state_dict(torch.load(f"{model_folder}/deep_kriging.model"))
models = {
    # 'DynaSTI-Orig': model_diff_saits,
    'SPAT-SADI': model_diff_saits_fft,
    # 'MEAN': None,
    # 'PriSTI': model_pristi,
    # 'GP': None,
    # 'IGNNK': model_ignnk,
    # 'DK': dk_model

}

mse_folder = f"results_awn/metric"
data_folder = f"results_awn/data"
print(f"data folder: {data_folder}")

filename = (data_file_test, data_file_test_loc, mean_std_file)

dynamic_rate = -1
is_subset = False
# evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, n_features=n_features, dataset_name='awn', batch_size=2, filename=filename, spatial=True, simple=simple, unnormalize=False, n_stations=n_spatial, n_steps=n_steps, total_locations=total_stations, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, zone=zone, dynamic_rate=dynamic_rate, is_subset=is_subset, latent_size=(latent_seq_dim, len(given_features), n_iters, lr, random))
evaluate_imputation_all(models=models, trials=1, mse_folder=data_folder, n_features=n_features, dataset_name='awn', batch_size=1, filename=filename, spatial=True, simple=simple, unnormalize=True, n_stations=n_spatial, n_steps=n_steps, total_locations=total_stations, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, zone=zone, dynamic_rate=dynamic_rate, is_subset=is_subset, data=True, latent_size=(latent_seq_dim, len(given_features), n_iters, lr, random))

# dyn_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
# for dynamic_rate in dyn_rates:
#     print(f"dynamic rate: {dynamic_rate}")
#     evaluate_imputation_all(models=models, trials=20, mse_folder=mse_folder, n_features=n_features, dataset_name='awn', batch_size=2, filename=filename, spatial=True, simple=simple, unnormalize=False, n_stations=n_spatial, n_steps=n_steps, total_locations=total_stations, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, zone=zone, dynamic_rate=dynamic_rate, latent_size=(latent_seq_dim, len(given_features), n_iters, lr, random))

