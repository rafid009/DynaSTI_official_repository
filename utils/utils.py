import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from sklearn.metrics.pairwise import haversine_distances
from torch.optim import AdamW, Adam
import sys
from json import JSONEncoder
import pickle
import json
from tqdm import tqdm
import os
import math
from datasets.dataset_nasce import get_testloader_nasce
from datasets.dataset_awn import get_testloader_awn
from datasets.dataset_metrla import get_testloader_metrla
from datasets.dataset_pemsbay import get_testloader_pemsbay
import matplotlib.pyplot as plt
import matplotlib
from models.ema import EMA
from models.gp import trainGP, testGP
from models.deep_kriging import prepare_test_dataloader_multi, test_deepkriging_model_multi, create_basis_embeddings
import os
import torch.distributed as dist
from scipy.stats import ttest_rel
from sklearn.neighbors import NearestNeighbors
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)
plt.rcParams.update({'font.size': 22})

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=torch.inf)

def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)

def to_device(batch, device, is_train=True):
    batch["observed_data"].cuda().float() #.to(device).float()
    batch["observed_mask"].cuda().float() #.to(device).float()
    batch["gt_mask"].cuda().float() #.to(device).float()

    batch["spatial_info"].cuda().float() #.to(device).float()
    if not is_train:
        batch["missing_data"].cuda().float() #.to(device).float()
        batch["missing_data_mask"].cuda().float() #.to(device).float()
        batch['missing_data_loc'].cuda().float() #.to(device).float()

    
    batch["timepoints"].cuda().float() #.to(device).float()

    # batch["gt_intact"].to(device).float()

    batch["mean_loc"].cuda().float() #.to(device).float()
    batch['std_loc'].cuda().float() #.to(device).float()

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
def reduce_dict(input_dict, average=True):
    world_size = float(dist.get_world_size())
    names, values = [], []
    for k in sorted(input_dict.keys()):
        names.append(k)
        values.append(input_dict[k])
    values = torch.stack(values, dim=0)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if average:
        values /= world_size
    reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def convert_seconds(seconds):
    # Get days, hours, minutes, and remaining seconds
    days = seconds // 86400  # 86400 seconds in a day
    seconds %= 86400
    hours = seconds // 3600  # 3600 seconds in an hour
    seconds %= 3600
    minutes = seconds // 60  # 60 seconds in a minute
    seconds %= 60
    
    # Format and return the result
    return f"{int(days)}-{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
    filename="",
    is_saits=False,
    data_type="",
    is_dit=False,
    d_spatial=None,
    d_time=None,
    is_spat=False,
    is_ema=False,
    name=None,
    device=None,
    latent_size=None
):
    # device = torch.device(f"cuda:{dist.get_rank()}")
    if is_dit:
        optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=0)
        # optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    else:
        optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + f"/{filename if len(filename) != 0 else 'model_csdi.pth'}"

    
    # p0 = int(0.6 * config["epochs"])
    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    p3 = int(0.8 * config["epochs"])
    # p4 = int(0.7 * config["epochs"])
    p5 = int(0.6 * config["epochs"])
    # exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    if is_dit:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
        )
        # pass
    elif is_saits:
        if data_type == 'agaid':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[p1, p2], gamma=0.1
            )
        elif data_type == 'pm25':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[p1, p2], gamma=0.1
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[p1, p2], gamma=0.1
            )
        # pa
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
        )
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=1000, T_mult=1, eta_min=1.0e-7
    #     )
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20)
    losses_y = []
    best_valid_loss = 1e10
    if is_ema:
        ema = EMA(model, decay=0.999)
        ema_model_filepath = f'{foldername}/ema_model_{name}.pth'

    # scaler = torch.cuda.amp.GradScaler()
    
    model = model.cuda()
    model.train()
    start_time_train = time.time()
    total_validation_time = 0
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        # if epoch_no == 1000:
        #     torch.save(model.state_dict(), output_path)
        #     model.load_state_dict(torch.load(f"{output_path}"))
        # if epoch_no > 1000 and epoch_no % 500 == 0:
        #     torch.save(model.state_dict(), output_path)
        #     model.load_state_dict(torch.load(f"{output_path}"))
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                # to_device(train_batch, device, True)
                optimizer.zero_grad()
                # print(f"train data: {train_batch}")
                # if epoch_no > 500:
                #     loss = model(train_batch, is_spat=is_spat)
                # else:
                # with torch.cuda.amp.autocast():
                loss = model(train_batch, latent_size=latent_size)
                # print(f"loss: {loss.shape}\nloss values: {loss}")
                if len(loss.shape) > 0:
                    loss = loss.mean()
                # print(f"loss: {loss}")
                loss.backward()
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         if param.grad is not None:
                #             print(f"{name}: {param.grad.data.sum()}")
                #         else:
                #             print(f"{name}: {param.grad}")
                avg_loss += loss.mean().item()
                avg_loss += loss.item()
                optimizer.step()

                if is_ema:
                    ema.update()
                # lr_scheduler.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                # exit()
            losses_y.append(avg_loss / batch_no)
            # exp_scheduler.step()
            # metric = avg_loss / batch_no
            if is_dit:
                # lr_scheduler.step()
                pass
            elif is_saits:
                # if data_type != 'pm25' and data_type != 'synth_v2' and data_type != 'synth_v3':
                #     lr_scheduler.step()
                # pass
                # if data_type == 'electricity':
                #     pass
                # else:
                if data_type != 'nasce' and data_type != 'pm25' and data_type != 'electricity' and data_type != 'synth_v8':
                    lr_scheduler.step()
            else:
                lr_scheduler.step()
                # pass

        # print(f"\nPrinting wieghts for epoch = {epoch_no}")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {param.data}")
        # print(f"Finished printing weights for epoch = {epoch_no}\n")
        start_valid_time = time.time()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        # to_device(valid_batch, device, False)
                        loss = model(valid_batch, is_train=0, latent_size=latent_size)
                        if len(loss.shape) > 0:
                            avg_loss_valid += loss.mean().item()
                        else:
                            avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            torch.save(model.state_dict(), output_path)
            if is_ema:
                ema.save(ema_model_filepath)
            model.train()
                # print(
                #     "\n avg loss is now ",
                #     avg_loss_valid / batch_no,
                #     "at",
                #     epoch_no,
                # )
            end_valid_time = time.time()
            total_validation_time += end_valid_time - start_valid_time
    end_time_train = time.time()

    full_train_duration = end_time_train - start_time_train
    print(f"Full train + valid duration: {convert_seconds(full_train_duration)}")
    print(f"Only training time: {convert_seconds(full_train_duration - total_validation_time)}")
    if filename != "":
        torch.save(model.state_dict(), output_path)
        if is_ema:
            ema.save(ema_model_filepath)
    # if filename != "":
    #     torch.save(model.state_dict(), filename)
    x = np.arange(len(losses_y))
    folder = "./spat_sadi_img"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    plt.figure(figsize=(16,9))
    plt.plot(x,losses_y, label='Train Loss')
    plt.title(f"Training Losses for {d_spatial} stations {d_time} day window")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{folder}/station_{d_spatial}_window_{d_time}.png", dpi=300)
    plt.tight_layout()
    plt.close()


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


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time, _, _ = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "mse_total": mse_total / evalpoints_total,
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        mse_total / evalpoints_total,
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("MSE:", mse_total / evalpoints_total)
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def geographical_distance(x=None, to_rad=True):
    _AVG_EARTH_RADIUS_KM = 6371.0088

    # Extract values of X if it is a DataFrame, else assume it is 2-dim array of lat-lon pairs
    latlon_pairs = x.values if isinstance(x, pd.DataFrame) else x

    # If the input values are in degrees, convert them in radians
    if to_rad:
        latlon_pairs = np.vectorize(np.radians)(latlon_pairs)

    distances = haversine_distances(latlon_pairs) * _AVG_EARTH_RADIUS_KM

    # Cast response
    if isinstance(x, pd.DataFrame):
        res = pd.DataFrame(distances, x.index, x.index)
    else:
        res = distances

    return res


def thresholded_gaussian_kernel(x, theta=None, threshold=None, threshold_on_input=False):
    if theta is None:
        theta = np.std(x)
    weights = np.exp(-np.square(x / theta))
    if threshold is not None:
        mask = x > threshold if threshold_on_input else weights < threshold
        weights[mask] = 0.
    return weights


def get_similarity_AQI(dist, thr=0.1, include_self=False, force_symmetric=False, sparse=False):
    theta = np.std(dist[:36, :36])  # use same theta for both air and air36
    adj = thresholded_gaussian_kernel(dist, theta=theta, threshold=thr)
    if not include_self:
        adj[np.diag_indices_from(adj)] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj


def get_adj_AQI36():
    df = pd.read_csv("./data/pm25/SampleData/pm25_latlng.txt")
    df = df[['latitude', 'longitude']]
    res = geographical_distance(df, to_rad=False).values
    adj = get_similarity_AQI(res)
    return adj


def get_similarity_metrla(thr=0.1, force_symmetric=False, sparse=False):
    dist = np.load('./data/metr_la/metr_la_dist.npy')
    finite_dist = dist.reshape(-1)
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < thr] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj


def get_similarity_pemsbay(thr=0.1, force_symmetric=False, sparse=False):
    dist = np.load('./data/pems_bay/pems_bay_dist.npy')
    finite_dist = dist.reshape(-1)
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < thr] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj


# in Graph-wavenet
def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def compute_support_gwn(adj, device=None):
    adj_mx = [asym_adj(adj), asym_adj(np.transpose(adj))]
    support = [torch.tensor(i).to(device) for i in adj_mx]
    return support

def get_nan_location(target_mask, n_features):
    indices = []
    # print(f"target: {target_mask.shape}")
    for i in range(target_mask.shape[1]):
        flag = False
        for n in range(n_features):
            if (target_mask[:, i, n] == 1).all() :
                flag = True
            else:
                break
        if flag:
            indices.append(i)
    return indices

def geographical_distance(x=None, to_rad=True):
    _AVG_EARTH_RADIUS_KM = 6371.0088

    # Extract values of X if it is a DataFrame, else assume it is 2-dim array of lat-lon pairs
    latlon_pairs = x.values if isinstance(x, pd.DataFrame) else x

    # If the input values are in degrees, convert them in radians
    if to_rad:
        latlon_pairs = np.vectorize(np.radians)(latlon_pairs)
    # print(f"latlong_pair: {latlon_pairs.shape}\n{latlon_pairs[0]}")
    distances = haversine_distances(latlon_pairs) * _AVG_EARTH_RADIUS_KM

    # Cast response
    if isinstance(x, pd.DataFrame):
        res = pd.DataFrame(distances, x.index, x.index)
    else:
        res = distances

    return res


def thresholded_gaussian_kernel(x, theta=None, threshold=None, threshold_on_input=False):
    if theta is None:
        theta = np.std(x)
    weights = np.exp(-np.square(x / theta))
    if threshold is not None:
        mask = x > threshold if threshold_on_input else weights < threshold
        weights[mask] = 0.
    return weights

def get_similarity(dist, thr=0.1, include_self=False, force_symmetric=False, sparse=False):
    theta = np.std(dist)  # use same theta for both air and air36
    adj = thresholded_gaussian_kernel(dist, theta=theta, threshold=thr)
    if not include_self:
        adj[np.diag_indices_from(adj)] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj

def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()

def batched_inverse_distance_weighting_with_nan(distances, values, power):
    """
    Perform Inverse Distance Weighting (IDW) for a batch of time-series vector-valued data,
    computing one interpolated value per time step, while ignoring NaN values.

    Args:
        distances (numpy.ndarray): Array of shape (batch_size, 1), where each entry corresponds
                                    to the distances for a batch.
        values (numpy.ndarray): Array of shape (batch_size, time_series_length, features), where
                                each slice along the time axis corresponds to a vector value.
        power (float): Power parameter to control the influence of distances.

    Returns:
        numpy.ndarray: Interpolated values of shape (batch_size, time_series_length, features).
    """
    # Add a small epsilon to distances to avoid division by zero
    epsilon = 1e-10
    distances = np.maximum(distances, epsilon)  # Shape: (batch_size, 1)
    
    # Calculate weights (1 / distance^power) for each distance
    weights = 1 / (distances ** power)  # Shape: (batch_size, 1)
    
    # Expand weights to match the shape of the values for broadcasting
    weights_expanded = weights[:, :, np.newaxis]  # Shape: (batch_size, 1, 1)
    
    # Mask out NaN values in the `values` array
    nan_mask = ~np.isnan(values)  # Shape: (batch_size, time_series_length, features)
    
    # Set NaN values to 0 in the values array (to avoid propagating them)
    values_without_nan = np.where(nan_mask, values, 0)  # Replace NaN with 0

    # Numerator: weighted sum of valid values (ignoring NaNs)
    numerator = np.sum(weights_expanded * values_without_nan, axis=1, keepdims=True)  # Shape: (batch_size, 1, features)
    
    # Denominator: sum of weights, only where values are valid (ignoring NaNs)
    valid_weights = weights_expanded * nan_mask.astype(float)  # Apply NaN mask to weights
    denominator = np.sum(valid_weights, axis=1, keepdims=True)  # Shape: (batch_size, 1, 1)
    
    # Avoid division by zero in the denominator
    denominator = np.maximum(denominator, epsilon)
    
    # Compute the interpolated values for each time step
    interpolated_values = numerator / denominator  # Shape: (batch_size, time_series_length, features)
    
    return interpolated_values

def get_distances(location, missing_locations):
    # print(f"location: {location.shape}, missing_location: {missing_locations.shape}")
    return torch.sqrt((location[:, :, 0]-missing_locations[:, :, 0]) ** 2 + (location[:, :, 1]-missing_locations[:, :, 1]) ** 2)# + (location[:, 2]-missing_locations[:, 2]) ** 2)

def get_mean_std(dataset_name, filename, mean=None, std=None):
    if dataset_name == 'pm25':
        path = "./data/pm25/pm25_meanstd.pk"
        with open(path, "rb") as f:
            train_mean, train_std = pickle.load(f)
            train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
            train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
    elif dataset_name == 'electricity':
        path_mean = "./data/Electricity/mean.npy"
        path_std = "./data/Electricity/std.npy"
        train_mean = np.load(path_mean)
        train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
        train_std = np.load(path_std)
        train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
    elif dataset_name == 'nasce':
        path_mean = f"{filename[2]}_mean.npy"
        path_std = f"{filename[2]}_std.npy"
        train_mean = np.load(path_mean)
        train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
        train_std = np.load(path_std)
        train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
    elif dataset_name == 'awn':
        path_mean = f"{filename[2]}_mean.npy"
        path_std = f"{filename[2]}_std.npy"
        train_mean = np.load(path_mean)
        train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
        train_std = np.load(path_std)
        train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
    elif dataset_name == 'metrla' or dataset_name == 'pemsbay':
        path_mean = f"{filename[2]}_mean.npy"
        path_std = f"{filename[2]}_std.npy"
        train_mean = np.load(path_mean)
        train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
        train_std = np.load(path_std)
        train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
    else:
        train_mean = torch.tensor(mean, dtype=torch.float32, device=device)
        train_std = torch.tensor(std, dtype=torch.float32, device=device)
    return train_mean, train_std

def calculate_mis(samples, y_true, alpha=0.1):
    """
    Calculate Mean Interval Score (MIS) for diffusion-based probabilistic forecasts.
    
    Args:
        samples (Tensor): shape (S, T, D) 
                          S = number of stochastic samples
        y_true  (Tensor): shape (T, D)
        alpha (float): miscoverage rate (e.g., 0.1 => 90% interval)

    Returns:
        Tensor (scalar): MIS
    """

    # Compute quantiles along sample dimension
    samples = samples.reshape(samples.shape[0], samples.shape[1], -1)
    y_true = y_true.reshape(y_true.shape[0], -1)

    # print(f"samples shape: {samples.shape}, y_true shape: {y_true.shape}")
    lower = samples.quantile(alpha/2, dim=1)     # shape (T, D)
    upper = samples.quantile(1 - alpha/2, dim=1) # shape (T, D)
    # print(f"lower shape: {lower.shape}, upper shape: {upper.shape}")
    # Interval width component
    width = upper - lower

    # Penalties for outside interval
    under_penalty = (lower - y_true).clamp(min=0)
    over_penalty  = (y_true - upper).clamp(min=0)

    # Interval score per timestep & variable
    interval_score = width + (2/alpha) * (under_penalty + over_penalty)

    # Mean across time and dimensions
    mis = interval_score.mean()
    return mis.cpu().numpy()


def evaluate_imputation_all(models, mse_folder, dataset_name='', batch_size=16, trials=3, length=-1, random_trial=False, forecasting=False, missing_ratio=0.01, test_indices=None, 
                            data=False, noise=False, filename=None, is_yearly=True, n_features=-1, n_steps=366, pattern=None, 
                            mean=None, std=None, partial_bm_config=None, spatial=False, unnormalize=False,
                             simple=False, n_stations=100, total_locations=179, is_neighbor=False, spatial_choice=None, is_separate=False, zone=7, spatial_slider=False, dynamic_rate=-1, is_subset=False, missing_dims=-1, is_multi=False, latent_size=None, deltas=False, parts=False):  
    nsample = 50
    if 'CSDI' in models.keys():
        models['CSDI'].eval()
    if 'SADI' in models.keys():
        models['SADI'].eval()

    if 'DynaSTI-Orig' in models.keys():
        models['DynaSTI-Orig'] = models['DynaSTI-Orig'].cuda()
        models['DynaSTI-Orig'].eval()
    if 'SPAT-SADI' in models.keys():
        models['SPAT-SADI'] = models['SPAT-SADI'].cuda()
        models['SPAT-SADI'].eval()
    if 'PriSTI' in models.keys():
        models['PriSTI'].eval()
    if 'DK' in models.keys():
        models['DK'].eval()

    results_trials_mse = {'csdi': {}, 'spat-sadi': {}, 'spat-sadi-orig': {}, 'sadi': {}, 'pristi': {}, 'mean': {}, 'mice': {},
                           'kriging': {}, 'ignnk': {}, 'interpolation': {}, 'gp': {}, 'dk': {}}
    results_trials_mae = {'csdi': {}, 'spat-sadi': {}, 'spat-sadi-orig': {}, 'sadi': {}, 'pristi': {}, 'mean': {}, 'mice': {}}
    results_mse = {'csdi': 0, 'spat-sadi': 0, 'spat-sadi-orig': 0, 'sadi': 0, 'mean': 0, 'pristi': 0, 'mice': 0,
                    'kriging': 0, 'ignnk': 0, 'interpolation': 0, 'gp': 0, 'dk': 0}
    results_mae = {'csdi': 0, 'spat-sadi': 0, 'sadi': 0}
    results_crps = {
        'csdi_trials':{}, 'csdi': 0, 
        'spat-sadi_trials': {}, 'spat-sadi': 0,
        'spat-sadi-orig_trials': {}, 'spat-sadi-orig': 0,
        'sadi_trials': {}, 'sadi': 0,
        'pristi_trials': {}, 'pristi': 0
        }
    results_mis = {
        'csdi_trials':{}, 'csdi': 0, 
        'spat-sadi_trials': {}, 'spat-sadi': 0,
        'spat-sadi-orig_trials': {}, 'spat-sadi-orig': 0,
        'sadi_trials': {}, 'sadi': 0,
        'pristi_trials': {}, 'pristi': 0
        }
    results_data = {}
    
    if forecasting and not data and isinstance(length, tuple):
            range_len = (length[0], length[1])
    else:
        range_len = None
    if data:
        trials = 1
    s = 10 #np.random.randint(0,100)
    for trial in range(trials):
        if forecasting and not data and range_len is not None:
            length = np.random.randint(low=range_len[0], high=range_len[1] + 1)
        if dataset_name == 'nasce':
            test_loader = get_testloader_nasce(total_locations, filename[2], n_features, n_steps=n_steps, batch_size=batch_size, missing_ratio=missing_ratio, seed=(s + trial), length=length, forecasting=forecasting, random_trial=random_trial, pattern=pattern, partial_bm_config=partial_bm_config, spatial=spatial, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, spatial_slider=spatial_slider, dynamic_rate=dynamic_rate, is_subset=is_subset, missing_dims=missing_dims, deltas=deltas, parts=parts)
        if dataset_name == 'awn':
            test_loader = get_testloader_awn(total_locations, filename[2], n_features, n_steps=n_steps, batch_size=batch_size, missing_ratio=missing_ratio, seed=(s + trial), length=length, forecasting=forecasting, random_trial=random_trial, pattern=pattern, partial_bm_config=partial_bm_config, spatial=spatial, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, zone=zone, spatial_slider=spatial_slider, dynamic_rate=dynamic_rate, is_subset=is_subset)
        if dataset_name == 'metrla':
            test_loader = get_testloader_metrla(total_locations, filename[2], n_features, n_steps=n_steps, batch_size=batch_size, missing_ratio=missing_ratio, seed=(s + trial), length=length, forecasting=forecasting, random_trial=random_trial, pattern=pattern, partial_bm_config=partial_bm_config, spatial=spatial, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, dynamic_rate=dynamic_rate)
        if dataset_name == 'pemsbay':
            test_loader = get_testloader_pemsbay(total_locations, filename[2], n_features, n_steps=n_steps, batch_size=batch_size, missing_ratio=missing_ratio, seed=(s + trial), length=length, forecasting=forecasting, random_trial=random_trial, pattern=pattern, partial_bm_config=partial_bm_config, spatial=spatial, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, dynamic_rate=dynamic_rate)
        if dataset_name == 'synth':
            test_loader = get_testloader_synth(filename[2], n_features, n_steps=n_steps, batch_size=batch_size, missing_ratio=missing_ratio, seed=(s + trial), length=length, forecasting=forecasting, random_trial=random_trial, pattern=pattern, partial_bm_config=partial_bm_config, spatial=spatial, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, dynamic_rate=dynamic_rate)
        csdi_rmse_avg = 0
        diffsaits_rmse_avg = 0
        diffsaits_rmse_avg_orig = 0
        ignnk_rmse_avg = 0
        sadi_rmse_avg = 0
        mean_avg_rmse = 0
        kriging_rmse_avg = 0
        interp_rmse_avg = 0

        saits_rmse_avg = 0
        knn_rmse_avg = 0
        mice_rmse_avg = 0
        brits_rmse_avg = 0
        pristi_rmse_avg = 0
        gp_rmse_avg = 0
        dk_rmse_avg = 0

        csdi_mae_avg = 0
        diffsaits_mae_avg = 0
        saits_mae_avg = 0


        csdi_crps_avg = 0
        diffsaits_crps_avg = 0
        diffsaits_crps_avg_orig = 0
        sadi_crps_avg = 0
        pristi_crps_avg = 0

        diffsaits_mis_avg = 0
        pristi_mis_avg = 0

        trial_miss = 0
        total_batch = 0
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            
            for j, test_batch in enumerate(it, start=1):
                # print(f"missing data loc test: {test_batch['missing_data_loc']}")
                
                missing_data = test_batch['missing_data'].squeeze(2).to(device)
                # print(f"missing data: {missing_data.shape}")
                missing_data_mask = test_batch['missing_data_mask'].squeeze(2).to(device)
                if missing_dims != -1:
                    missing_data = missing_data.reshape(-1, n_steps, missing_dims * n_features)
                    missing_data_mask = missing_data_mask.reshape(-1, n_steps, missing_dims * n_features)
                missing_data_loc = test_batch["missing_data_loc"].to(device)
                spatial_loc = test_batch['spatial_info'].to(device)
                if 'CSDI' in models.keys():
                    with torch.no_grad():
                        output = models['CSDI'].evaluate(test_batch, nsample)
                        samples, c_target, eval_points, observed_points, _, gt_intact, missing_data, missing_data_mask = output
                        samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                        c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                        eval_points = eval_points.permute(0, 2, 1)
                        observed_points = observed_points.permute(0, 2, 1)
                        samples_median = samples.median(dim=1)

                if 'OKriging' in models.keys():
                    kriging_mean, kriging_std = okrigging(test_batch)
                        
                
                if 'GP' in models.keys():
                    spatial_tensor = test_batch['spatial_info'].to(device) # B, N, d
                    d = spatial_tensor.shape[-1]
                    input_data = test_batch['gt_intact'].to(device) # B, L, N, K
                    # print(f"input data: {input_data.shape}, spatial_tensor: {spatial_tensor.shape}")

                    train_mean, train_std = get_mean_std(dataset_name, filename, mean, std)

                    input_data = (input_data - train_mean) / train_std
                    
                    n_time = input_data.shape[1]
                    n_spatial_locations = input_data.shape[2]
                    locations = spatial_tensor
                    
                    # -----------------------------
                    # Step 1: Create the new coordinate tensor
                    # -----------------------------
                    # We first expand the spatial locations so that for each time step we have the same spatial info.
                    # Start with 'locations' of shape (B, 1, 3). We add a time dimension in position 1.
                    locations_expanded = locations.unsqueeze(1)  # shape becomes (B, 1, n_spatial_locations, 3)

                    # Now, expand along the time dimension to replicate for each time step.
                    locations_expanded = locations_expanded.expand(input_data.shape[0], n_time, n_spatial_locations, d)  
                    # shape: (B, time_steps, n_spatial_locations, 3)

                    # -----------------------------
                    # Step 1: Generate unique time indices per batch
                    # -----------------------------
                    # For each batch, we want the time indices to be shifted by (batch_index * time_steps)
                    # For example:
                    #   For batch 0: time indices are [0, 1, ..., time_steps - 1]
                    #   For batch 1: time indices are [time_steps, time_steps + 1, ..., 2*time_steps - 1]
                    batch_offsets = torch.arange(input_data.shape[0]).unsqueeze(1) * n_time  # shape: (batch_size, 1)
                    # Create a (1, time_steps) tensor for the base time indices
                    base_time = torch.arange(n_time).unsqueeze(0)  # shape: (1, time_steps)
                    # Add the batch offset to get unique time indices for each batch; shape becomes (batch_size, time_steps)
                    time_indices = base_time + batch_offsets  
                    # Reshape to make it broadcastable: (batch_size, time_steps, 1, 1)
                    time_indices = time_indices.unsqueeze(2).unsqueeze(3).float().to(device)
                    # Then explicitly expand to (B, time_steps, n_spatial_locations, 1)
                    time_indices = time_indices.expand(input_data.shape[0], n_time, n_spatial_locations, 1)
                    # Concatenate the expanded spatial locations with the time indices along the last dimension.
                    # This creates a tensor with shape (B, time_steps, n_spatial_locations, 4)
                    new_coords = torch.cat([locations_expanded, time_indices], dim=-1)



                    # Reshape new_coords to have shape (n_time*n_spatial_locations, 4)
                    train_x = new_coords.reshape(-1, d+1)
                    train_y = input_data.reshape((-1, input_data.shape[-1]))
                    # print(f"1. train_x: {train_x.shape}, train_y: {train_y.shape}")

                    # Create a boolean mask that is True only for rows in train_y without any NaN values.
                    mask = ~torch.isnan(train_y).any(dim=1)

                    # Apply the mask to both train_x and train_y.
                    train_x = train_x[mask]
                    train_y = train_y[mask]
                    # print(f"2. train_x: {train_x.shape}, train_y: {train_y.shape}")
                    gp_start = time.time()
                    gp, likelihood, x_scaler = trainGP(train_x, train_y, d)

                    ##################### Test Data #####################
                    target_data = test_batch['missing_data'].to(device) # B, L, 1, K
                    target_spatial_tensor = test_batch["missing_data_loc"].to(device) # B, 1, d

                    n_spatial_locations = target_data.shape[2]
                    n_time = target_data.shape[1]
                    locations = target_spatial_tensor

                    
                    # -----------------------------
                    # Step 1: Create the new coordinate tensor
                    # -----------------------------
                    # We first expand the spatial locations so that for each time step we have the same spatial info.
                    # Start with 'locations' of shape (B, 1, 3). We add a time dimension in position 1.
                    locations_expanded = locations.unsqueeze(1)  # shape becomes (B, 1, n_spatial_locations, 3)

                    # Now, expand along the time dimension to replicate for each time step.
                    locations_expanded = locations_expanded.expand(target_data.shape[0], n_time, n_spatial_locations, d)  
                    # shape: (B, time_steps, n_spatial_locations, 3)

                    # -----------------------------
                    # Step 1: Generate unique time indices per batch
                    # -----------------------------
                    # For each batch, we want the time indices to be shifted by (batch_index * time_steps)
                    # For example:
                    #   For batch 0: time indices are [0, 1, ..., time_steps - 1]
                    #   For batch 1: time indices are [time_steps, time_steps + 1, ..., 2*time_steps - 1]
                    batch_offsets = torch.arange(target_data.shape[0]).unsqueeze(1) * n_time  # shape: (batch_size, 1)
                    # Create a (1, time_steps) tensor for the base time indices
                    base_time = torch.arange(n_time).unsqueeze(0)  # shape: (1, time_steps)
                    # Add the batch offset to get unique time indices for each batch; shape becomes (batch_size, time_steps)
                    time_indices = base_time + batch_offsets  
                    # Reshape to make it broadcastable: (batch_size, time_steps, 1, 1)
                    time_indices = time_indices.unsqueeze(2).unsqueeze(3).float().to(device)
                    # Then explicitly expand to (B, time_steps, n_spatial_locations, 1)
                    time_indices = time_indices.expand(target_data.shape[0], n_time, n_spatial_locations, 1)

                    # Concatenate the expanded spatial locations with the time indices along the last dimension.
                    # This creates a tensor with shape (B, time_steps, n_spatial_locations, 4)
                    new_coords = torch.cat([locations_expanded, time_indices], dim=-1)

                    test_x = new_coords.reshape(-1, d+1)

                    pred_gp, _ = testGP(gp, likelihood, test_x, x_scaler) # 
                    pred_gp = pred_gp.reshape(target_data.shape).squeeze(dim=2)
                    gp_end = time.time()
                    # print(f"GP time: {gp_end - gp_start}s")
                    del gp
                    del likelihood
                    torch.cuda.empty_cache()

                if 'DK' in models.keys():
                    dk_start = time.time()
                    coords_tensor = test_batch['missing_data_loc'].to(device)
                    time_steps = test_batch['missing_data'].shape[1]
                    embedder, W_basis, T_basis = create_basis_embeddings()
                    loader, dk_shape = prepare_test_dataloader_multi(coords_tensor, time_steps, embedder, device=device)
                    dk_output = test_deepkriging_model_multi(model=models['DK'], dataloader=loader, reshape_dims=dk_shape, device=device)
                    # print(f"dk_output: {dk_output.shape}")
                    dk_output = dk_output.permute(0, 2, 1, 3).squeeze(dim=2) # B, L, 1, K
                    dk_end = time.time()
                    # print(f"DK time: {dk_end-dk_start}s")
                
                if 'DynaSTI-Orig' in models.keys():
                    with torch.no_grad():
                        spat_start = time.time()
                        output_diff_saits_orig = models['DynaSTI-Orig'].evaluate(test_batch, nsample, latent_size=latent_size, missing_dims=missing_dims)
                        spat_end = time.time()
                        print(f"orig time: {(spat_end-spat_start)/batch_size}s")
                        if 'CSDI' not in models.keys():
                            samples_diff_saits_orig, c_target, eval_points, observed_points, _, gt_intact, _, _, attn_spat_mean, attn_spat_std = output_diff_saits_orig
                        
                            c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                            eval_points = eval_points.permute(0, 2, 1)
                            observed_points = observed_points.permute(0, 2, 1)
                            # if is_separate:
                                # print(f"missing data: {missing_data.shape}")
                                # missing_data = missing_data.squeeze(1).permute(0, 2, 1)
                                # missing_data_mask = missing_data_mask.squeeze(1).permute(0, 2, 1)
                                # missing_data_loc = test_batch["missing_data_loc"]
                                # spatial_loc = test_batch['spatial_info']
                                # print(f"missing data loc in utils: {missing_data_loc}")
                        else:
                            samples_diff_saits_orig, _, _, _, _ ,_, _= output_diff_saits_orig
                        samples_diff_saits_orig = samples_diff_saits_orig.permute(0, 1, 3, 2)
                        samples_diff_saits_orig_mean = samples_diff_saits_orig.mean(dim=1)

                if 'SPAT-SADI' in models.keys():
                    with torch.no_grad():
                        spat_start = time.time()
                        output_diff_saits = models['SPAT-SADI'].evaluate(test_batch, nsample, latent_size=latent_size, missing_dims=missing_dims)
                        spat_end = time.time()
                        print(f"spat time: {spat_end-spat_start}s")
                        if 'CSDI' not in models.keys():
                            samples_diff_saits, c_target, eval_points, observed_points, _, gt_intact, _, _, attn_spat_mean, attn_spat_std = output_diff_saits
                        
                            c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                            eval_points = eval_points.permute(0, 2, 1)
                            observed_points = observed_points.permute(0, 2, 1)
                            # if is_separate:
                                # print(f"missing data: {missing_data.shape}")
                                # missing_data = missing_data.squeeze(1).permute(0, 2, 1)
                                # missing_data_mask = missing_data_mask.squeeze(1).permute(0, 2, 1)
                                # missing_data_loc = test_batch["missing_data_loc"]
                                # spatial_loc = test_batch['spatial_info']
                                # print(f"missing data loc in utils: {missing_data_loc}")
                        else:
                            samples_diff_saits, _, _, _, _ ,_, _= output_diff_saits
                        samples_diff_saits = samples_diff_saits.permute(0, 1, 3, 2)
                        samples_diff_saits_mean = samples_diff_saits.mean(dim=1)

                        # print(f"spat-sadi mean: {samples_diff_saits_mean.shape}")

                if 'Interpolation' in models.keys():
                    distances = get_distances(test_batch['spatial_info'].cpu().numpy(), test_batch["missing_data_loc"].cpu().numpy())
                    values = test_batch['observed_data'].cpu().numpy()
                    power = 1
                    output_interpolation = batched_inverse_distance_weighting_with_nan(distances, values, power)

                if 'IGNNK' in models.keys():
                    observed_data = test_batch['observed_data'].to(device).float()
                    observed_mask = test_batch['observed_mask'].to(device).float()
                    observed_data = observed_data.permute(0, 2, 3, 1) # B, N, K, L
                    observed_mask = observed_mask.permute(0, 2, 3, 1) # B, N, K, L
                    spatial_info = test_batch['spatial_info'].to(device).float()
                    missing_data_temp = test_batch['missing_data'].to(device).float()
                    missing_data_mask_temp = test_batch["missing_data_mask"].to(device).float()
                    missing_data_loc_temp = test_batch["missing_data_loc"].to(device).float()
                    missing_data_temp = missing_data_temp.permute(0, 2, 3, 1) # B, 1, K, L
                    missing_data_mask_temp = missing_data_mask_temp.permute(0, 2, 3, 1) # B, 1, K, L
                    if is_multi:
                        _, M, _, _ = missing_data_temp.shape
                    observed_data = torch.cat([observed_data, missing_data_temp], dim=1) # B, N+1, K, L
                    observed_mask = torch.cat([observed_mask, missing_data_mask_temp], dim=1) # B, N+1, K, L
                    locations = torch.cat([spatial_info, missing_data_loc_temp], dim=1) # B, N+1, K, L
                    locations = locations[0, :, :2]
                    dist_graph = geographical_distance(locations.cpu().numpy())
                    adj = get_similarity(dist_graph)

                    A_q = torch.from_numpy((calculate_random_walk_matrix(adj).T).astype('float32')).to(device=device)
                    A_h = torch.from_numpy((calculate_random_walk_matrix(adj.T).T).astype('float32')).to(device=device)

                    B, N, K, L = observed_data.shape
                    input_data = observed_data.clone()
                    if is_multi:
                        input_data[:, -M:, :, :] = 0.0
                    else:
                        input_data[:, -1, :, :] = 0.0
                    input_data = input_data.reshape((B, N, K*L)).permute(0, 2, 1).to(device=device)
                    
                    # loss = STmodel(valid_batch, is_train=0)
                    ignnk_start = time.time()
                    output_ignnk = models['IGNNK'](input_data, A_q, A_h)
                    ignnk_end = time.time()
                    # print(f"ignnk time: {ignnk_end-ignnk_start}s")
                    if is_separate:
                        # print(f"output: {output_ignnk.shape}")
                        if is_multi:
                            output_ignnk = output_ignnk.permute(0,2,1)[:,-M:,:].reshape((B, M, K, L)).permute(0, 3, 1, 2) # B, L, M, K
                        else:
                            output_ignnk = output_ignnk.permute(0,2,1)[:,-1,:].reshape((B, 1, K, L)).permute(0, 3, 1, 2)
                    else:
                        output_ignnk = output_ignnk.permute(0,2,1).reshape((B, N, K, L)).permute(0, 3, 1, 2)

                if 'PriSTI' in models.keys():
                    with torch.no_grad():
                        start = time.time()
                        output_pristi = models['PriSTI'].evaluate(test_batch, nsample)
                        end = time.time()
                        # print(f"time taken for pristi: {(end-start)/batch_size}")
                        # exit()
                        samples_pristi, c_target_pristi, eval_points_pristi, observed_points_pristi, _, gt_intact, _, _, _, _ = output_pristi
                        samples_pristi = samples_pristi.permute(0, 1, 3, 2)
                        samples_pristi_mean = samples_pristi.mean(dim=1)
                        c_target_pristi = c_target_pristi.permute(0, 2, 1)  # (B,L,K)
                        eval_points_pristi = eval_points_pristi.permute(0, 2, 1)
                        observed_points_pristi = observed_points_pristi.permute(0, 2, 1)
                    
                if 'SADI' in models.keys():
                    with torch.no_grad():
                        output_sadi = models['SADI'].evaluate(test_batch, nsample)
                        if 'CSDI' not in models.keys():
                            samples_sadi, c_target, eval_points, observed_points, _, _, _, _ = output_sadi
                            c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                            eval_points = eval_points.permute(0, 2, 1)
                            observed_points = observed_points.permute(0, 2, 1)
                        else:
                            samples_sadi, _, _, _, _, _ = output_sadi
                        samples_sadi = samples_sadi.permute(0, 1, 3, 2)
                        samples_sadi_mean = samples_sadi.mean(dim=1)
                    
                if 'SAITS' in models.keys():
                    gt_intact = gt_intact.squeeze(axis=0)
                    saits_X = gt_intact #test_batch['obs_data_intact']
                    if batch_size == 1:
                        saits_X = saits_X.unsqueeze(0)
                    saits_output = models['SAITS'].impute({'X': saits_X})

                if 'KNN' in models.keys():
                    observed_values = test_batch['observed_data']
                    
                    
                    batch_locations = test_batch['spatial_info']
                    knn_output = None
                    if not is_neighbor:
                        for k in range(batch_locations.shape[0]):
                            locations = batch_locations[k]
                            knn = NearestNeighbors(n_neighbors=7)
                            knn.fit(locations[k])
                            neighbors = knn.kneighbors(test_batch['missing_data_loc'][k], n_neighbors=50)
                            neighbor_indices = neighbors[1][0].tolist()
                            distances = neighbors[0][0]
                            inverse_distances = 1/distances
                            knn_pred = np.sum(observed_values[k,:,neighbor_indices, :] * inverse_distances, axis=1) / np.sum(inverse_distances)
                            if knn_output is None:
                                knn_output = np.expand_dims(knn_pred, axis=0)
                            else:
                                knn_output = np.concatenate([knn_output, np.expand_dims(knn_pred, axis=0)], axis=0)
                    else:
                        distances = get_distances(batch_locations, test_batch['missing_data_loc'])
                        inverse_distances = 1/distances
                        knn_output = np.sum(observed_values[:,:,neighbor_indices, :] * inverse_distances, axis=2) / np.sum(inverse_distances)



                    # gt_intact = gt_intact.squeeze(axis=0)
                    # knn_X = gt_intact #test_batch['obs_data_intact']
                    # if batch_size == 1:
                    #     knn_X = knn_X.unsqueeze(0)
                    # knn_output = None
                    # for k in range(knn_X.shape[0]):
                    #     knn_pred = models['KNN'].transform(knn_X[k])
                    #     if knn_output is None:
                    #         knn_output = knn_pred
                    #     else:
                    #         knn_output = np.stack([knn_output, knn_pred], axis=0)

                if 'BRITS' in models.keys():
                    # brits_output = test_evaluate(models['BRITS'], f'json/json_eval_{dataset_name}', test_batch['observed_data'], c_target, observed_points, eval_points)
                    gt_intact = gt_intact.squeeze(axis=0)
                    brits_X = gt_intact #test_batch['obs_data_intact']
                    if batch_size == 1:
                        brits_X = brits_X.unsqueeze(0)
                    brits_output = models['BRITS'].impute({'X': brits_X})

                if 'MICE' in models.keys():
                    # print(f"gt_intact in MICE: {gt_intact.shape}")

                    if dataset_name == 'pm25':
                        path = "./data/pm25/pm25_meanstd.pk"
                        with open(path, "rb") as f:
                            train_mean, train_std = pickle.load(f)
                            train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
                            train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
                    elif dataset_name == 'electricity':
                        path_mean = "./data/Electricity/mean.npy"
                        path_std = "./data/Electricity/std.npy"
                        train_mean = np.load(path_mean)
                        train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
                        train_std = np.load(path_std)
                        train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
                    elif dataset_name == 'nasce':
                        path_mean = f"{filename[2]}_mean.npy"
                        path_std = f"{filename[2]}_std.npy"
                        train_mean = np.load(path_mean)
                        train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
                        train_std = np.load(path_std)
                        train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
                    elif dataset_name == 'awn':
                        path_mean = f"{filename[2]}_mean.npy"
                        path_std = f"{filename[2]}_std.npy"
                        train_mean = np.load(path_mean)
                        train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
                        train_std = np.load(path_std)
                        train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
                    elif dataset_name == 'metrla' or dataset_name == 'pemsbay' or dataset_name == 'synth':
                        path_mean = f"{filename[2]}_mean.npy"
                        path_std = f"{filename[2]}_std.npy"
                        train_mean = np.load(path_mean)
                        train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
                        train_std = np.load(path_std)
                        train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
                    else:
                        train_mean = torch.tensor(mean, dtype=torch.float32, device=device)
                        train_std = torch.tensor(std, dtype=torch.float32, device=device)
                    mice_X = (gt_intact - train_mean) / train_std
                    mice_X = mice_X.reshape(-1, n_steps, n_stations * n_features)
                    # mice_X = mice_X.squeeze(axis=0)
                    # mice_X = gt_intact
                    if batch_size == 1:
                        mice_X = mice_X.unsqueeze(0)
                    mice_output = None
                    for k in range(mice_X.shape[0]):
                        mice_pred = models['MICE'].transform(mice_X[k].cpu())
                        if mice_output is None:
                            mice_output = np.expand_dims(mice_pred, axis=0)
                        else:
                            # print(f"mice out: {mice_output.shape}\nmice pred: {mice_pred.shape}")
                            mice_pred = np.expand_dims(mice_pred, axis=0)
                            mice_output = np.concatenate([mice_output, mice_pred], axis=0)
                
                if unnormalize:
                    if dataset_name == 'pm25':
                        path = "./data/pm25/pm25_meanstd.pk"
                        with open(path, "rb") as f:
                            train_mean, train_std = pickle.load(f)
                            train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
                            train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
                    elif dataset_name == 'electricity':
                        path_mean = "./data/Electricity/mean.npy"
                        path_std = "./data/Electricity/std.npy"
                        train_mean = np.load(path_mean)
                        train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
                        train_std = np.load(path_std)
                        train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
                    elif dataset_name == 'nasce':
                        path_mean = f"{filename[2]}_mean.npy"
                        path_std = f"{filename[2]}_std.npy"
                        train_mean = np.load(path_mean)
                        train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
                        train_std = np.load(path_std)
                        train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
                    elif dataset_name == 'awn':
                        path_mean = f"{filename[2]}_mean.npy"
                        path_std = f"{filename[2]}_std.npy"
                        train_mean = np.load(path_mean)
                        train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
                        train_std = np.load(path_std)
                        train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
                    elif dataset_name == 'metrla' or dataset_name == 'pemsbay' or dataset_name == 'synth':
                        path_mean = f"{filename[2]}_mean.npy"
                        path_std = f"{filename[2]}_std.npy"
                        train_mean = np.load(path_mean)
                        train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
                        train_std = np.load(path_std)
                        train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
                    else:
                        train_mean = torch.tensor(mean, dtype=torch.float32, device=device)
                        train_std = torch.tensor(std, dtype=torch.float32, device=device)

                    if 'CSDI' in models.keys():
                        csdi_median = samples_median.values
                        csdi_median = csdi_median.reshape(-1, n_steps, n_stations, train_mean.shape[1])
                        samples_median_csdi = (csdi_median * train_std) + train_mean
                        samples_median_csdi = samples_median_csdi.reshape(-1, n_steps, n_stations * train_mean.shape[1])
                    
                    if 'GP' in models.keys():
                        pred_gp = pred_gp.reshape(-1, n_steps, 1, train_mean.shape[1])
                        pred_gp = (pred_gp * train_std) + train_mean
                        pred_gp = pred_gp.reshape(-1, n_steps, 1 * train_mean.shape[1])
                    
                    if 'DK' in models.keys():
                        dk_output = dk_output.reshape(-1, n_steps, 1, train_mean.shape[1])
                        dk_output = (dk_output * train_std) + train_mean
                        dk_output = dk_output.reshape(-1, n_steps, 1 * train_mean.shape[1])

                    if 'DynaSTI-Orig' in models.keys():
                        # print(f"mean: {train_mean.shape}")
                        if is_separate:
                            if missing_dims != -1:
                                samples_diff_saits_orig_mean = samples_diff_saits_orig_mean.reshape(-1, n_steps, missing_dims, train_mean.shape[1])
                                samples_diff_saits_orig_mean = (samples_diff_saits_orig_mean * train_std) + train_mean
                                samples_diff_saits_orig_mean = samples_diff_saits_orig_mean.reshape(-1, n_steps, missing_dims * train_mean.shape[1])
                            else:
                                samples_diff_saits_orig_mean = samples_diff_saits_orig_mean.reshape(-1, n_steps, 1, train_mean.shape[1])
                                samples_diff_saits_orig_mean = (samples_diff_saits_orig_mean * train_std) + train_mean
                                samples_diff_saits_orig_mean = samples_diff_saits_orig_mean.reshape(-1, n_steps, 1 * train_mean.shape[1])
                        else:
                            samples_diff_saits_orig_mean = samples_diff_saits_orig_mean.reshape(-1, n_steps, n_stations, train_mean.shape[1])
                            samples_diff_saits_orig_mean = (samples_diff_saits_orig_mean * train_std) + train_mean
                            samples_diff_saits_orig_mean = samples_diff_saits_orig_mean.reshape(-1, n_steps, n_stations * train_mean.shape[1])

                    if 'SPAT-SADI' in models.keys():
                        # print(f"mean: {train_mean.shape}")
                        if is_separate:
                            if missing_dims != -1:
                                samples_diff_saits_mean = samples_diff_saits_mean.reshape(-1, n_steps, missing_dims, train_mean.shape[1])
                                samples_diff_saits_mean = (samples_diff_saits_mean * train_std) + train_mean
                                samples_diff_saits_mean = samples_diff_saits_mean.reshape(-1, n_steps, missing_dims * train_mean.shape[1])
                            else:
                                samples_diff_saits_mean = samples_diff_saits_mean.reshape(-1, n_steps, 1, train_mean.shape[1])
                                samples_diff_saits_mean = (samples_diff_saits_mean * train_std) + train_mean
                                samples_diff_saits_mean = samples_diff_saits_mean.reshape(-1, n_steps, 1 * train_mean.shape[1])
                        else:
                            samples_diff_saits_mean = samples_diff_saits_mean.reshape(-1, n_steps, n_stations, train_mean.shape[1])
                            samples_diff_saits_mean = (samples_diff_saits_mean * train_std) + train_mean
                            samples_diff_saits_mean = samples_diff_saits_mean.reshape(-1, n_steps, n_stations * train_mean.shape[1])


                    if 'Interpolation'  in models.keys():
                        output_interpolation = output_interpolation.rehsape(-1, n_steps, 1, train_mean.shape[1])
                        output_interpolation = (output_interpolation * train_std) + train_mean
                        output_interpolation = output_interpolation.rehsape(-1, n_steps, 1 * train_mean.shape[1])

                    if 'IGNNK' in models.keys():
                        if missing_dims != -1:
                            output_ignnk = output_ignnk.reshape(-1, n_steps, missing_dims, train_mean.shape[1])
                            output_ignnk = (output_ignnk * train_std) + train_mean
                            output_ignnk = output_ignnk.reshape(-1, n_steps, missing_dims * train_mean.shape[1])
                        else:
                            output_ignnk = output_ignnk.reshape(-1, n_steps, 1, train_mean.shape[1])
                            output_ignnk = (output_ignnk * train_std) + train_mean
                            output_ignnk = output_ignnk.reshape(-1, n_steps, 1 * train_mean.shape[1])

                    if 'PriSTI' in models.keys():
                        # print(f"mean: {train_mean.shape}")
                        samples_pristi_mean = samples_pristi_mean.reshape(-1, n_steps, total_locations, train_mean.shape[1])
                        samples_pristi_mean = (samples_pristi_mean * train_std) + train_mean
                        samples_pristi_mean = samples_pristi_mean.reshape(-1, n_steps, total_locations * train_mean.shape[1])

                    if 'SADI' in models.keys():
                        samples_sadi_mean = (samples_sadi_mean * train_std) + train_mean
                    if 'SAITS' in models.keys():
                        saits_output = (torch.tensor(saits_output, device=device) * train_std) + train_mean
                    if 'MICE' in models.keys():
                        mice_output = (torch.tensor(mice_output, device=device) * train_std) + train_mean
                    if 'BRITS' in models.keys():
                        brits_output = (torch.tensor(brits_output, device=device) * train_std) + train_mean
                    
                    if not data:
                        if 'MEAN' in models.keys():
                            mean_start = time.time()
                            if is_separate:
                                # print(f"missing data shape: {missing_data.shape}, train mean: {train_mean.shape}")
                                # observed_values = test_batch['observed_data']

                                if missing_dims != -1:
                                    missing_data_temp = missing_data.reshape(-1, n_steps, missing_dims, train_mean.shape[1])
                                    missing_data_mask_temp = missing_data_mask.reshape(-1, n_steps, missing_dims, train_mean.shape[1])
                                    missing_data_temp = (missing_data_temp * train_std) + train_mean
                                    missing_data = missing_data_temp.reshape(-1, n_steps, missing_dims * train_mean.shape[1])

                                else:
                                    missing_data_temp = missing_data.reshape(-1, n_steps, 1, train_mean.shape[1])
                                    missing_data_mask_temp = missing_data_mask.reshape(-1, n_steps, 1, train_mean.shape[1])
                                    missing_data_temp = (missing_data_temp * train_std) + train_mean
                                    missing_data = missing_data_temp.reshape(-1, n_steps, 1 * train_mean.shape[1])

                                gt_intact = gt_intact.permute((0, 1, 3, 2)) # B, L, K, N
                                # print(f"gt intact 1: {gt_intact}")
                                # batch_locations = test_batch['spatial_info']
                                # distances = get_distances(batch_locations, test_batch['missing_data_loc'])
                                # inverse_distances = 1.0 / (distances+1e-6)
                                # inverse_distances = inverse_distances.unsqueeze(1).unsqueeze(-1) # B, 1, N, 1
                                # # knn_output = np.sum(observed_values * inverse_distances, axis=2) / np.sum(inverse_distances)
                                # gt_intact_temp = torch.nan_to_num(gt_intact)
                                
                                # station_mean = (gt_intact_temp * inverse_distances).sum(dim=2, keepdim=True) / inverse_distances.sum(dim=2, keepdim=True)

                                station_mean = torch.nanmean(gt_intact, dim=-2).unsqueeze(-2)
                                if torch.isnan(station_mean).sum() > 0:
                                    station_mean = torch.nan_to_num(station_mean)
                                rmse_mean = ((missing_data_temp-station_mean) * missing_data_mask_temp) ** 2

                                rmse_mean = rmse_mean.sum().item() / missing_data_mask_temp.sum().item()
                                if np.isnan(rmse_mean):
                                    print(f"rmse mean nan and station mean 1: {station_mean}")
                                mean_avg_rmse += rmse_mean ** 0.5
                                
                            else:    
                                c_target = c_target.reshape(-1, n_steps, n_stations, train_mean.shape[1]) #.transpose(2,3)
                                eval_points_mean = eval_points.reshape(-1, n_steps, n_stations, train_mean.shape[1])
                                
                                c_target = (c_target * train_std) + train_mean

                                gt_intact = gt_intact.permute((0, 1, 3, 2)) # B, L, K, N
                                station_mean = torch.nanmean(gt_intact, dim=-1).unsqueeze(-2)
                                
                                rmse_mean = ((c_target-station_mean) * eval_points_mean) ** 2
                                rmse_mean = rmse_mean.sum().item() / eval_points_mean.sum().item()
                                mean_avg_rmse += rmse_mean ** 0.5
                                c_target = c_target.reshape(-1, n_steps, n_stations * train_mean.shape[1]) #.transpose(1,2)
                            mean_end = time.time()
                            # print(f"MEAN time: {mean_end-mean_start}s")
                        
                        # exit()
                else:
                    if 'CSDI' in models.keys():
                        samples_median_csdi = samples_median.values

                    if dataset_name == 'pm25':
                        path = "./data/pm25/pm25_meanstd.pk"
                        with open(path, "rb") as f:
                            train_mean, train_std = pickle.load(f)
                            train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
                            train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
                    elif dataset_name == 'electricity':
                        path_mean = "./data/Electricity/mean.npy"
                        path_std = "./data/Electricity/std.npy"
                        train_mean = np.load(path_mean)
                        train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
                        train_std = np.load(path_std)
                        train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
                    elif dataset_name == 'awn':
                        path_mean = f"{filename[2]}_mean.npy"
                        path_std = f"{filename[2]}_std.npy"
                        train_mean = np.load(path_mean)
                        train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
                        train_std = np.load(path_std)
                        train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
                    elif dataset_name == 'nasce':
                        path_mean = f"{filename[2]}_mean.npy"
                        path_std = f"{filename[2]}_std.npy"
                        train_mean = np.load(path_mean)
                        train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
                        train_std = np.load(path_std)
                        train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
                    elif dataset_name == 'metrla' or dataset_name == 'pemsbay' or dataset_name == 'synth':
                        path_mean = f"{filename[2]}_mean.npy"
                        path_std = f"{filename[2]}_std.npy"
                        train_mean = np.load(path_mean)
                        train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
                        train_std = np.load(path_std)
                        train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
                    else:
                        train_mean = torch.tensor(mean, dtype=torch.float32, device=device)
                        train_std = torch.tensor(std, dtype=torch.float32, device=device)

                    
                    # c_target = (c_target * train_std) + train_mean
                    # batch_locations = test_batch['spatial_info'].to(device)
                    # distances = get_distances(batch_locations, test_batch['missing_data_loc'].to(device))
                    # inverse_distances = 1.0 / (distances+1e-6)
                    # inverse_distances = inverse_distances.unsqueeze(1).unsqueeze(-1) # B, 1, N, 1
                    # # knn_output = np.sum(observed_values * inverse_distances, axis=2) / np.sum(inverse_distances)
                    # gt_intact_temp = torch.nan_to_num(gt_intact)
                    
                    # station_mean = (gt_intact_temp * inverse_distances).sum(dim=2, keepdim=True) / inverse_distances.sum(dim=2, keepdim=True)
                    if "MEAN" in models.keys():
                        mean_start = time.time()
                        gt_intact = gt_intact.permute((0, 1, 3, 2)) # B, L, K, N
                        station_mean = torch.nanmean(gt_intact, dim=-1).unsqueeze(-2)
                        if torch.isnan(station_mean).sum() > 0:
                            station_mean = torch.nan_to_num(station_mean)
                        station_mean = (station_mean - train_mean) / train_std
                        

                        if is_separate:
                            if missing_dims != -1:
                                missing_data_temp = missing_data.reshape(-1, n_steps, missing_dims, n_features)
                                missing_data_mask_temp = missing_data_mask.reshape(-1, n_steps, missing_dims, n_features)
                            else:
                                missing_data_temp = missing_data.reshape(-1, n_steps, 1, n_features)
                                missing_data_mask_temp = missing_data_mask.reshape(-1, n_steps, 1, n_features)
                            # print(f"missing_data: {missing_data_temp.shape}, station mean: {station_mean.shape}")
                            rmse_mean = ((missing_data_temp-station_mean) * missing_data_mask_temp) ** 2
                            # if missing_data_mask_temp.sum().item() == 0:
                            #     continue
                            rmse_mean = rmse_mean.sum().item() / missing_data_mask_temp.sum().item()
                        else:
                            c_target = c_target.reshape(-1, n_steps, n_stations, train_mean.shape[1]) #.transpose(2,3)
                            eval_points_mean = eval_points.reshape(-1, n_steps, n_stations, train_mean.shape[1])
                            rmse_mean = ((c_target-station_mean) * eval_points_mean) ** 2
                            rmse_mean = rmse_mean.sum().item() / eval_points.sum().item()
                        mean_end = time.time()
                        # print(f"MEAN time: {mean_end-mean_start}s")
                        if np.isnan(rmse_mean):
                            print(f"rmse mean nan and station mean 2: {station_mean}")
                        mean_avg_rmse += rmse_mean ** 0.5
                    if 'Spat-SADI' in models.keys():
                        c_target = c_target.reshape(-1, n_steps, n_stations * train_mean.shape[1]) 
                if data:
                    # if not spatial_slider:
                    samples_size = samples.shape[1] if 'CSDI' in models.keys() else samples_diff_saits.shape[1]
                    # else:
                    #     samples_size = samples.shape[0] if 'CSDI' in models.keys() else samples_diff_saits.shape[0]
                    if is_separate:
                        if missing_dims != -1:
                            missing_data_temp = missing_data.reshape(-1, n_steps, missing_dims, train_mean.shape[1])
                            missing_data_temp = (missing_data_temp * train_std) + train_mean
                            missing_data = missing_data_temp.reshape(-1, n_steps, missing_dims * train_mean.shape[1])
                        else:
                            missing_data_temp = missing_data.reshape(-1, n_steps, 1, train_mean.shape[1])
                            missing_data_temp = (missing_data_temp * train_std) + train_mean
                            missing_data = missing_data_temp.reshape(-1, n_steps, 1 * train_mean.shape[1])
                    for idx in range(samples_size):

                        if 'CSDI' in models.keys():
                            # if not spatial_slider:
                            samples_csdi_temp = samples[0, idx].reshape(-1, n_steps, n_stations, train_mean.shape[1])
                            # else:
                            #     samples_csdi_temp = samples[idx].reshape(-1, n_steps, n_stations, train_mean.shape[1])
                            samples_csdi_temp = (samples_csdi_temp * train_std) + train_mean
                            # if spatial_slider:
                            #     samples[idx] = samples_csdi_temp.reshape(-1, n_steps, n_stations * train_mean.shape[1])
                            # else:
                            samples[0, idx] = samples_csdi_temp.reshape(-1, n_steps, n_stations * train_mean.shape[1])
                        
                        if 'DynaSTI-Orig' in models.keys():
                            if is_separate:
                                if missing_dims != -1:
                                    samples_diff_saits_orig_temp = samples_diff_saits_orig[0, idx].reshape(-1, n_steps, missing_dims, train_mean.shape[1])
                                    samples_diff_saits_orig_temp = (samples_diff_saits_orig_temp * train_std) + train_mean
                                    samples_diff_saits_orig[0, idx] = samples_diff_saits_orig_temp.reshape(-1, n_steps, missing_dims * train_mean.shape[1])
                                else:
                                    samples_diff_saits_orig_temp = samples_diff_saits_orig[0, idx].reshape(-1, n_steps, 1, train_mean.shape[1])
                                    samples_diff_saits_orig_temp = (samples_diff_saits_orig_temp * train_std) + train_mean
                                    samples_diff_saits_orig[0, idx] = samples_diff_saits_orig_temp.reshape(-1, n_steps, 1 * train_mean.shape[1])
                            else:
                                samples_diff_saits_orig_temp = samples_diff_saits_orig[0, idx].reshape(-1, n_steps, n_stations, train_mean.shape[1])
                                # samples_diff_saits_temp = samples_diff_saits[idx].reshape(-1, n_steps, n_stations, train_mean.shape[1])
                                samples_diff_saits_orig_temp = (samples_diff_saits_orig_temp * train_std) + train_mean
                                # samples_diff_saits[idx] = samples_diff_saits_temp.reshape(-1, n_steps, n_stations * train_mean.shape[1])
                                samples_diff_saits_orig[0, idx] = samples_diff_saits_orig_temp.reshape(-1, n_steps, n_stations * train_mean.shape[1])
                        
                        
                        if 'SPAT-SADI' in models.keys():
                            if is_separate:
                                # if not spatial_slider:
                                if missing_dims != -1:
                                    samples_diff_saits_temp = samples_diff_saits[0, idx].reshape(-1, n_steps, missing_dims, train_mean.shape[1])
                                    samples_diff_saits_temp = (samples_diff_saits_temp * train_std) + train_mean
                                    samples_diff_saits[0, idx] = samples_diff_saits_temp.reshape(-1, n_steps, missing_dims * train_mean.shape[1])
                                
                                else:
                                    samples_diff_saits_temp = samples_diff_saits[0, idx].reshape(-1, n_steps, 1, train_mean.shape[1])
                                # else:
                                #     samples_diff_saits_temp = samples_diff_saits[idx].reshape(-1, n_steps, 1, train_mean.shape[1])
                                    samples_diff_saits_temp = (samples_diff_saits_temp * train_std) + train_mean
                                # if spatial_slider:
                                # samples_diff_saits[idx] = samples_diff_saits_temp.reshape(-1, n_steps, 1 * train_mean.shape[1])
                                # else:
                                    samples_diff_saits[0, idx] = samples_diff_saits_temp.reshape(-1, n_steps, 1 * train_mean.shape[1])
                            else:
                                samples_diff_saits_temp = samples_diff_saits[0, idx].reshape(-1, n_steps, n_stations, train_mean.shape[1])
                                # samples_diff_saits_temp = samples_diff_saits[idx].reshape(-1, n_steps, n_stations, train_mean.shape[1])
                                samples_diff_saits_temp = (samples_diff_saits_temp * train_std) + train_mean
                                # samples_diff_saits[idx] = samples_diff_saits_temp.reshape(-1, n_steps, n_stations * train_mean.shape[1])
                                samples_diff_saits[0, idx] = samples_diff_saits_temp.reshape(-1, n_steps, n_stations * train_mean.shape[1])
                        
                        # if 'GP' in models.keys():
                        #     pred_gp_temp = pred_gp[0, idx].reshape(-1, n_steps, 1, train_mean.shape[1])
                        #     pred_gp_temp = (pred_gp_temp * train_std) + train_mean
                        #     pred_gp[0, idx] = 

                        if 'SADI' in models.keys():
                            # if not spatial_slider:
                            samples_sadi_temp = samples_sadi[0, idx].reshape(-1, n_steps, n_stations, train_mean.shape[1])
                            # else:
                            #     samples_sadi_temp = samples_sadi[idx].reshape(-1, n_steps, n_stations, train_mean.shape[1])
                            samples_sadi_temp = (samples_sadi_temp * train_std) + train_mean
                            # if not spatial_slider:
                            samples_sadi[0, idx] = samples_pristi_temp.reshape(-1, n_steps, n_stations * train_mean.shape[1])
                            # else:
                            #     samples_sadi[idx] = samples_pristi_temp.reshape(-1, n_steps, n_stations * train_mean.shape[1])
                        if 'PriSTI' in models.keys():
                            # if not spatial_slider:
                            samples_pristi_temp = samples_pristi[0, idx].reshape(-1, n_steps, total_locations, train_mean.shape[1])
                            # else:
                            #     samples_pristi_temp = samples_pristi[idx].reshape(-1, n_steps, total_locations, train_mean.shape[1])
                            samples_pristi_temp = (samples_pristi_temp * train_std) + train_mean
                            # if spatial_slider:
                            #     samples_pristi[idx] = samples_pristi_temp.reshape(-1, n_steps, total_locations * train_mean.shape[1])
                            # else:
                            samples_pristi[0, idx] = samples_pristi_temp.reshape(-1, n_steps, total_locations * train_mean.shape[1])
                    
                    if is_separate:
                        station_mean = torch.nanmean(gt_intact, dim=-2).unsqueeze(-2)
                        # if not spatial_slider:
                        results_data[j] = {
                            'target_mask': missing_data_mask[0, :, :].cpu().numpy(),
                            'target_mask_pristi': eval_points_pristi[0, :, :].cpu().numpy() if 'PriSTI' in models.keys() else None,
                            'observed_mask_pristi': observed_points_pristi[0, :, :].cpu().numpy() if 'PriSTI' in models.keys() else None,
                            'target': missing_data[0, :, :].cpu().numpy(),
                            'observed_mask': observed_points[0, :, :].cpu().numpy(),
                            'mean': station_mean.squeeze(-2)[0,:,:].cpu().numpy(),
                            'missing_loc': missing_data_loc.cpu().numpy(),
                            'attn_spat_mean': attn_spat_mean.cpu().numpy(),
                            'attn_spat_std': attn_spat_std.cpu().numpy(),
                            'spatial_loc': spatial_loc.cpu().numpy()
                        }
                        
                        # else:
                        #     results_data[j] = {
                        #         'target_mask': missing_data_mask.cpu().numpy(),
                        #         'target_mask_pristi': eval_points_pristi.cpu().numpy() if 'PriSTI' in models.keys() else None,
                        #         'observed_mask_pristi': observed_points_pristi.cpu().numpy() if 'PriSTI' in models.keys() else None,
                        #         'target': missing_data.cpu().numpy(),
                        #         'observed_mask': observed_points.cpu().numpy(),
                        #         'mean': station_mean.squeeze(-3).cpu().numpy(),
                        #         'missing_loc': missing_data_loc.numpy()
                        #     }
                    else:
                        # if not spatial_slider:
                        results_data[j] = {
                            'target_mask': eval_points[0, :, :].cpu().numpy(),
                            'target_mask_pristi': eval_points_pristi[0, :, :].cpu().numpy() if 'PriSTI' in models.keys() else None,
                            'observed_mask_pristi': observed_points_pristi[0, :, :].cpu().numpy() if 'PriSTI' in models.keys() else None,
                            'target': c_target[0, :, :].cpu().numpy(),
                            'observed_mask': observed_points[0, :, :].cpu().numpy(),
                            'attn_spat_mean': attn_spat_mean.cpu().numpy(),
                            'attn_spat_std': attn_spat_std.cpu().numpy(),
                            'spatial_loc': spatial_loc.cpu().numpy()
                        }
                        # else:
                        #     results_data[j] = {
                        #         'target_mask': eval_points.cpu().numpy(),
                        #         'target_mask_pristi': eval_points_pristi.cpu().numpy() if 'PriSTI' in models.keys() else None,
                        #         'observed_mask_pristi': observed_points_pristi.cpu().numpy() if 'PriSTI' in models.keys() else None,
                        #         'target': c_target.cpu().numpy(),
                        #         'observed_mask': observed_points.cpu().numpy()
                        #     }
                    if 'CSDI' in models.keys():
                        # if not spatial_slider:
                        results_data[j]['csdi_median'] = samples_median_csdi[0, :, :].cpu().numpy()
                        results_data[j]['csdi_samples'] = samples[0].cpu().numpy()
                        # else:
                        #     results_data[j]['csdi_median'] = samples_median_csdi.cpu().numpy()
                        #     results_data[j]['csdi_samples'] = samples.cpu().numpy()

                    if 'IGNNK' in models.keys():
                        # if not spatial_slider:
                        results_data[j]['ignnk'] = output_ignnk[0,:,:].detach().cpu().numpy()
                        # else:
                        #     results_data[j]['ignnk'] = output_ignnk.detach().cpu().numpy()

                    if 'OKriging' in models.keys():
                        results_data[j]['kriging_mean'] = kriging_mean
                        results_data[j]['kriging_std'] = kriging_std

                    if 'Interpolation' in models.keys():
                        # if not spatial_slider:
                        results_data[j]['interpolation'] = output_interpolation[0, :, :].cpu().numpy()
                        # else:
                        #     results_data[j]['interpolation'] = output_interpolation.cpu().numpy()
                    if 'DynaSTI-Orig' in models.keys():
                        # if not spatial_slider:
                        results_data[j]['spat-sadi_orig_mean'] = samples_diff_saits_orig_mean[0, :, :].cpu().numpy()
                        results_data[j]['spat-sadi_samples_orig'] = samples_diff_saits_orig[0].cpu().numpy()

                    if 'SPAT-SADI' in models.keys():
                        # if not spatial_slider:
                        results_data[j]['spat-sadi_mean'] = samples_diff_saits_mean[0, :, :].cpu().numpy()
                        results_data[j]['spat-sadi_samples'] = samples_diff_saits[0].cpu().numpy()
                        # else:
                        #     results_data[j]['spat-sadi_mean'] = samples_diff_saits_mean.cpu().numpy()
                        #     results_data[j]['spat-sadi_samples'] = samples_diff_saits.cpu().numpy()
                    if 'GP' in models.keys():
                        results_data[j]['pred_gp_mean'] = pred_gp[0, :, :].cpu().numpy()
                        # results_data[j]['pred_gp_cov'] = pred_cov[0, :, :].cpu().numpy()

                    if 'SADI' in models.keys():
                        # if not spatial_slider:
                        results_data[j]['sadi_mean'] = samples_sadi_mean[0, :, :].cpu().numpy()
                        results_data[j]['sadi_samples'] = samples_sadi[0].cpu().numpy()
                        # else:
                        #     results_data[j]['sadi_mean'] = samples_sadi_mean.cpu().numpy()
                        #     results_data[j]['sadi_samples'] = samples_sadi.cpu().numpy()
                    if 'PriSTI' in models.keys():
                        # if not spatial_slider:
                        results_data[j]['pristi_mean'] = samples_pristi_mean[0, :, :].cpu().numpy()
                        results_data[j]['pristi_samples'] = samples_pristi[0].cpu().numpy()
                        # else:
                        #     results_data[j]['pristi_mean'] = samples_pristi_mean.cpu().numpy()
                        #     results_data[j]['pristi_samples'] = samples_pristi.cpu().numpy()

                    if 'SAITS' in models.keys():
                        # if not spatial_slider:
                        results_data[j]['saits'] = saits_output[0, :, :].cpu().numpy()
                        # else:
                        #     results_data[j]['saits'] = saits_output.cpu().numpy()

                    if 'KNN' in models.keys():
                        # if not spatial_slider:
                        results_data[j]['knn'] = knn_output[0, :, :].cpu().numpy()
                        # else:
                        #     results_data[j]['knn'] = knn_output.cpu().numpy()

                    if 'MICE' in models.keys():
                        # if not spatial_slider:
                        results_data[j]['mice'] = mice_output[0, :, :].cpu().numpy()
                        # else:
                        #     results_data[j]['mice'] = mice_output.cpu().numpy()
                    
                    if 'BRITS' in models.keys():
                        # if not spatial_slider:
                        results_data[j]['brits'] = brits_output[0, :, :].cpu().numpy()
                        # else:
                        #     results_data[j]['brits'] = brits_output.cpu().numpy()
                else:
                    ###### CSDI ######
                    if 'CSDI' in models.keys():
                        rmse_csdi = ((samples_median_csdi - c_target) * eval_points) ** 2
                        rmse_csdi = rmse_csdi.sum().item() / eval_points.sum().item()
                        csdi_rmse_avg += rmse_csdi ** 0.5

                        # mae_csdi = torch.abs((samples_median_csdi - c_target) * eval_points)
                        # mae_csdi = mae_csdi.sum().item() / eval_points.sum().item()
                        # csdi_mae_avg += mae_csdi

                        csdi_crps = calc_quantile_CRPS(c_target, samples, eval_points, 0, 1)
                        csdi_crps_avg += csdi_crps

                    if 'OKriging' in models.keys():
                        # print(f"kriging mean: {kriging_mean.shape}, c_target: {c_target.shape}, eval points: {eval_points.shape}")
                        missing_data_temp = missing_data.reshape(-1, n_steps, 1, n_features)
                        missing_data_mask_temp = missing_data_mask.reshape(-1, n_steps, 1, n_features)
                        kriging_rmse = ((torch.tensor(kriging_mean, device=device) - missing_data_temp) * missing_data_mask_temp) ** 2
                        kriging_rmse = kriging_rmse.sum().item() / missing_data_mask_temp.sum().item()
                        kriging_rmse_avg += kriging_rmse ** 0.5

                    if 'Interpolation' in models.keys():
                        missing_data_temp = missing_data.reshape(-1, n_steps, 1, n_features)
                        missing_data_mask_temp = missing_data_mask.reshape(-1, n_steps, 1, n_features)
                        interp_rmse = ((torch.tensor(output_interpolation, device=device) - missing_data_temp) * missing_data_mask_temp) ** 2
                        interp_rmse = interp_rmse.sum().item() / missing_data_mask_temp.sum().itm()
                        interp_rmse_avg += interp_rmse ** 0.5

                    ###### DiffSAITS ######
                    if 'DynaSTI-Orig' in models.keys():
                        # print(f"sample mean: {samples_diff_saits_mean.shape}\nc_target: {c_target.shape}\neval_points: {eval_points.shape}")
                        if is_separate:
                            # print(f"spat-sadi mean: {samples_diff_saits_mean.shape}, missing data: {missing_data.shape}, missing mask: {missing_data_mask.shape}")
                            rmse_diff_saits_orig = ((samples_diff_saits_orig_mean - missing_data) * missing_data_mask) ** 2
                            rmse_diff_saits_orig = rmse_diff_saits_orig.sum().item() / missing_data_mask.sum().item()
                        else:
                            rmse_diff_saits_orig = ((samples_diff_saits_orig_mean - c_target) * eval_points) ** 2
                            rmse_diff_saits_orig = rmse_diff_saits_orig.sum().item() / eval_points.sum().item()
                        diffsaits_rmse_avg_orig += rmse_diff_saits_orig ** 0.5
                        if is_separate:
                            diff_saits_crps_orig = calc_quantile_CRPS(missing_data, samples_diff_saits_orig, missing_data_mask, 0, 1)
                        else:
                            diff_saits_crps_orig = calc_quantile_CRPS(c_target, samples_diff_saits_orig, eval_points, 0, 1)
                        diffsaits_crps_avg_orig += diff_saits_crps_orig

                    if 'SPAT-SADI' in models.keys():
                        # print(f"sample mean: {samples_diff_saits_mean.shape}\nc_target: {c_target.shape}\neval_points: {eval_points.shape}")
                        # if is_separate and missing_dims != -1:
                        #     samples_diff_saits_mean = samples_diff_saits_mean.reshape(-1, n_steps, missing_dims, train_mean.shape[1])
                        
                        if is_separate:
                            # print(f"spat-sadi mean: {samples_diff_saits_mean.shape}, missing data: {missing_data.shape}, missing mask: {missing_data_mask.shape}")
                            
                            rmse_diff_saits = ((samples_diff_saits_mean - missing_data) * missing_data_mask) ** 2
                            rmse_diff_saits = rmse_diff_saits.sum().item() / missing_data_mask.sum().item()
                        else:
                            rmse_diff_saits = ((samples_diff_saits_mean - c_target) * eval_points) ** 2
                            rmse_diff_saits = rmse_diff_saits.sum().item() / eval_points.sum().item()
                        diffsaits_rmse_avg += rmse_diff_saits ** 0.5
                        if is_separate:
                            diff_saits_crps = calc_quantile_CRPS(missing_data, samples_diff_saits, missing_data_mask, 0, 1)
                            diffsaits_mis = calculate_mis(samples_diff_saits, missing_data, alpha=0.05)
                        else:
                            diff_saits_crps = calc_quantile_CRPS(c_target, samples_diff_saits, eval_points, 0, 1)
                        diffsaits_crps_avg += diff_saits_crps
                        diffsaits_mis_avg += diffsaits_mis
                    
                    if 'GP' in models.keys():
                        # print(f"pred_gp: {pred_gp.shape}, missing_data: {missing_data.shape}, missing_data_mask: {missing_data_mask.shape}")
                        rmse_gp = ((pred_gp - missing_data) * missing_data_mask) ** 2
                        rmse_gp = rmse_gp.sum().item() / missing_data_mask.sum().item()
                        gp_rmse_avg += rmse_gp ** 0.5

                    if 'DK' in models.keys():
                        rmse_dk = ((dk_output - missing_data) * missing_data_mask) ** 2
                        rmse_dk = rmse_dk.sum().item() / missing_data_mask.sum().item()
                        dk_rmse_avg += rmse_dk ** 0.5

                    if 'IGNNK' in models.keys():
                        if is_separate:
                            # print(f"Out ignnk: {output_ignnk.shape}, missing data: {missing_data.shape}, mask: {missing_data_mask.shape}")
                            if is_multi:
                                output_ignnk = output_ignnk.reshape((output_ignnk.shape[0], output_ignnk.shape[1], output_ignnk.shape[2] * output_ignnk.shape[3]))
                            else:
                                output_ignnk = output_ignnk.reshape((output_ignnk.shape[0], output_ignnk.shape[1], output_ignnk.shape[3]))
                            rmse_ignnk = ((output_ignnk - missing_data) * missing_data_mask) ** 2
                            rmse_ignnk = rmse_ignnk.sum().item() / missing_data_mask.sum().item()
                        else:
                            rmse_ignnk = ((output_ignnk - c_target) * eval_points) ** 2
                            rmse_ignnk = rmse_ignnk.sum().item() / eval_points.sum().item()
                        ignnk_rmse_avg += rmse_ignnk ** 0.5

                    if 'SADI' in models.keys():
                        rmse_sadi = ((samples_sadi_mean - c_target) * eval_points) ** 2
                        rmse_sadi = rmse_sadi.sum().item() / eval_points.sum().item()
                        sadi_rmse_avg += rmse_sadi ** 0.5

                        sadi_crps = calc_quantile_CRPS(c_target, samples_sadi, eval_points, 0, 1)
                        sadi_crps_avg += sadi_crps

                    if 'PriSTI' in models.keys():
                        rmse_pristi = ((samples_pristi_mean - c_target_pristi) * eval_points_pristi) ** 2
                        rmse_pristi = rmse_pristi.sum().item() / eval_points_pristi.sum().item()
                        pristi_rmse_avg += rmse_pristi ** 0.5

                        pristi_crps = calc_quantile_CRPS(c_target_pristi, samples_pristi, eval_points_pristi, 0, 1)
                        pristi_crps_avg += pristi_crps

                        pristi_mis = calculate_mis(samples_pristi, c_target_pristi, alpha=0.05)
                        pristi_mis_avg += pristi_mis

                    ###### SAITS ######
                    if 'SAITS' in models.keys():
                        rmse_saits = ((torch.tensor(saits_output, device=device)- c_target) * eval_points) ** 2
                        rmse_saits = rmse_saits.sum().item() / eval_points.sum().item()
                        saits_rmse_avg += rmse_saits ** 0.5
                    
                        mae_saits = torch.abs((torch.tensor(saits_output, device=device)- c_target) * eval_points)
                        mae_saits = mae_saits.sum().item() / eval_points.sum().item()
                        saits_mae_avg += mae_saits

                    ###### KNN ######
                    if 'KNN' in models.keys():
                        rmse_knn = ((torch.tensor(knn_output, device=device)- c_target) * eval_points) ** 2
                        rmse_knn = rmse_knn.sum().item() / eval_points.sum().item()
                        knn_rmse_avg += rmse_knn ** 0.5

                    ###### MICE ######
                    if 'MICE' in models.keys():
                        rmse_mice = ((torch.tensor(mice_output, device=device) - c_target) * eval_points) ** 2
                        rmse_mice = rmse_mice.sum().item() / eval_points.sum().item()
                        mice_rmse_avg += rmse_mice ** 0.5

                    ###### BRITS ######
                    if 'BRITS' in models.keys():
                        rmse_brits = ((torch.tensor(brits_output, device=device) - c_target) * eval_points) ** 2
                        rmse_brits = rmse_brits.sum().item() / eval_points.sum().item()
                        brits_rmse_avg += rmse_brits ** 0.5
                total_batch += 1
                # exit()
        if not data:
            # print(f"batch size: {batch_size}, trial miss: {trial_miss}")
            # batch_size = j
            if 'CSDI' in models.keys():
                results_trials_mse['csdi'][trial] = csdi_rmse_avg / total_batch # (batch_size - trial_miss)
                results_mse['csdi'] += csdi_rmse_avg / total_batch # (batch_size - trial_miss)
                # results_trials_mae['csdi'][trial] = csdi_mae_avg / batch_size
                # results_mae['csdi'] += csdi_mae_avg / batch_size
                results_crps['csdi_trials'][trial] = csdi_crps_avg / total_batch # (batch_size - trial_miss)
                results_crps['csdi'] += csdi_crps_avg / total_batch # (batch_size - trial_miss)

                

            if 'OKriging' in models.keys():
                results_trials_mse['kriging'][trial] = kriging_rmse_avg / total_batch #  (batch_size - trial_miss)
                results_mse['kriging'] += kriging_rmse_avg / total_batch #  (batch_size - trial_miss)

            if 'Interpolation' in models.keys():
                results_trials_mse['interpolation'][trial] = interp_rmse_avg / total_batch #  (batch_size - trial_miss)
                results_mse['interpolation'] += interp_rmse_avg / total_batch #  (batch_size - trial_miss)

            if 'DynaSTI-Orig' in models.keys():
                results_trials_mse['spat-sadi-orig'][trial] = diffsaits_rmse_avg_orig / total_batch #  (batch_size - trial_miss)
                results_mse['spat-sadi-orig'] += diffsaits_rmse_avg_orig / total_batch # (batch_size - trial_miss)

                results_crps['spat-sadi-orig_trials'][trial] = diffsaits_crps_avg_orig / total_batch #  (batch_size - trial_miss)
                results_crps['spat-sadi-orig'] += diffsaits_crps_avg_orig / total_batch #  (batch_size - trial_miss)

            if 'SPAT-SADI' in models.keys():
                results_trials_mse['spat-sadi'][trial] = diffsaits_rmse_avg / total_batch #  (batch_size - trial_miss)
                results_mse['spat-sadi'] += diffsaits_rmse_avg / total_batch # (batch_size - trial_miss)

                results_crps['spat-sadi_trials'][trial] = diffsaits_crps_avg / total_batch #  (batch_size - trial_miss)
                results_crps['spat-sadi'] += diffsaits_crps_avg / total_batch #  (batch_size - trial_miss)

                results_mis['spat-sadi_trials'][trial] = diffsaits_mis_avg / total_batch 
                results_mis['spat-sadi'] += diffsaits_mis_avg / total_batch 
            
            if 'GP' in models.keys():
                results_trials_mse['gp'][trial] = gp_rmse_avg / total_batch # (batch_size - trial_miss)
                results_mse['gp'] += gp_rmse_avg / total_batch # (batch_size - trial_miss)
            
            if 'DK' in models.keys():
                results_trials_mse['dk'][trial] = dk_rmse_avg / total_batch # (batch_size - trial_miss)
                results_mse['dk'] += dk_rmse_avg / total_batch # (batch_size - trial_miss)

            if 'IGNNK' in models.keys():
                results_trials_mse['ignnk'][trial] = ignnk_rmse_avg / total_batch #  (batch_size - trial_miss)
                results_mse['ignnk'] += ignnk_rmse_avg / total_batch #  (batch_size - trial_miss)

            if 'PriSTI' in models.keys():
                results_trials_mse['pristi'][trial] = pristi_rmse_avg /  total_batch # (batch_size - trial_miss)
                results_mse['pristi'] += pristi_rmse_avg / total_batch #  (batch_size - trial_miss)

                results_crps['pristi_trials'][trial] = pristi_crps_avg /  total_batch # (batch_size - trial_miss)
                results_crps['pristi'] += pristi_crps_avg / total_batch # (batch_size - trial_miss)

                results_mis['pristi_trials'][trial] = pristi_mis_avg / total_batch 
                results_mis['pristi'] += pristi_mis_avg / total_batch

            results_trials_mse['mean'][trial] = mean_avg_rmse /  total_batch # (batch_size - trial_miss)
            results_mse['mean'] += mean_avg_rmse / total_batch # (batch_size - trial_miss)
                
            if 'SADI' in models.keys():
                results_trials_mse['sadi'][trial] = sadi_rmse_avg / total_batch # (batch_size - trial_miss)
                results_mse['sadi'] += sadi_rmse_avg / total_batch # (batch_size - trial_miss)
                results_crps['sadi_trials'][trial] = sadi_crps_avg / total_batch #  (batch_size - trial_miss)
                results_crps['sadi'] += sadi_crps_avg /  total_batch #(batch_size - trial_miss)

            if 'SAITS' in models.keys():
                results_trials_mse['saits'][trial] = saits_rmse_avg / total_batch # (batch_size - trial_miss)
                results_mse['saits'] += saits_rmse_avg / total_batch #  (batch_size - trial_miss)
                results_trials_mae['saits'][trial] = saits_mae_avg /  total_batch # (batch_size - trial_miss)
                results_mae['saits'] += saits_mae_avg /  total_batch # (batch_size - trial_miss)
     
            if 'KNN' in models.keys():
                results_trials_mse['knn'][trial] = knn_rmse_avg / total_batch # (batch_size - trial_miss)
                results_mse['knn'] += knn_rmse_avg / total_batch # (batch_size - trial_miss)
            
            if 'MICE' in models.keys():
                results_trials_mse['mice'][trial] = mice_rmse_avg / total_batch # (batch_size - trial_miss)
                results_mse['mice'] += mice_rmse_avg / total_batch # (batch_size - trial_miss)

            if 'BRITS' in models.keys():
                results_trials_mse['brits'][trial] = brits_rmse_avg /  total_batch # (batch_size - trial_miss)
                results_mse['brits'] += brits_rmse_avg / total_batch # (batch_size - trial_miss)
    
    if not os.path.isdir(mse_folder):
        os.makedirs(mse_folder)
    
    if not data:
        results_mse['csdi'] /= trials
        results_mse['spat-sadi'] /= trials
        results_mse['spat-sadi-orig'] /= trials
        results_mse['sadi'] /= trials
        results_mse['mean'] /= trials
        results_mse['pristi'] /= trials
        results_mse['kriging'] /= trials
        results_mse['ignnk'] /= trials
        results_mse['interpolation'] /= trials
        results_mse['gp'] /= trials
        results_mse['dk'] /= trials



        # results_mse['saits'] /= trials
        # results_mse['knn'] /= trials
        # results_mse['mice'] /= trials
        # results_mse['brits'] /= trials

        z = 1.96
        csdi_trials = -1
        csdi_crps_ci = -1
        diffsaits_trials = -1
        diffsaits_crps_ci = -1
        diffsaits_trials_orig = -1
        diffsaits_crps_ci_orig = -1

        diffsaits_mis_ci = -1
        pristi_mis_ci = -1

        ignnk_trials = -1
        pristi_trials = -1
        pristi_crps_ci = -1
        gp_trials = -1
        dk_trials = -1
        sadi_trials = -1
        sadi_crps_ci = -1
        saits_trials = -1
        knn_trials = -1
        mice_trials = -1
        brits_trials = -1
        kriging_trials = -1
        interp_trials = -1
        mean_trials = [results_trials_mse['mean'][i] for i in results_trials_mse['mean'].keys()]
        mean_trials = (z * np.std(mean_trials)) / math.sqrt(len(mean_trials))
        if 'CSDI' in models.keys():
            csdi_trials = [results_trials_mse['csdi'][i] for i in results_trials_mse['csdi'].keys()]
            csdi_trials = (z * np.std(csdi_trials)) / math.sqrt(len(csdi_trials))
            csdi_crps_ci = [results_crps['csdi_trials'][i] for i in results_crps['csdi_trials'].keys()]
            csdi_crps_ci = (z * np.std(csdi_crps_ci)) / math.sqrt(len(csdi_crps_ci))
        
        if 'OKrigging' in models.keys():
            kriging_trials = [results_trials_mse['kriging'][i] for i in results_trials_mse['kriging'].keys()]
            kriging_trials = (z * np.std(kriging_trials)) / math.sqrt(len(kriging_trials))

        if 'Interpolation' in models.keys():
            interp_trials = [results_trials_mse['interpolation'][i] for i in results_trials_mse['interpolation'].keys()]
            interp_trials = (z * np.std(interp_trials)) / math.sqrt(len(interp_trials))

        if 'DynaSTI-Orig' in models.keys():
            diffsaits_trials_orig = [results_trials_mse['spat-sadi-orig'][i] for i in results_trials_mse['spat-sadi-orig'].keys()]
            diffsaits_trials_orig = (z * np.std(diffsaits_trials_orig)) / math.sqrt(len(diffsaits_trials_orig))
            diffsaits_crps_ci_orig = [results_crps['spat-sadi-orig_trials'][i] for i in results_crps['spat-sadi-orig_trials'].keys()]
            diffsaits_crps_ci_orig = (z * np.std(diffsaits_crps_ci_orig)) / math.sqrt(len(diffsaits_crps_ci_orig))

        if 'SPAT-SADI' in models.keys():
            diffsaits_trials = [results_trials_mse['spat-sadi'][i] for i in results_trials_mse['spat-sadi'].keys()]
            diffsaits_trials = (z * np.std(diffsaits_trials)) / math.sqrt(len(diffsaits_trials))
            diffsaits_crps_ci = [results_crps['spat-sadi_trials'][i] for i in results_crps['spat-sadi_trials'].keys()]
            diffsaits_crps_ci = (z * np.std(diffsaits_crps_ci)) / math.sqrt(len(diffsaits_crps_ci))

            diffsaits_mis_ci = [results_mis['spat-sadi_trials'][i] for i in results_mis['spat-sadi_trials'].keys()]
            diffsaits_mis_ci = (z * np.std(diffsaits_mis_ci)) / math.sqrt(len(diffsaits_mis_ci))
        
        if 'GP' in models.keys():
            gp_trials = [results_trials_mse['gp'][i] for i in results_trials_mse['gp'].keys()]
            gp_trials = (z * np.std(gp_trials)) / math.sqrt(len(gp_trials))
        
        if 'DK' in models.keys():
            dk_trials = [results_trials_mse['dk'][i] for i in results_trials_mse['dk'].keys()]
            dk_trials = (z * np.std(dk_trials)) / math.sqrt(len(dk_trials))

        if 'IGNNK' in models.keys():
            ignnk_trials = [results_trials_mse['ignnk'][i] for i in results_trials_mse['ignnk'].keys()]
            ignnk_trials = (z * np.std(ignnk_trials)) / math.sqrt(len(ignnk_trials))

        if 'PriSTI' in models.keys():
            pristi_trials = [results_trials_mse['pristi'][i] for i in results_trials_mse['pristi'].keys()]
            pristi_trials = (z * np.std(pristi_trials)) / math.sqrt(len(pristi_trials))
            pristi_crps_ci = [results_crps['pristi_trials'][i] for i in results_crps['pristi_trials'].keys()]
            pristi_crps_ci = (z * np.std(pristi_crps_ci)) / math.sqrt(len(pristi_crps_ci))
            pristi_mis_ci = [results_mis['pristi_trials'][i] for i in results_mis['pristi_trials'].keys()]
            pristi_mis_ci = (z * np.std(pristi_mis_ci)) / math.sqrt(len(pristi_mis_ci))

        if 'SADI' in models.keys():
            sadi_trials = [results_trials_mse['sadi'][i] for i in results_trials_mse['sadi'].keys()]
            sadi_trials = (z * np.std(sadi_trials)) / math.sqrt(len(sadi_trials))
            sadi_crps_ci = [results_crps['sadi_trials'][i] for i in results_crps['sadi_trials'].keys()]
            sadi_crps_ci = (z * np.std(sadi_crps_ci)) / math.sqrt(len(sadi_crps_ci))

        if "SAITS" in models.keys():
            saits_trials = [results_trials_mse['saits'][i] for i in results_trials_mse['saits'].keys()]
            saits_trials = (z * np.std(saits_trials)) / math.sqrt(len(saits_trials))

        if 'KNN' in models.keys():
            knn_trials = [results_trials_mse['knn'][i] for i in results_trials_mse['knn'].keys()]
            knn_trials = (z * np.std(knn_trials)) / math.sqrt(len(knn_trials))

        if 'MICE' in models.keys():
            mice_trials = [results_trials_mse['mice'][i] for i in results_trials_mse['mice'].keys()]
            mice_trials = (z * np.std(mice_trials)) / math.sqrt(len(mice_trials))

        if 'BRITS' in models.keys():
            brits_trials = [results_trials_mse['brits'][i] for i in results_trials_mse['brits'].keys()]
            brits_trials = (z * np.std(brits_trials)) / math.sqrt(len(brits_trials))

        print(f"RMSE loss:\n\tCSDI: {results_mse['csdi']} ({csdi_trials})\n\tFDynaSTI: {results_mse['spat-sadi']} ({diffsaits_trials})\n\tDynaSTI-Orig: {results_mse['spat-sadi-orig']} ({diffsaits_trials_orig})\n\tSADI: {results_mse['sadi']} ({sadi_trials})\n\tMEAN: {results_mse['mean']} ({mean_trials})\n\tPriSTI: {results_mse['pristi']} ({pristi_trials})\n\tMICE: {results_mse['mice']} ({mice_trials})\n\tOKriging: {results_mse['kriging']} ({kriging_trials})\n\tIGNNK: {results_mse['ignnk']} ({ignnk_trials})\n\tInterp: {results_mse['interpolation']} ({interp_trials})\n\tGP: {results_mse['gp']} ({gp_trials})\n\tDeepKriging: {results_mse['dk']} ({dk_trials})")
            #    \
            #   ({diffsaits_trials})\n\tSAITS: {results_mse['saits']} ({saits_trials})\n\tKNN: {results_mse['knn']} ({knn_trials}) \
            #         \n\tMICE: {results_mse['mice']} ({mice_trials})\n\tBRITS: {results_mse['brits']} ({brits_trials})")

        if 'SPAT-SADI' in models.keys():
            diffsaits_trials = np.array([results_trials_mse['spat-sadi'][i] for i in results_trials_mse['spat-sadi'].keys()])
            mean_trials = np.array([results_trials_mse['mean'][i] for i in results_trials_mse['mean'].keys()])

            t_stat, p_value = ttest_rel(diffsaits_trials, mean_trials)

            # Output results
            print("SPAT-SADI VS Mean")
            print("t-statistic:", t_stat)
            print("p-value:", p_value)

            # Interpretation
            if p_value < 0.05:
                print("The difference between the models is statistically significant.")
            else:
                print("No statistically significant difference between the models.")

        # print(f"MAE loss:\n\tCSDI: {results_mae['csdi']}\n\tDiffSAITS: {results_mae['diffsaits']}\n\tSAITS: {results_mae['saits']}")

        results_crps['csdi'] /= trials
        results_crps['spat-sadi'] /= trials
        results_crps['spat-sadi-orig'] /= trials
        results_crps['sadi'] /= trials

        results_mis['spat-sadi'] /= trials
        results_mis['pristi'] /= trials

        print(f"CRPS:\n\tCSDI: {results_crps['csdi']} ({csdi_crps_ci})\n\tFDynaSTI: {results_crps['spat-sadi']} ({diffsaits_crps_ci})\n\tDynaSTI-Orig: {results_crps['spat-sadi-orig']} ({diffsaits_crps_ci_orig})\n\tSADI: {results_crps['sadi']} ({sadi_crps_ci})\n\tPriSTI: {results_crps['pristi']} ({pristi_crps_ci})")
        print("\n\n")
        print(f"MIS:\n\tFDynaSTI: {results_mis['spat-sadi']} ({diffsaits_mis_ci})\n\tPriSTI: {results_mis['pristi']} ({pristi_mis_ci})")

        fp = open(f"{mse_folder}/mse-trials-random-{random_trial}-forecasting-{forecasting}-blackout-{not (random_trial or forecasting)}_l_{length}_miss_{missing_ratio}_pbm_{-1 if partial_bm_config is None else partial_bm_config['features']}.json", "w")
        json.dump(results_trials_mse, fp=fp, indent=4)
        fp.close()

        fp = open(f"{mse_folder}/mse-random-{random_trial}-forecasting-{forecasting}-blackout-{not (random_trial or forecasting)}_l_{length}_miss_{missing_ratio}_pbm_{-1 if partial_bm_config is None else partial_bm_config['features']}.json", "w")
        json.dump(results_mse, fp=fp, indent=4)
        fp.close()

        
        fp = open(f"{mse_folder}/crps-random-{random_trial}-forecasting-{forecasting}-blackout-{not (random_trial or forecasting)}_l_{length}_miss_{missing_ratio}_pbm_{-1 if partial_bm_config is None else partial_bm_config['features']}.json", "w")
        json.dump(results_crps, fp=fp, indent=4)
        fp.close()
    else:
        fp = open(f"{mse_folder}/data-random-{random_trial}-forecasting-{forecasting}-blackout-{not (random_trial or forecasting)}_l_{length}_miss_{missing_ratio}_pbm_{-1 if partial_bm_config is None else partial_bm_config['features']}.json", "w")
        json.dump(results_data, fp=fp, indent=4, cls=NumpyArrayEncoder)
        fp.close()

