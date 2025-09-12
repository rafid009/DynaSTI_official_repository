import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json

given_features = [1]

def dynamic_sensor_selection(X_train, X_loc, rate=0.5, L=30, K=1):
    inverse_rate = int((1 - rate) * X_train.reshape(L, -1, K).shape[1])
    indices = np.random.choice(X_train.reshape(L, -1, K).shape[1], inverse_rate, replace=False)
    shp = X_train.shape
    X_train = X_train.reshape(X_train.shape[0], -1, K)
    X_train[:, indices, :] = np.nan
    X_loc[indices, :] = 0
    X_train = X_train.reshape(shp)
    return X_train, X_loc

def parse_data(sample, rate=0.2, is_test=False, length=100, include_features=None, forward_trial=-1, lte_idx=None, random_trial=False, pattern=None, partial_bm_config=None, spatial=False, X_test=None, X_loc_train=None, X_loc_test=None, X_pristi=None, is_separate=False, is_dynamic=False, dynamic_rate=-1):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""
    if isinstance(sample, torch.Tensor):
        sample = sample.numpy()

    obs_mask = ~np.isnan(sample)
    if spatial and X_test is not None:
        L, K = sample.shape
        index = int(np.random.choice(X_test.reshape(L, -1, 1).shape[1], 1, replace=False))
        
        if is_dynamic:
            sample, X_loc_train = dynamic_sensor_selection(sample, X_loc_train, dynamic_rate, L, 1)
            obs_mask = ~np.isnan(sample)

        evals, values, evals_loc, missing_data, missing_data_loc, evals_pristi, values_pristi = get_test_data_spatial(X_train=sample, X_test=X_test, X_loc_train=X_loc_train,
                                                          X_loc_test=X_loc_test, index=index, X_pristi=X_pristi)

        missing_data_mask = ~np.isnan(missing_data)
        mask = np.zeros_like(missing_data)
        mask_pristi = ~np.isnan(values_pristi)

        obs_data_pristi = np.nan_to_num(evals_pristi, copy=True)
        obs_data = np.nan_to_num(evals, copy=True)
        
        obs_mask_pristi = ~np.isnan(evals_pristi)
        missing_data = np.nan_to_num(missing_data, copy=True)
        return obs_data, obs_mask, mask, evals_loc, values, missing_data, missing_data_mask, missing_data_loc, obs_data_pristi, mask_pristi, obs_mask_pristi
      
    elif pattern is not None:
        shp = sample.shape
        choice = np.random.randint(low=pattern['start'], high=(pattern['start'] + pattern['num_patterns'] - 1))
        # print(f"start: {pattern['start']} end: {(pattern['start'] + pattern['num_patterns'] - 1)} choice: {choice}")
        filename = f"{pattern['pattern_dir']}/pattern_{choice}.npy"
        mask = np.load(filename)
        mask = mask * obs_mask
        evals = sample.reshape(-1).copy()
        
        
        eval_mask = mask.reshape(-1).copy()
        gt_indices = np.where(eval_mask)[0].tolist()
        miss_indices = np.random.choice(
            gt_indices, (int)(len(gt_indices) * rate), replace=False
        )
        gt_intact = sample.reshape(-1).copy()
        gt_intact[miss_indices] = np.nan
        gt_intact = gt_intact.reshape(shp)
        obs_data = np.nan_to_num(evals, copy=True)
        obs_data = obs_data.reshape(shp)
    elif not is_test:
        shp = sample.shape
        evals = sample.reshape(-1).copy()
        indices = np.where(~np.isnan(evals))[0].tolist()
        indices = np.random.choice(indices, int(len(indices) * rate))
        values = evals.copy()
        # values[indices] = np.nan
        mask = ~np.isnan(values)
        mask = mask.reshape(shp)
        gt_intact = values.reshape(shp).copy()
        obs_data = np.nan_to_num(evals, copy=True)
        obs_data = obs_data.reshape(shp)
        # obs_data_intact = evals.reshape(shp)
    elif random_trial:
        evals = sample.copy()
        values = evals.copy()
        for i in range(evals.shape[1]):
            indices = np.where(~np.isnan(evals[:, i]))[0].tolist()
            indices = np.random.choice(indices, int(len(indices) * rate))
            if len(indices) != 0:
                values[indices, i] = np.nan
        mask = ~np.isnan(values)
        gt_intact = values
        obs_data = np.nan_to_num(evals, copy=True)
    elif forward_trial != -1:
        indices = np.where(~np.isnan(sample[:, lte_idx]))[0].tolist()
        start = indices[forward_trial]
        obs_data = np.nan_to_num(sample, copy=True)
        gt_intact = sample.copy()
        gt_intact[start:, :] = np.nan
        mask = ~np.isnan(gt_intact)
    elif partial_bm_config is not None:
        total_features = np.arange(sample.shape[1])
        features = np.random.choice(total_features, partial_bm_config['features'])
        obs_data, mask, gt_intact = partial_bm(sample, features, partial_bm_config['length_range'], partial_bm_config['n_chunks'])
    else:
        shp = sample.shape
        evals = sample.reshape(-1).copy()
        a = np.arange(sample.shape[0] - length)
        # print(f"a: {a}\nsample: {sample.shape}")
        start_idx = np.random.choice(a)
        # print(f"random choice: {start_idx}")
        end_idx = start_idx + length
        obs_data_intact = sample.copy()
        if include_features is None or len(include_features) == 0:
            obs_data_intact[start_idx:end_idx, :] = np.nan
        else:
            obs_data_intact[start_idx:end_idx, include_features] = np.nan
        mask = ~np.isnan(obs_data_intact)
        gt_intact = obs_data_intact
        obs_data = np.nan_to_num(evals, copy=True)
        obs_data = obs_data.reshape(shp)
    return obs_data, obs_mask, mask#, sample, gt_intact

def get_train_data(train_indices, X, X_loc):
    X_real = np.zeros(X.shape) # B, L, N, K
    X_real[:, :, :len(train_indices), :] = X[:, :, train_indices, :]

    X_loc_real = np.zeros(X_loc.shape) # B, N, 3
    X_loc_real[:len(train_indices), :] = X_loc[train_indices, :]
    return X_real, X_loc_real


def get_test_data(X_train, X, X_loc_train, X_loc, index, train_indices):
    if index in train_indices:
        X_test = X_train.copy()
        X_test = X_test.reshape(X_test.shape[0], -1, len(given_features))
        X_test_values = X_test.copy()
        X_test_values[:, index, :] = np.nan
        X_test_loc = X_loc_train.copy()
    else:
        X_test = X_train.copy() # L, N*K
        X_test = X_test.reshape(X_test.shape[0], -1, len(given_features)) # L, N, K
        X = X.reshape(X.shape[0], -1, len(given_features))
        X_test[:, len(train_indices), :] = X[:, index, :]
        X_test_values = X_test.copy()
        X_test_values[:, len(train_indices), :] = np.nan
        X_test = X_test.reshape(X_test.shape[0], -1)
        X_test_values = X_test_values.reshape(X_test_values.shape[0], -1)

        X_test_loc = X_loc_train.copy()
        X_test_loc = X_test_loc.reshape(-1, 2)
        X_loc = X_loc.reshape(-1, 2)
        X_test_loc[len(train_indices), :] = X_loc[index, :]
        X_test_loc = X_test_loc.reshape(-1)
    return X_test, X_test_values, X_test_loc

def get_test_data_spatial(X_train, X_test, X_loc_train, X_loc_test, index, X_pristi):
    X_train = X_train.reshape(X_train.shape[0], -1, 1)
    X_test_missing = np.expand_dims(X_test.reshape(X_test.shape[0], -1, len(given_features))[:, index,:], axis=1)
    
    X_loc_test_missing = np.expand_dims(X_loc_test[index,:], axis=0)
    X_pristi = X_pristi.reshape(X_pristi.shape[0], -1, 1)
    # print(f"X_pristi: {X_pristi.shape}, X_test: {X_test.shape}, X_train: {X_train.shape}, index: {index}")
    X_pristi[:, X_train.shape[1] - 1 + index, :] = X_test.reshape(X_test.shape[0], -1, 1)[:,index,:]
    
    values = X_train.copy()
    values_pristi = X_pristi.copy()
    values_pristi[:, X_train.shape[1] - 1 + index, :] = np.nan

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test_missing = X_test_missing.reshape(X_test_missing.shape[0], -1)
    values = values.reshape(X_train.shape[0], -1)
    return X_train, values, X_loc_train, X_test_missing, X_loc_test_missing, X_pristi, values_pristi

def get_location_index(X_loc, loc):
    index = 0
    for loc_x in X_loc:
        if loc_x[0] == loc[0] and loc_x[1] == loc[1]:
            break
        index += 1
    return index 


def parse_data_spatial(sample, X_loc, X_test_loc, neighbor_location, spatial_choice=None, is_separate=False):
    
    
    L, K = sample.shape
    evals = sample.copy().reshape(L, -1, 2)

    chosen_location = np.random.choice(np.arange(X_test_loc.shape[0]))
    location_idx = get_location_index(X_loc, X_test_loc[chosen_location])

    neighbors = json.load(open(neighbor_location, 'r'))

    # print(f"neighbors: {neighbors}")

    locations = neighbors[f"{location_idx}"]
    
    if spatial_choice is not None:
        new_locations = X_loc - np.expand_dims(X_loc[location_idx, :], axis=0)
        new_locations = new_locations[locations]
        
        if spatial_choice == 'sole-delta':
            X_loc = new_locations
        elif spatial_choice == 'add-delta':
            X_loc = X_loc[locations]
            X_loc = np.concatenate([X_loc, new_locations], axis=-1)


    evals_pristi = np.zeros(evals.shape)
    evals_pristi[:, locations, :] = evals[:, locations, :]
    obs_mask_pristi = ~np.isnan(evals_pristi)

    values = evals.copy()
    if is_separate:
        missing_data = values[:, location_idx, :]
        missing_data_mask = ~np.isnan(missing_data)
        missing_data = np.nan_to_num(missing_data, copy=True)
        values[:, location_idx, :] = np.nan
    else: 
        values[:, location_idx, :] = np.nan
        values = values[:, locations, :]
        mask = ~np.isnan(values)
    
    
    evals = evals[:, locations, :]
    obs_mask = ~np.isnan(evals)
    if is_separate:
        mask = obs_mask

    values_pristi = evals_pristi.copy()
    values_pristi[:, location_idx, :] = np.nan
    
    
    mask_pristi = ~np.isnan(values_pristi)

    evals = evals.reshape(L, -1)
    evals = np.nan_to_num(evals)
    obs_mask = obs_mask.reshape(L, -1)
    mask = mask.reshape(L, -1)

    evals_pristi = evals_pristi.reshape(L, -1)
    evals_pristi = np.nan_to_num(evals_pristi)
    obs_mask_pristi = obs_mask_pristi.reshape(L, -1)
    mask_pristi = mask_pristi.reshape(L, -1)

    evals_loc = X_loc #[locations]

    if is_separate:
        return evals, obs_mask, mask, evals_loc, evals_pristi, mask_pristi, obs_mask_pristi, X_test_loc[chosen_location], values, missing_data, missing_data_mask
    else:
        return evals, obs_mask, mask, evals_loc, evals_pristi, mask_pristi, obs_mask_pristi, X_test_loc[chosen_location], values

class METRLA_Dataset(Dataset):
    def __init__(self, total_stations, mean_std_file, n_features, rate=0.1, is_test=False, length=100, seed=10, forward_trial=-1, random_trial=False, pattern=None, partial_bm_config=None, is_valid=False, spatial=False, is_neighbor=False, spatial_choice=None, is_separate=False, dynamic_rate=-1, is_pristi=False) -> None:
        super().__init__()
        
        self.observed_values = []
        self.observed_values_pristi = []
        self.spatial_info = []
        # self.obs_data_intact = []
        self.observed_masks = []
        self.observed_masks_pristi = []
        self.gt_masks = []
        self.gt_masks_pristi = []
        self.total_loc = []
        self.gt_intact = []
        if is_separate:
            self.missing_data = []
            self.missing_data_mask = []
            self.missing_data_loc = []
        self.is_separate = is_separate
        self.is_test = is_test or is_valid
        folder = './data/metr-la'

        
        if is_test or is_valid:
            X_test = np.load(f"{folder}/X_test_test.npy")
            # print(f"init X_test: {X_test.shape}")
            X_loc_test = np.load(f"{folder}/X_test_locs.npy")
            # print(f"X loc test: {X_loc_test.shape}")
            X = np.load(f"{folder}/X_test_train.npy")
            # print(f"X: {X.shape}")
        else:
            if is_pristi:
                X = np.load(f"{folder}/X_total_train.npy")
            else:
                X = np.load(f"{folder}/X_train.npy")
        if is_pristi:
            X_loc = np.load(f"{folder}/X_total_locs.npy")
        else:
            X_loc = np.load(f"{folder}/X_train_locs.npy")
        # print(f"X: {X.shape}\nX_loc: {X_loc.shape}")
        B, L, N = X.shape
        K = 1

        X_ = X.reshape(B, L, -1, n_features)
        X = X.reshape(B, L, -1)

        X_temp = np.zeros((X_.shape[0], X_.shape[1], total_stations, X_.shape[3]))
        X_temp[:, :, :X_.shape[2], :] = X_
        X_pristi = X_temp.reshape(B, L, -1)

        self.eval_length = X.shape[1]

        # X_orig, X_loc_orig = X.copy(), X_loc.copy()

        X_new = X.reshape(B, L, -1, K) # B, L, N, K
    

        X_new = np.transpose(X_new, [3, 0, 1, 2])
        X_new = X_new.reshape(X_new.shape[0], -1)

        if is_test or is_valid:
            self.mean = np.load(f"{mean_std_file}_mean.npy")
            self.std = np.load(f"{mean_std_file}_std.npy")
            self.mean_loc = np.load(f"{mean_std_file}_mean_loc.npy")
            self.std_loc = np.load(f"{mean_std_file}_std_loc.npy")
        else:
            train_X = X_new.copy()
            # train_X = train_X.reshape((-1, X.shape[2]))
            self.mean = np.nanmean(train_X, axis=1)
            self.mean = np.where(np.isnan(self.mean), 0, self.mean)
            self.mean = self.mean.reshape((1, -1))
            np.save(f"{mean_std_file}_mean.npy", self.mean)

            self.std = np.nanstd(train_X, axis=1)
            self.std = np.where(self.std == 0, 1, self.std)
            self.std = self.std.reshape((1, -1))
            np.save(f"{mean_std_file}_std.npy", self.std)
            # print(f"X loc: {X_loc.shape}")
            
            # print(f"X_loc reshape: {X_loc.reshape(B, L, -1, 3)[0,0,:,:]}")
            self.mean_loc = np.mean(X_loc, axis=0)
            self.std_loc = np.std(X_loc, axis=0)
            # print(f"spatial los mean: {self.mean_loc} and std: {self.std_loc}")

            np.save(f"{mean_std_file}_mean_loc.npy", self.mean_loc)
            np.save(f"{mean_std_file}_std_loc.npy", self.std_loc)
            
            # print(f"mean: {self.mean}\n\nstd: {self.std}")
        include_features = []


            
        for i in tqdm(range(X.shape[0])):
            if is_test or is_valid:
                is_dynamic = dynamic_rate != -1
                obs_val, obs_mask, mask, X_loc_temp, values, missing_data, missing_data_mask, missing_data_loc, obs_val_pristi, mask_pristi, obs_mask_pristi = parse_data(X[i], rate, is_test, length, include_features=include_features, \
                                                                    forward_trial=forward_trial, random_trial=random_trial, \
                                                                        pattern=pattern, partial_bm_config=partial_bm_config, \
                                                                            spatial=spatial, X_test=X_test[i], \
                                                                                X_loc_train=X_loc,\
                                                                                X_loc_test=X_loc_test, X_pristi=X_pristi[i], is_dynamic=is_dynamic, dynamic_rate=dynamic_rate)
            
            else:
                obs_val, obs_mask, mask = parse_data(X[i], rate, False, length, include_features=include_features, \
                                                                    forward_trial=forward_trial, random_trial=random_trial)
            
            

            if obs_mask.sum() == 0 or ((is_test or is_valid) and missing_data_mask.sum() == 0):
                continue

            if (is_test or is_valid):
                self.gt_intact.append(values)
                self.observed_values_pristi.append(obs_val_pristi)
                self.observed_masks_pristi.append(obs_mask_pristi)
                self.gt_masks_pristi.append(mask_pristi)

            self.observed_values.append(obs_val)
            if is_test or is_valid:
                self.spatial_info.append(X_loc_temp)
                if self.is_separate:
                    self.missing_data.append(missing_data)
                    self.missing_data_mask.append(missing_data_mask)
                    self.missing_data_loc.append(missing_data_loc)
            else:
                self.spatial_info.append(X_loc)
            
            
            self.observed_masks.append(obs_mask)
            self.gt_masks.append(mask)

            if (is_test or is_valid):
                self.gt_intact.append(values)


        self.gt_masks = torch.tensor(np.array(self.gt_masks), dtype=torch.float32)   
        self.observed_values = torch.tensor(np.array(self.observed_values), dtype=torch.float32)
        self.spatial_info = torch.tensor(np.array(self.spatial_info, dtype=np.float64), dtype=torch.float64)
        self.observed_masks = torch.tensor(np.array(self.observed_masks), dtype=torch.float32)

        if is_test or is_valid:
            self.observed_values_pristi = torch.tensor(np.array(self.observed_values_pristi), dtype=torch.float32)
            self.observed_masks_pristi = torch.tensor(np.array(self.observed_masks_pristi), dtype=torch.float32)
            self.gt_masks_pristi = torch.tensor(np.array(self.gt_masks_pristi), dtype=torch.float32)
            self.gt_intact = torch.tensor(np.array(self.gt_intact), dtype=torch.float32)
            if self.is_separate:
                self.missing_data = torch.tensor(np.array(self.missing_data), dtype=torch.float32)
                self.missing_data_mask = torch.tensor(np.array(self.missing_data_mask), dtype=torch.float32)
                self.missing_data_loc = torch.tensor(np.array(self.missing_data_loc, dtype=np.float64), dtype=torch.float64)
                self.missing_data = ((self.missing_data.reshape(self.missing_data.shape[0], L, -1, len(given_features)) - self.mean) / self.std) * self.missing_data_mask.reshape(self.missing_data_mask.shape[0], L, -1, len(given_features))
 
        self.observed_values = ((self.observed_values.reshape(self.observed_values.shape[0], L, -1, len(given_features)) - self.mean) / self.std) * self.observed_masks.reshape(self.observed_masks.shape[0], L, -1, len(given_features))
        if is_test or is_valid:
            self.observed_values_pristi = ((self.observed_values_pristi.reshape(self.observed_values_pristi.shape[0], L, -1, len(given_features)) - self.mean) /self.std) * self.observed_masks_pristi.reshape(self.observed_masks_pristi.shape[0], L, -1, len(given_features))
            
        self.neighbor_location = None
        
        

        
    def __getitem__(self, index):
        s = {
            "observed_data": self.observed_values[index].reshape(self.observed_values[index].shape[0], -1, len(given_features)),
            "spatial_info": self.spatial_info[index],
            "observed_mask": self.observed_masks[index].reshape(self.observed_masks[index].shape[0], -1, len(given_features)),
            "mean_loc":  np.expand_dims(self.mean_loc, axis=0),
            "std_loc":  np.expand_dims(self.std_loc, axis=0),
            "timepoints": np.arange(self.eval_length),
            "gt_intact": self.gt_intact[index].reshape(self.gt_intact[index].shape[0], -1, len(given_features)) if len(self.gt_intact) != 0 else self.gt_intact,
        }
        if self.is_separate and self.is_test:
            s["missing_data"] = self.missing_data[index].reshape(self.missing_data[index].shape[0], -1, len(given_features))
            s['missing_data_mask'] = self.missing_data_mask[index].reshape(self.missing_data[index].shape[0], -1, len(given_features))
            s['missing_data_loc'] = self.missing_data_loc[index]

        if self.is_test:
            s["observed_data_pristi"] = self.observed_values_pristi[index].reshape(self.observed_values_pristi[index].shape[0], -1, len(given_features))
            s["observed_mask_pristi"] = self.observed_masks_pristi[index].reshape(self.observed_masks_pristi[index].shape[0], -1, len(given_features))
            s['gt_mask_pristi'] = self.gt_masks_pristi[index].reshape(self.gt_masks_pristi[index].shape[0], -1, len(given_features))
        
       
        if len(self.gt_masks) == 0:
            s["gt_mask"] = None
            s['gt_mask_pristi'] = None
        else:
            s["gt_mask"] = self.gt_masks[index].reshape(self.gt_masks[index].shape[0], -1, len(given_features))
        return s
    
    def __len__(self):
        return len(self.observed_values)


def get_dataloader(total_stations, mean_std_file, n_features, batch_size=16, missing_ratio=0.2, is_test=False, simple=False, is_neighbor=False, spatial_choice=None, is_separate=False, is_pristi=False):
    # np.random.seed(seed=seed)
    train_dataset = METRLA_Dataset(total_stations, mean_std_file, n_features, rate=0.0001, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, is_pristi=is_pristi)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = METRLA_Dataset(total_stations, mean_std_file, n_features, rate=missing_ratio, pattern=None, is_valid=True, spatial=True, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, is_pristi=is_pristi)
    
    if is_test:
        test_loader = DataLoader(test_dataset, batch_size=1)
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader


def get_testloader_metrla(total_stations, mean_std_file, n_features, n_steps=288, batch_size=16, missing_ratio=0.2, seed=10, length=100, forecasting=False, random_trial=False, pattern=None, partial_bm_config=None, spatial=False, simple=False, is_neighbor=False, spatial_choice=None, is_separate=False, spatial_slider=False, dynamic_rate=-1):
    np.random.seed(seed=seed)
    if forecasting:
        forward = n_steps - length
        test_dataset = METRLA_Dataset(total_stations, mean_std_file, n_features, rate=missing_ratio, is_test=True, length=length, forward_trial=forward, pattern=pattern, partial_bm_config=partial_bm_config)
    else:
        test_dataset = METRLA_Dataset(total_stations, mean_std_file, n_features, rate=missing_ratio, is_test=True, length=length, random_trial=random_trial, pattern=pattern, partial_bm_config=partial_bm_config, spatial=spatial, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, dynamic_rate=dynamic_rate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader