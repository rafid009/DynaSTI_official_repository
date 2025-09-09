import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
from torch.utils.data.distributed import DistributedSampler

given_features = [
    'AIR_TEMP_F',
    'SECOND_AIR_TEMP_F',
    'RELATIVE_HUMIDITY_%',
    'DEWPOINT_F',
    'WIND_SPEED_2M_MPH',
    'WIND_GUST_2M_MPH',
    'SOLAR_RAD_WM2'
]

def dynamic_sensor_selection(X_train, X_loc, rate=0.5, L=30, K=len(given_features)):
    inverse_rate = int((1 - rate) * X_train.reshape(L, -1, K).shape[1])
    indices = np.random.choice(X_train.reshape(L, -1, K).shape[1], inverse_rate, replace=False)
    shp = X_train.shape
    X_train = X_train.reshape(X_train.shape[0], -1, K)
    X_train[:, indices, :] = np.nan
    X_loc[indices, :] = 0
    X_train = X_train.reshape(shp)
    return X_train, X_loc

def parse_data(sample, rate=0.2, is_test=False, length=100, include_features=None, forward_trial=-1, lte_idx=None, random_trial=False, pattern=None, partial_bm_config=None, spatial=False, X_test=None, X_loc_train=None, X_loc_test=None, X_pristi=None, is_separate=False, is_dynamic=False, dynamic_rate=-1, is_subset=False):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""
    if isinstance(sample, torch.Tensor):
        sample = sample.numpy()

    obs_mask = ~np.isnan(sample)
    if spatial and X_test is not None:
        L, K = sample.shape
        # evals = sample.reshape(L, -1, 2)
        
        # print(f"evals: {evals.shape}")
        index = int(np.random.choice(X_test.reshape(L, -1, len(given_features)).shape[1], 1, replace=False))
        
        if is_subset:
            feature_idxs = np.random.choice(2, 1, replace=False)
        else:
            feature_idxs = None

        if is_dynamic:
            sample, X_loc_train = dynamic_sensor_selection(sample, X_loc_train, dynamic_rate, L, len(given_features))
            obs_mask = ~np.isnan(sample)

        evals, values, evals_loc, missing_data, missing_data_loc, evals_pristi, values_pristi = get_test_data_spatial(X_train=sample, X_test=X_test, X_loc_train=X_loc_train,
                                                          X_loc_test=X_loc_test, index=index, X_pristi=X_pristi)

        missing_data_mask = ~np.isnan(missing_data)
        if feature_idxs is None:
            mask = np.zeros_like(missing_data)
        else:
            shp = missing_data_mask.shape
            mask = np.copy(missing_data_mask).reshape((missing_data_mask.shape[0], -1, len(given_features)))
            mask[:,:,feature_idxs] = 0.0
            mask = mask.reshape(shp)
        mask_pristi = ~np.isnan(values_pristi)

        obs_data = np.nan_to_num(evals, copy=True)
        obs_data_pristi = np.nan_to_num(evals_pristi, copy=True)
        missing_data = np.nan_to_num(missing_data, copy=True)

        obs_mask_pristi = ~np.isnan(evals_pristi)
        # return obs_data, obs_mask, mask, evals_loc, obs_data_pristi, mask_pristi, obs_mask_pristi, values, missing_data, missing_data_mask, missing_data_loc
        return obs_data, obs_mask, mask, evals_loc, values, missing_data, missing_data_mask, missing_data_loc, obs_data_pristi, mask_pristi, obs_mask_pristi
      
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
        X_test_loc = X_test_loc.reshape(-1, 3)
        X_loc = X_loc.reshape(-1, 3)
        X_test_loc[len(train_indices), :] = X_loc[index, :]
        X_test_loc = X_test_loc.reshape(-1)
    return X_test, X_test_values, X_test_loc

def get_test_data_spatial(X_train, X_test, X_loc_train, X_loc_test, index):
    X_test_missing = np.expand_dims(X_test.reshape(X_test.shape[0], -1, len(given_features))[:, index,:], axis=1)
    X_loc_test_missing = np.expand_dims(X_loc_test[index,:], axis=0)
    X_pristi = X_pristi.reshape(X_pristi.shape[0], -1, 2)
    X_pristi[:, X_train.shape[1] - 1 + index, :] = X_test.reshape(X_test.shape[0], -1, 2)[:,index,:]
    
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
        if loc_x[0] == loc[0] and loc_x[1] == loc[1] and loc_x[2] == loc[2]:
            break
        index += 1
    return index 


def parse_data_spatial(sample, X_loc, X_test_loc, neighbor_location, spatial_choice=None, is_separate=False):
    
    
    L, K = sample.shape
    evals = sample.copy().reshape(L, -1, 2)

    chosen_location = np.random.choice(np.arange(X_test_loc.shape[0]))
    location_idx = get_location_index(X_loc, X_test_loc[chosen_location])

    neighbors = json.load(open(neighbor_location, 'r'))

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

class AWN_Dataset(Dataset):
    def __init__(self, total_stations, mean_std_file, n_features, rate=0.1, is_test=False, length=100, seed=10, forward_trial=-1, random_trial=False, pattern=None, partial_bm_config=None, is_valid=False, spatial=False, simple=False, is_neighbor=False, spatial_choice=None, is_separate=False, zone=8, spatial_slider=False, dynamic_rate=-1, is_subset=False) -> None:
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


        if is_neighbor:
            if is_test or is_valid:
                X = np.load(f"./data/awn/zone_{zone}_X_test.npy")
                self.neighbor_location = "./data/nacse/neighbors.json"
                X_loc_test = np.load(f"./data/nacse/X_OR_temps_test_loc.npy")
            else:
                X = np.load(f"./data/awn/X_OR_temps_total_train.npy")
                self.neighbor_location = "./data/awn/neighbors_train.json"
                train_indices = np.load(f"./data/awn/X_OR_temps_train_indices.npy")
            X_loc = np.load(f"./data/nacse/zone_{zone}_total_loc.npy")
            B, L, _ = X.shape
            

            if not (is_test or is_valid):
                X = X.reshape(B, L, -1, len(given_features))
                X = X[:,:,train_indices,:]
                X = X.reshape(B, L, -1)   
                X_loc = X_loc[train_indices]

            B, L, K = X.shape

            self.eval_length = X.shape[1]

            X_new = X.reshape(X.shape[0], X.shape[1], -1, len(given_features)) # B, L, N, K
            

            X_new = np.transpose(X_new, [3, 0, 1, 2]) # K, B, L, N
            X_new = X_new.reshape(X_new.shape[0], -1)

            if is_test or is_valid:
                self.mean = np.load(f"{mean_std_file}_mean.npy")
                self.std = np.load(f"{mean_std_file}_std.npy")
                self.mean_loc = np.load(f"{mean_std_file}_mean_loc.npy")
                self.std_loc = np.load(f"{mean_std_file}_std_loc.npy")
            else:
                train_X = X_new.copy()
                self.mean = np.nanmean(train_X, axis=1)
                self.mean = np.where(np.isnan(self.mean), 0, self.mean)
                self.mean = self.mean.reshape((1, -1))
                np.save(f"{mean_std_file}_mean.npy", self.mean)

                self.std = np.nanstd(train_X, axis=1)
                self.std = np.where(self.std == 0, 1, self.std)
                self.std = self.std.reshape((1, -1))
                np.save(f"{mean_std_file}_std.npy", self.std)

                self.mean_loc = np.mean(X_loc, axis=0)
                self.std_loc = np.std(X_loc, axis=0)

                np.save(f"zone_{zone}_{mean_std_file}_mean_loc.npy", self.mean_loc)
                np.save(f"zone_{zone}_{mean_std_file}_std_loc.npy", self.std_loc)

            if is_test or is_valid:
                for i in tqdm(range(X.shape[0])):
                    if is_separate:
                        # obs_val, obs_mask, mask, X_loc_temp, obs_val_pristi, mask_pristi, obs_mask_pristi, loc, gt_intact, missing_data, missing_data_mask = parse_data_spatial(X[i], X_loc, X_loc_test, self.neighbor_location, spatial_choice, is_separate)
                        obs_val, obs_mask, mask, X_loc_temp, loc, gt_intact, missing_data, missing_data_mask = parse_data_spatial(X[i], X_loc, X_loc_test, self.neighbor_location, spatial_choice, is_separate)
                        
                        self.missing_data.append(missing_data)
                        self.missing_data_mask.append(missing_data_mask)
                    else:
                        obs_val, obs_mask, mask, X_loc_temp, obs_val_pristi, mask_pristi, obs_mask_pristi, loc, gt_intact = parse_data_spatial(X[i], X_loc, X_loc_test, self.neighbor_location, spatial_choice)
                    self.observed_values.append(obs_val)
                   
                    if spatial_choice is not None:
                        new_locations = X_loc_temp - loc
                        if spatial_choice == 'sole-delta':
                            X_loc_temp = new_locations
                        elif spatial_choice == 'add-delta':
                            X_loc_temp = np.concatenate([X_loc_temp, new_locations], axis=-1)
                    self.spatial_info.append(X_loc_temp)
                    self.observed_masks.append(obs_mask)
                    self.gt_masks.append(mask)
                    self.observed_values_pristi.append(obs_val_pristi)
                    self.observed_masks_pristi.append(obs_mask_pristi)
                    self.gt_masks_pristi.append(mask_pristi)
                    self.gt_intact.append(gt_intact)
                self.spatial_info = torch.tensor(np.array(self.spatial_info), dtype=torch.float32)
            else:
                self.observed_values = torch.tensor(np.nan_to_num(X), dtype=torch.float32)
                self.observed_masks = torch.tensor(~np.isnan(X), dtype=torch.float32)
                self.gt_masks = self.observed_masks
                X_loc_train = np.load("./data/nacse/X_OR_temps_train_loc.npy")
                self.spatial_info = np.repeat(X_loc_train[np.newaxis, :, :], B, axis=0)
                self.total_loc = torch.tensor(X_loc, dtype=torch.float32)
                self.total_loc = (self.total_loc.reshape(-1, 3) - self.mean_loc) / self.std_loc
                        
                self.observed_masks_pristi = torch.tensor(~np.isnan(X_pristi), dtype=torch.float32)

                self.gt_masks_pristi = self.observed_masks_pristi
                self.observed_values_pristi = torch.tensor(np.nan_to_num(X_pristi), dtype=torch.float32)
                
                self.spatial_info = torch.tensor(self.spatial_info, dtype=torch.float32)
                self.spatial_info = (self.spatial_info.reshape(B, -1, 3) - self.mean_loc) / self.std_loc
                

            self.gt_masks = torch.tensor(np.array(self.gt_masks), dtype=torch.float32)   
            self.observed_values = torch.tensor(np.array(self.observed_values), dtype=torch.float32)
            if is_separate and (is_test or is_valid):
                self.missing_data = torch.tensor(np.array(self.missing_data), dtype=torch.float32)
                self.missing_data_mask = torch.tensor(np.array(self.missing_data_mask), dtype=torch.float32)
                self.missing_data = ((self.missing_data.reshape(B, L, -1, 2) - self.mean) / self.std) * self.missing_data_mask.reshape(B, L, -1, 2)

            self.observed_masks = torch.tensor(np.array(self.observed_masks), dtype=torch.float32)
            
            self.gt_masks_pristi = torch.tensor(np.array(self.gt_masks_pristi), dtype=torch.float32)
            self.observed_values_pristi = torch.tensor(np.array(self.observed_values_pristi), dtype=torch.float32)
            self.observed_masks_pristi = torch.tensor(np.array(self.observed_masks_pristi), dtype=torch.float32)

            
            self.observed_values = ((self.observed_values.reshape(B, L, -1, 2) - self.mean) / self.std) * self.observed_masks.reshape(B, L, -1, 2)
            self.observed_values_pristi = ((self.observed_values_pristi.reshape(B, L, -1, 2) - self.mean) / self.std) * self.observed_masks_pristi.reshape(B, L, -1, 2)
            
            self.gt_intact = torch.tensor(np.array(self.gt_intact), dtype=torch.float32)
        else:
            if is_test or is_valid:
                X_test = np.load(f"./data/awn/zone_{zone}_X_test.npy")
                X_loc_test = np.load(f"./data/awn/zone_{zone}_test_locs.npy")
                X = np.load(f"./data/awn/zone_{zone}_X_train_test.npy")
                if zone == 8:
                    X_test = X_test[:int(len(X_test) * 0.3)]
                    X = X[:int(len(X) * 0.3)]
            else:
                X = np.load(f"./data/awn/zone_{zone}_X_train.npy")
            X_loc = np.load(f"./data/awn/zone_{zone}_train_locs.npy")
            
            B, L, N, K = X.shape

            X_ = X.reshape(B, L, -1, n_features)
            if self.is_separate:
                X = X.reshape(B, L, -1)
                
            else:
                
                X_temp = np.zeros((X_.shape[0], X_.shape[1], X_.shape[2] + 1, X_.shape[3]))
                X_temp[:, :, :X_.shape[2], :] = X_
                X = X_temp.reshape(B, L, -1)
                X_loc_ = np.zeros((X_loc.shape[0] + 1, X_loc.shape[1]))
                X_loc_[:X_loc.shape[0]] = X_loc
                X_loc = X_loc_
            

            X_loc = X_loc.reshape(-1) # N*3

            B, L, K = X.shape
            

            self.eval_length = X.shape[1]


            X_new = X.reshape(X.shape[0], X.shape[1], -1, n_features) # B, L, N, K


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
                self.mean_loc = np.mean(X_loc.reshape(-1, 3), axis=0)
                self.std_loc = np.std(X_loc.reshape(-1, 3), axis=0)
                # print(f"spatial los mean: {self.mean_loc} and std: {self.std_loc}")

                np.save(f"{mean_std_file}_mean_loc.npy", self.mean_loc)
                np.save(f"{mean_std_file}_std_loc.npy", self.std_loc)
                
                # print(f"mean: {self.mean}\n\nstd: {self.std}")
            include_features = []
            X_loc = X_loc.reshape(-1, 3)
            # print(f"X_loc: {X_loc}")

                
            for i in tqdm(range(X.shape[0])):
                if is_test or is_valid:
                    is_dynamic = dynamic_rate != -1
                    obs_val, obs_mask, mask, X_loc_temp, values, missing_data, missing_data_mask, missing_data_loc, obs_data_pristi, mask_pristi, obs_mask_pristi = parse_data(X[i], rate, is_test, length, include_features=include_features, \
                                                                        forward_trial=forward_trial, random_trial=random_trial, \
                                                                            pattern=pattern, partial_bm_config=partial_bm_config, \
                                                                                spatial=spatial, X_test=X_test[i], \
                                                                                    X_loc_train=X_loc,\
                                                                                    X_loc_test=X_loc_test, X_pristi=None, is_dynamic=is_dynamic, dynamic_rate=dynamic_rate, is_subset=is_subset)
                
                
                else:
                    obs_val, obs_mask, mask = parse_data(X[i], rate, False, length, include_features=include_features, \
                                                                        forward_trial=forward_trial, random_trial=random_trial)
                
                if (is_test or is_valid):
                    self.gt_intact.append(values)
                    self.observed_values_pristi.append(obs_val_pristi)
                    self.observed_masks_pristi.append(obs_mask_pristi)
                    self.gt_masks_pristi.append(mask_pristi)

                if obs_mask.sum() == 0 or ((is_test or is_valid) and missing_data_mask.sum() == 0) or ((is_test or is_valid) and (~np.isnan(self.gt_intact) * 1.0).sum() == 0):
                    continue

                
                self.observed_values.append(obs_val)
                if is_test or is_valid:
                    self.spatial_info.append(X_loc_temp)
                    if self.is_separate:
                        self.missing_data.append(missing_data)
                        self.missing_data_mask.append(missing_data_mask)
                        self.missing_data_loc.append(missing_data_loc)
                        # print(f"missing data loc: {missing_data_loc}")
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
                    

            # print(f"spatial_info: {self.spatial_info.shape}")
            self.observed_values = ((self.observed_values.reshape(self.observed_values.shape[0], L, -1, len(given_features)) - self.mean) / self.std) * self.observed_masks.reshape(self.observed_masks.shape[0], L, -1, len(given_features))
            if is_test or is_valid:
                self.observed_values_pristi = ((self.observed_values_pristi.reshape(self.observed_values_pristi.shape[0], L, -1, 2) - self.mean) /self.std) * self.observed_masks_pristi.reshape(self.observed_masks_pristi.shape[0], L, -1, 2)
            
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
            s["observed_data_pristi"] = self.observed_values_pristi[index].reshape(self.observed_values_pristi[index].shape[0], -1, 2)
            s["observed_mask_pristi"] = self.observed_masks_pristi[index].reshape(self.observed_masks_pristi[index].shape[0], -1, 2)
            s['gt_mask_pristi'] = self.gt_masks_pristi[index].reshape(self.gt_masks_pristi[index].shape[0], -1, 2)
        
        if len(self.gt_masks) == 0:
            s["gt_mask"] = None
            s['gt_mask_pristi'] = None
        else:
            s["gt_mask"] = self.gt_masks[index].reshape(self.gt_masks[index].shape[0], -1, len(given_features))
        return s
    
    def __len__(self):
        return len(self.observed_values)


def get_dataloader(total_stations, mean_std_file, n_features, batch_size=16, missing_ratio=0.2, is_test=False, simple=False, is_neighbor=False, spatial_choice=None, is_separate=False, zone=7):
    train_dataset = AWN_Dataset(total_stations, mean_std_file, n_features, rate=0.0001, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, zone=zone)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = AWN_Dataset(total_stations, mean_std_file, n_features, rate=missing_ratio, pattern=None, is_valid=True, spatial=True, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, zone=zone)
    
    if is_test:
        test_loader = DataLoader(test_dataset, batch_size=1)
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader


def get_testloader_awn(total_stations, mean_std_file, n_features, n_steps=672, batch_size=16, missing_ratio=0.2, seed=10, length=100, forecasting=False, random_trial=False, pattern=None, partial_bm_config=None, spatial=False, simple=False, is_neighbor=False, spatial_choice=None, is_separate=False, zone=7, spatial_slider=False, dynamic_rate=-1, is_subset=False):
    np.random.seed(seed=seed)
    if forecasting:
        forward = n_steps - length
        test_dataset = AWN_Dataset(total_stations, mean_std_file, n_features, rate=missing_ratio, is_test=True, length=length, forward_trial=forward, pattern=pattern, partial_bm_config=partial_bm_config)
    else:
        test_dataset = AWN_Dataset(total_stations, mean_std_file, n_features, rate=missing_ratio, is_test=True, length=length, random_trial=random_trial, pattern=pattern, partial_bm_config=partial_bm_config, spatial=spatial, simple=simple, is_neighbor=is_neighbor, spatial_choice=spatial_choice, is_separate=is_separate, zone=zone, spatial_slider=spatial_slider, dynamic_rate=dynamic_rate, is_subset=is_subset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader