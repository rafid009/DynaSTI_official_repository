import numpy as np
from models.ignnk import IGNNK
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import haversine_distances

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_spatial_mask(observed_mask, locations, multi=False):
        # potential_locs = torch.arange(locations.shape[1]) * 1.0
        # chosen_location = int(torch.multinomial(potential_locs, 1)[0])
        cond_mask = observed_mask.clone()
        chosen_locations = []
        num_indices = torch.randint(2, int(observed_mask.shape[1]/2), (1,)).item()
        for i in range(observed_mask.shape[0]):  # First dimension
             # Second dimension
            # Find valid indices in the 3rd dimension for this (i, j)
            valid_indices = torch.where(torch.any(observed_mask[i, :, :, :].reshape(-1, observed_mask.shape[2] * observed_mask.shape[3]), dim=1))[0]
            if len(valid_indices) > 0:
                # Randomly select one valid index
                if multi:
                    chosen_location = valid_indices[torch.randint(len(valid_indices), (num_indices,))]
                else:
                    chosen_location = valid_indices[torch.randint(len(valid_indices), (1,)).item()]

       
                cond_mask[i, chosen_location, :, :] = 0
                chosen_locations.append(chosen_location)
        if multi:
            chosen_locations = np.array(chosen_locations)
        return locations, cond_mask, chosen_locations

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

def get_missing_mask(X_res, obs_data, obs_mask, chosen_locs, is_multi=False):
    if is_multi:
        missing_data_mask = torch.zeros((obs_mask.shape[0], obs_mask.shape[1], chosen_locs.shape[1]), device=device)
        X_res_copy = torch.zeros((obs_mask.shape[0], obs_mask.shape[1], chosen_locs.shape[1]), device=device)
        obs_data_copy = torch.zeros((obs_mask.shape[0], obs_mask.shape[1], chosen_locs.shape[1]), device=device)
    else:
        missing_data_mask = torch.zeros((obs_mask.shape[0], obs_mask.shape[1], 1), device=device)
        X_res_copy = torch.zeros((obs_mask.shape[0], obs_mask.shape[1], 1), device=device)
        obs_data_copy = torch.zeros((obs_mask.shape[0], obs_mask.shape[1], 1), device=device)
    for i in range(obs_mask.shape[0]):
        missing_data_mask[i] = obs_mask[i, :, chosen_locs[i]].unsqueeze(-1)
        X_res_copy[i] = X_res[i, :, chosen_locs[i]].unsqueeze(-1)
        obs_data_copy[i] = obs_data[i, :, chosen_locs[i]].unsqueeze(-1)
    return X_res_copy, obs_data_copy, missing_data_mask
        

def train_ignnk(STmodel, learning_rate, max_iter, train_loader, valid_loader, is_multi=False, output_path='saved_model_ignnk.model'):

    # STmodel = IGNNK(h, z, K)  # The graph neural networks

    criterion = nn.MSELoss()
    optimizer = optim.Adam(STmodel.parameters(), lr=learning_rate)
    RMSE_list = []
    MAE_list = []
    MAPE_list = []
    for epoch in range(max_iter):
        avg_loss = 0
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                observed_data = train_batch['observed_data'].to(device).float()
                observed_mask = train_batch['observed_mask'].to(device).float()
                observed_data = observed_data.permute(0, 2, 3, 1) # B, N, K, L
                observed_mask = observed_mask.permute(0, 2, 3, 1) # B, N, K, L
                spatial_info = train_batch['spatial_info'].to(device).float()
                locations, cond_mask, chosen_locations = get_spatial_mask(observed_mask, spatial_info, multi=is_multi)
                locations = locations[0, :, :2]
                dist_graph = geographical_distance(locations.cpu().numpy())
                adj = get_similarity(dist_graph)

                A_q = torch.from_numpy((calculate_random_walk_matrix(adj).T).astype('float32')).to(device=device)
                A_h = torch.from_numpy((calculate_random_walk_matrix(adj.T).T).astype('float32')).to(device=device)

                B, N, K, L = observed_data.shape
                input_data = observed_data * cond_mask
                input_data = input_data.reshape((B, N, K*L)).permute(0, 2, 1).to(device=device)
                observed_mask = observed_mask.reshape((B, N, K*L)).permute(0,2,1).to(device=device)

                optimizer.zero_grad()
                X_res = STmodel(input_data, A_q, A_h)

                observed_data = observed_data.reshape((B, N, K*L)).permute(0, 2, 1).to(device=device)
                # print(f"X_res: {X_res.shape}, observed_data: {input_data.shape}, observed_mask: {observed_mask.shape}")
                # X_res = X_res[:,:,chosen_locations]
                # observed_data = observed_data[:,:,chosen_locations]
                # missing_data_mask = observed_mask[:,:,chosen_locations]
                X_res, observed_data, missing_data_mask = get_missing_mask(X_res, observed_data, observed_mask, chosen_locations, is_multi=is_multi)
                # print(f"X-res: {X_res}\n\nobs data: {observed_data}\n\nmissing: {missing_data_mask}")
                loss = ((X_res - observed_data) ** 2) * missing_data_mask
                # print(f"loss: {loss}\n\nmissing mask sum: {missing_data_mask.sum()}")
                loss = loss.sum() / missing_data_mask.sum()

                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch,
                    },
                    refresh=False,
                )

        valid_epoch_interval=5
        if valid_loader is not None and (epoch + 1) % valid_epoch_interval == 0:
            STmodel.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        observed_data = valid_batch['observed_data'].to(device).float()
                        observed_mask = valid_batch['observed_mask'].to(device).float()
                        observed_data = observed_data.permute(0, 2, 3, 1) # B, N-1, K, L
                        observed_mask = observed_mask.permute(0, 2, 3, 1) # B, N-1, K, L
                        spatial_info = valid_batch['spatial_info'].to(device).float()

                        missing_data = valid_batch['missing_data'].to(device).float()
                        missing_data_mask = valid_batch["missing_data_mask"].to(device).float()
                        missing_data_loc = valid_batch["missing_data_loc"].to(device).float()
                        missing_data = missing_data.permute(0, 2, 3, 1) # B, 1, K, L
                        missing_data_mask = missing_data_mask.permute(0, 2, 3, 1) # B, 1, K, L

                        observed_data = torch.cat([observed_data, missing_data], dim=1) # B, N, K, L
                        observed_mask = torch.cat([observed_mask, missing_data_mask], dim=1) # B, N, K, L
                        locations = torch.cat([spatial_info, missing_data_loc], dim=1) # B, N+1, 3
                        locations = locations[0, :, :2]
                        dist_graph = geographical_distance(locations.cpu().numpy())
                        adj = get_similarity(dist_graph)

                        A_q = torch.from_numpy((calculate_random_walk_matrix(adj).T).astype('float32')).to(device=device)
                        A_h = torch.from_numpy((calculate_random_walk_matrix(adj.T).T).astype('float32')).to(device=device)

                        if is_multi:
                            _, M, _, _ = missing_data.shape
                        B, N, K, L = observed_data.shape
                        input_data = observed_data.clone()
                        if is_multi:
                            input_data[:, :-M, :, :] = 0.0
                        else:
                            input_data[:, -1, :, :] = 0.0
                        input_data = input_data.reshape((B, N, K*L)).permute(0, 2, 1).to(device=device)
                        observed_mask = observed_mask.reshape((B, N, K*L)).permute(0, 2, 1).to(device=device)
                        
                        # loss = STmodel(valid_batch, is_train=0)
                        X_res = STmodel(input_data, A_q, A_h)
                        observed_data = observed_data.reshape((B, N, K*L)).permute(0, 2, 1).to(device=device)
                        if is_multi:
                            X_res = X_res[:,:,:-M]
                            observed_data = observed_data[:,:,:-M]
                            missing_data_mask = missing_data_mask.permute(0, 2, 3, 1).reshape((B, K*L, M))
                        else:
                            X_res = X_res[:,:,-1]
                            observed_data = observed_data[:,:,-1]
                            missing_data_mask = missing_data_mask.reshape((B, K*L))
                        loss = ((X_res - observed_data) ** 2) * missing_data_mask
                        loss = loss.sum() / missing_data_mask.sum()


                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch,
                            },
                            refresh=False,
                        )
            torch.save(STmodel.state_dict(), output_path)
            STmodel.train()

        
        # for i in range(training_set.shape[0]//(h * batch_size)):  #using time_length as reference to record test_error
        #     t_random = np.random.randint(0, high=(training_set_s.shape[0] - h), size=batch_size, dtype='l')
        #     know_mask = set(random.sample(range(0,training_set_s.shape[1]),n_o_n_m)) #sample n_o + n_m nodes
        #     feed_batch = []
        #     for j in range(batch_size):
        #         feed_batch.append(training_set_s[t_random[j]: t_random[j] + h, :][:, list(know_mask)]) #generate 8 time batches
            
        #     inputs = np.array(feed_batch)
        #     inputs_omask = np.ones(np.shape(inputs))
        #     if not dataset == 'NREL': 
        #         inputs_omask[inputs == 0] = 0           # We found that there are irregular 0 values for METR-LA, so we treat those 0 values as missing data,
        #                                                 # For other datasets, it is not necessary to mask 0 values
                                                    
        #     missing_index = np.ones((inputs.shape))
        #     for j in range(batch_size):
        #         missing_mask = random.sample(range(0,n_o_n_m),n_m) #Masked locations
        #         missing_index[j, :, missing_mask] = 0
        #     # if dataset == 'NREL':
        #     #     Mf_inputs = inputs * inputs_omask * missing_index / capacities[:, None]
        #     # else:
        #         # Mf_inputs = inputs * inputs_omask * missing_index / E_maxvalue #normalize the value according to experience
        #     Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32'))
        #     mask = torch.from_numpy(inputs_omask.astype('float32'))   #The reconstruction errors on irregular 0s are not used for training
            
        #     A_dynamic = A_s[list(know_mask), :][:, list(know_mask)]   #Obtain the dynamic adjacent matrix
        #     A_q = torch.from_numpy((calculate_random_walk_matrix(A_dynamic).T).astype('float32'))
        #     A_h = torch.from_numpy((calculate_random_walk_matrix(A_dynamic.T).T).astype('float32'))
            
        #     if dataset == 'NREL':
        #         outputs = torch.from_numpy(inputs/capacities[:, None])
        #     else:
        #         outputs = torch.from_numpy(inputs/E_maxvalue) #The label
            
        #     optimizer.zero_grad()
        #     X_res = STmodel(Mf_inputs, A_q, A_h)  #Obtain the reconstruction
            
        #     loss = criterion(X_res*mask, outputs*mask)
        #     loss.backward()
        #     optimizer.step()