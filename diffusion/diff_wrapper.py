import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from models.diff_model import SADI, DiT, diff_CSDI, DynaSTI, Guide_diff
import json
import scipy.sparse as sp
import math
from sklearn.metrics.pairwise import haversine_distances

def cosine_beta_schedule(timesteps: int, dtype=torch.float32):
    """
    Cosine beta schedule based on α_t = cos(0.5πt).

    Parameters:
    - timesteps (int): Total number of timesteps.
    - dtype (torch.dtype): The desired data type of the output.

    Returns:
    - torch.Tensor: A tensor of beta values for each timestep.
    """
    # Generate timesteps from 0 to 1
    t = torch.linspace(0, 1, timesteps + 1, dtype=dtype)

    # Calculate alpha_t using the cosine formula
    alpha_t = torch.cos(0.5 * math.pi * t)

    # Compute beta_t as 1 - (alpha_t / alpha_t+1)
    # α_t+1 starts at index 1 of alpha_t
    beta_t = 1 - (alpha_t[1:] / alpha_t[:-1])

    return beta_t

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class Diffusion_base(nn.Module):
    def __init__(self, config, device, n_spatial) -> None:
        super().__init__()
        self.device = device
        self.ablation_config = config['ablation']
        self.target_strategy = config['model']['target_strategy']
        
        self.diffmodel = None
        self.num_steps = config['diffusion']['num_steps']
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + 1

        # self.emb_total_dim += 1 

        if config['diffusion']['schedule'] == 'quad':
            self.betas = np.linspace(
                config['diffusion']["beta_start"] ** 0.5, config['diffusion']["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config['diffusion']["schedule"] == "linear":
            self.betas = np.linspace(
                config['diffusion']["beta_start"], config['diffusion']["beta_end"], self.num_steps
            )
        elif config['diffusion']["schedule"] == "cosine":
            self.betas = betas_for_alpha_bar(
                self.num_steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2, config['diffusion']['beta_end']
            )

        self.alphas = torch.tensor(1 - self.betas)
        self.n_spatial = n_spatial
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)
        self.alpha_torch = torch.tensor(self.alpha_hats).float().unsqueeze(1).unsqueeze(1).unsqueeze(1)
        self.is_dit = config['is_dit'] if 'is_dit' in config else False
        self.is_neighbor = config['is_neighbor'] if 'is_neighbor' in config else False
        self.is_separate = config['is_separate'] if 'is_separate' in config else False
        self.is_ignnk = config['is_ignnk'] if 'is_ignnk' in config else False
        self.is_dit_ca2 = config['is_dit_ca2'] if 'is_dit_ca2' in config else False
        self.ddim = config['ddim'] if 'ddim' in config else False
        self.ddim_steps = config['ddim_steps'] if 'ddim_steps' in config else -1
        self.is_multi = config['is_multi'] if 'is_multi' in config.keys() else False
        self.is_pristi = config['is_pristi'] if 'is_pristi' in config.keys() else False
        
        if self.is_pristi:
            self.embed_layer = nn.Embedding(
                num_embeddings=config['model']['d_spatial']*config['model']['n_feature'], embedding_dim=self.emb_feature_dim
            )
        
        self.target_dim = config['model']['d_spatial'] * config['model']['n_feature']
        if self.is_dit_ca2:
            self.diffmodel = DynaSTI(
                    config=config,
                    d_time=config['model']['d_time'],
                    d_feature=config['model']['n_feature'],
                    n_spatial=n_spatial,
                    d_k=config['model']['d_k'],
                    d_v=config['model']['d_v'],
                    n_layer=config['model']['n_layers'],
                    n_spatial_layer=config['model']['n_spatial_layers'],
                    num_heads=config['model']['n_head']
                )
        elif self.is_pristi:
            # self.train_stations = config['train_stations']
            config["side_dim"] = self.emb_total_dim
            self.diffmodel = Guide_diff(
                config=config,
                inputdim=2,
                target_dim=config['model']['n_feature'] * config['model']['d_spatial'],
                is_itp=False
            )
        else:
            self.diffmodel = SADI(
                        diff_steps=config['diffusion']['num_steps'],
                        n_layers=config['model']['n_layers'],
                        d_time=config['model']['d_time'],
                        d_feature=config['model']['n_feature'] * self.n_spatial,
                        d_model=config['model']['d_model'],
                        d_inner=config['model']['d_inner'],
                        n_head=config['model']['n_head'],
                        d_k=config['model']['d_k'],
                        d_v=config['model']['d_v'],
                        dropout=config['model']['dropout'],
                        diff_emb_dim=config['diffusion']['diffusion_embedding_dim'],
                        diagonal_attention_mask=config['model']['diagonal_attention_mask'],
                        ablation_config=config['ablation']
                    )
            

    def get_location_index(self, X_loc, loc):
        index = 0
        for loc_x in X_loc:
            if loc_x[0] == loc[0] and loc_x[1] == loc[1] and loc_x[2] == loc[2]:
                break
            index += 1
        return index if index != X_loc.shape[0] else None
    
    def get_spatial_nodes(self, observed_data, observed_mask, locations, neighbor_location):        
        potential_locs = torch.arange(locations.shape[1]) * 1.0
        # print(f"locs: {locations.shape}")
        chosen_location = int(torch.multinomial(potential_locs, 1)[0])
        # print(f"chosen: {chosen_location}")
        selected_location = locations[:,chosen_location,:].unsqueeze(1)
        neighbors = json.load(open(neighbor_location, 'r'))
        location_indices = neighbors[f"{chosen_location}"]
        cond_mask = observed_mask.clone()
        # exact_location = self.get_location_index(total_loc, chosen_coords)
        
        if self.is_separate:
            missing_data = observed_data[:, chosen_location, :, :].clone().unsqueeze(1)
            missing_data_mask = cond_mask[:, chosen_location, :, :].clone().unsqueeze(1)
            cond_mask[:, chosen_location, :, :] = 0.0
            missing_data_loc = selected_location
            # print(f"missing data in spatial node selection: {missing_data.shape}")
        else:
            cond_mask[:, chosen_location, :, :] = 0.0
            missing_data = None
            missing_data_loc = None
            missing_data_mask = None
        
        
        observed_data = observed_data[:, location_indices, :, :]
        observed_mask = observed_mask[:, location_indices, :, :]
        locations = locations[:, location_indices, :]
        if self.is_separate:
            cond_mask = cond_mask[:, chosen_location, :, :].unsqueeze(1)
        else:
            cond_mask = cond_mask[:, location_indices, :, :]

        return observed_data, observed_mask, cond_mask, locations, missing_data, missing_data_loc, missing_data_mask

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask
    
    def get_spatial_mask_separate(self, observed_data, observed_mask, locations):
        if self.is_multi:
            num_indices = torch.randint(2, int(observed_data.shape[1]/2), (1,)).item()
            missing_data = torch.zeros((observed_data.shape[0], num_indices, observed_data.shape[2], observed_data.shape[3])).to(self.device) #.cuda()
            missing_data_mask = torch.zeros((observed_mask.shape[0], num_indices, observed_mask.shape[2], observed_mask.shape[3])).to(self.device) #.cuda()
            missing_location = torch.zeros((locations.shape[0], num_indices, locations.shape[2])).to(self.device)
        else:
            missing_data = torch.zeros((observed_data.shape[0], 1, observed_data.shape[2], observed_data.shape[3])).to(self.device) #.cuda()
            missing_data_mask = torch.zeros((observed_mask.shape[0], 1, observed_mask.shape[2], observed_mask.shape[3])).to(self.device) #.cuda()
            missing_location = torch.zeros((locations.shape[0], 1, locations.shape[2])).to(self.device) #.cuda()
        cond_mask = observed_mask.clone()
        
        for i in range(observed_data.shape[0]):  # First dimension
             # Second dimension
            # Find valid indices in the 3rd dimension for this (i, j)
            valid_indices = torch.where(torch.any(observed_mask[i, :, :, :].reshape(-1, observed_mask.shape[2] * observed_mask.shape[3]), dim=1))[0]
            if len(valid_indices) > 0:
                # Randomly select one valid index
                if self.is_multi:
                    # num_indices = torch.randint(2, int(len(valid_indices)/2), (1,)).item()
                    chosen_location = valid_indices[torch.randint(len(valid_indices), (num_indices,))]
                else:
                    chosen_location = valid_indices[torch.randint(len(valid_indices), (1,)).item()]
                
                # Update the new mask for the chosen index in the 3rd dimension
                # new_mask[i, j, chosen_index, :] = 0  # Ensure it stays marked as valid


                # potential_locs = torch.arange(locations.shape[1]) * 1.0
                # chosen_location = int(torch.multinomial(potential_locs, 1)[0])
                if self.is_multi:
                    missing_location[i] = locations[i, chosen_location, :].clone() #.unsqueeze(0)
    

                    missing_data[i] = observed_data[i, chosen_location, :, :].clone() #.unsqueeze(0)
                    missing_data_mask[i] = observed_mask[i, chosen_location, :, :].clone()
                
                else:
                    missing_location[i] = locations[i, chosen_location, :].clone().unsqueeze(0)
    

                    missing_data[i] = observed_data[i, chosen_location, :, :].clone().unsqueeze(0)
                    missing_data_mask[i] = observed_mask[i, chosen_location, :, :].clone().unsqueeze(0)
                
                
                observed_data[i, chosen_location, :, :] = 0
                observed_mask[i, chosen_location, :, :] = 0

                
                cond_mask[i, chosen_location, :, :] = 0

                locations[i, chosen_location, :] = 0
                    
                if not self.is_pristi and not self.is_multi:
                    cond_mask[i] = cond_mask[i, chosen_location, :, :].unsqueeze(0)

        return observed_data, observed_mask, locations, cond_mask, missing_data, missing_location, missing_data_mask


    
    def get_spatial_mask(self, observed_mask):
        cond_mask = observed_mask.clone() # B, N, K, L
        # rand_mask = self.get_randmask(observed_mask)
        # if is_train == 1:
        #     cond_mask = rand_mask
        # else:
        for i in range(len(cond_mask)):
 
            index = np.random.choice(observed_mask.shape[1], size=1, replace=False)
            cond_mask[i,index,:,:] = 0
        
        return cond_mask
 
    def get_hist_mask(self, observed_mask):

        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()

        for i in range(len(cond_mask)):

            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:
                cond_mask[i] = cond_mask[i] * observed_mask[i - 1]
        return cond_mask
    
    def get_bm_mask(self, observed_mask):
        cond_mask = observed_mask.clone()
        for i in range(cond_mask.shape[0]):
            start = np.random.randint(0, cond_mask.shape[2] - int(cond_mask.shape[2] * 0.1))
            length = np.random.randint(int(cond_mask.shape[2] * 0.1), int(cond_mask.shape[2] * 0.2))
            start_feat = np.random.randint(0, observed_mask.shape[1])
            if start_feat != observed_mask.shape[1] - 1:
                end_feat = np.random.randint(start_feat+1, observed_mask.shape[1])
                cond_mask[i, start_feat:end_feat, start : (start + length - 1)] = 0.0
            else:
                cond_mask[i, start_feat, start : (start + length - 1)] = 0.0
        return cond_mask
        
    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def get_side_info(self, observed_tp, cond_mask):
        B, N, K, L = cond_mask.shape
        cond_mask = cond_mask.reshape(B, -1, L)
        B, K, L = cond_mask.shape
        # print(f"side: {observed_tp.shape}")
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        # print(f"time: {time_embed.shape} and feat: {feature_embed.shape}")
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
        side_info = torch.cat([side_info, side_mask], dim=1)
        return side_info

    def geographical_distance(self, x=None, to_rad=True):
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


    def thresholded_gaussian_kernel(self, x, theta=None, threshold=None, threshold_on_input=False):
        if theta is None:
            theta = np.std(x)
        weights = np.exp(-np.square(x / theta))
        if threshold is not None:
            mask = x > threshold if threshold_on_input else weights < threshold
            weights[mask] = 0.
        return weights

    def get_similarity(self, dist, thr=0.1, include_self=False, force_symmetric=False, sparse=False):
        theta = np.std(dist)  # use same theta for both air and air36
        adj = self.thresholded_gaussian_kernel(dist, theta=theta, threshold=thr)
        if not include_self:
            adj[np.diag_indices_from(adj)] = 0.
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)
        return adj
    
    def calculate_random_walk_matrix(self, adj_mx):
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

    def calc_loss_valid(
        self, observed_data, spatial_info, cond_mask, observed_mask, is_train, side_info=None, is_spat=False, missing_data=None, missing_data_mask=None,
        A_q=None, A_h=None, missing_location=None
    ):
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data, spatial_info, cond_mask, observed_mask, is_train, set_t=t, side_info=side_info, missing_data=missing_data, A_q=A_q, A_h=A_h, missing_data_mask=missing_data_mask, missing_location=missing_location
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, spatial_info, cond_mask, observed_mask, is_train, set_t=-1, side_info=None, is_spat=False, missing_data=None, missing_data_mask=None,
        A_q=None, A_h=None, missing_location=None
    ):
        
        B, N, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().cuda() #to(self.device) #.cuda() #.to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).cuda() #.to(self.device) #.cuda() #.to(self.device)
        alpha_torch = self.alpha_torch.clone().cuda() #to(self.device) #.cuda()
        current_alpha = alpha_torch[t]  # (B,1,1)
        
        
        # print(f"observed data: {observed_data}\nspatial: {spatial_info}")
        if self.is_separate:
            noise = torch.randn_like(missing_data) # B, 1, K, L
            noisy_data = (current_alpha ** 0.5) * missing_data
        else:
            noise = torch.randn_like(observed_data)
            noisy_data = (current_alpha ** 0.5) * observed_data 
        noisy_data = noisy_data + ((1.0 - current_alpha) ** 0.5) * noise

        # print(f"cond mask: {cond_mask.shape}\nnoisy data: {noisy_data.shape}\nobserved_data: {observed_data.shape}")
        if self.is_separate:
            cond_obs, noisy_data = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
            total_mask = observed_mask
            # num_eval = torch.ones_like(noisy_data).sum()
            # print(f"missing data: {missing_data}\n\nmissing data mask: {missing_data_mask}\n")
        else:
            total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        # print(f"observed data: {total_input[:,0,:,:]}\nnoisy_data: {total_input[:,1,:,:]}")
            

            if self.is_ignnk:
                total_mask = cond_mask
            else:
                temp_mask = cond_mask.unsqueeze(dim=1)
                
                total_mask = torch.cat([temp_mask, (1 - temp_mask)], dim=1)
        if self.is_separate:
            # print(f"missing mask: {missing_data_mask.shape}, cond_mask: {cond_mask.shape}")
            target_mask = missing_data_mask # - cond_mask
        else:
            target_mask = observed_mask - cond_mask
        
        num_eval = target_mask.sum()
        
        predicted_1 = None
        predicted_2 = None
        predicted_3 = None
        if self.is_ignnk:
            predicted_3 = self.diffmodel(total_input, total_mask, A_q, A_h, t)
        elif self.is_dit_ca2:
            inputs = {
                    'X_input': cond_obs,
                    'missing_mask': total_mask,
                    'spatial_context': spatial_info,
                    'X_target': noisy_data,
                    'missing_loc': missing_location,
                    'missing_data_mask': torch.zeros_like(missing_data_mask),
                    'A_q': A_q,
                    'A_h': A_h
                }
            predicted_3, attn_spat = self.diffmodel(inputs, t, is_train)
        elif self.is_pristi:
            # inputs = {
            #     'X': total_input,
            #     'missing_mask': total_mask,
            #     'spatial_context': spatial_info
            # }
            total_input = total_input.reshape(B, 2, -1, L)
            predicted_3 = self.diffmodel(total_input, side_info, t, None)
        elif self.is_dit:
            # print(f"cond_obs: {cond_obs.shape}, noisy data: {noisy_data.shape}")
            if self.is_separate:
                inputs = {
                    'X_input': cond_obs,
                    'missing_mask': total_mask,
                    'spatial_context': spatial_info,
                    'X_target': noisy_data,
                    'missing_loc': missing_location
                }
            else:
                inputs = {
                    'X': total_input,
                    'missing_mask': total_mask,
                    'spatial_context': spatial_info,
                }
            predicted_3 = self.diffmodel(inputs, t, is_spat=is_spat)
            # print(f"prediction: {predicted_3}")
        else:
            B, _, N, K, L = total_input.shape
            total_input = total_input.reshape(B, 2, -1, L)
            total_mask = total_mask.reshape(B, 2, -1, L)
            inputs = {
                'X': total_input,
                'missing_mask': total_mask,
            }
            predicted_1, predicted_2, predicted_3 = self.diffmodel(inputs, t)
        # B, N, K, L = noise.shape
        noise = noise.reshape(B, -1, L)
        target_mask = target_mask.reshape(B, -1, L)
        residual_3 = (noise - predicted_3) * target_mask
        
        if not self.is_ignnk and not self.is_dit and not self.is_pristi and is_train != 0 and (predicted_1 is not None) and (predicted_2 is not None):
            pred_loss_1 = (noise - predicted_1) * target_mask
            pred_loss_2 = (noise - predicted_2) * target_mask
            pred_loss = ((pred_loss_1 ** 2).sum() + (pred_loss_2 ** 2).sum()) / 2 
            loss = (residual_3 ** 2).sum()
            loss = (loss + pred_loss) / (2 * (num_eval if num_eval > 0 else 1))
        else:
            loss = (residual_3 ** 2).sum() / (num_eval if num_eval > 0 else 1)
            if torch.any(torch.isnan(loss)):
                print(f"cond obs: {cond_obs}\n\npredicted: {predicted_3}\n\nnoisy data: {noisy_data}")
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_separate:
            cond_obs = observed_data
            return cond_obs, noisy_data
        else:
            
            if self.is_ignnk:
                cond_obs = (cond_mask * observed_data) #.unsqueeze(1)
                noisy_target = ((1 - cond_mask) * noisy_data) #.unsqueeze(1)
                total_input = cond_obs + noisy_target
            else:
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
                total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
            return total_input
    
    def impute(self, observed_data, spatial_info, cond_mask, n_samples, side_info=None, A_q=None, A_h=None, missing_location=None, missing_data_mask=None, missing_data=None, eta=0.000, missing_dims=10):
        B, N, K, L = observed_data.shape
        if self.is_separate and (self.is_dit or self.is_dit_ca2):
            if self.is_multi:
                imputed_samples = torch.zeros(B, n_samples, missing_dims * K, L).to(self.device) #.cuda() #.to(self.device)
            else:
                imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device) #.cuda() #.to(self.device)
        else:
            imputed_samples = torch.zeros(B, n_samples, N*K, L).to(self.device) #.cuda() #.to(self.device)

        if self.ddim:
            steps = torch.linspace(self.num_steps - 1, 0, self.ddim_steps + 1, dtype=torch.float32)[1:].long()
            steps = steps.to(self.device)
            num_steps = len(steps)
        else:
            num_steps = self.num_steps
        all_samples_attn_spat = []
        for i in range(n_samples):
            if self.is_separate and (self.is_dit or self.is_dit_ca2):
                if self.is_multi:
                    if missing_data is not None:
                        current_sample = missing_data * missing_data_mask + (1 - missing_data_mask) * torch.randn((B, missing_dims, K, L)).to(self.device)
                    else:
                        current_sample = torch.randn((B, missing_dims, K, L)).to(self.device) #.cuda() #.to(self.device)
                else:
                    if missing_data is not None:
                        current_sample = missing_data * missing_data_mask + (1 - missing_data_mask) * torch.randn((B, 1, K, L)).to(self.device)
                    else:
                        current_sample = torch.randn((B, 1, K, L)).to(self.device) #.cuda() #.to(self.device)
            else:
                current_sample = torch.randn((B, N, K, L)).to(self.device) #.cuda() #.to(self.device)
            num_steps = self.num_steps
            avg_attn_spat = 0

            if self.ddim:
                for j in range(len(steps) - 1):
                    t = steps[j]
                    t_next = steps[j + 1]
                    alpha_bar_t = self.alpha_hats[t]
                    alpha_bar_t_next = self.alpha_hats[t_next]
                    if self.is_separate and (self.is_dit or self.is_dit_ca2):
                        cond_obs = observed_data
                        noisy_target = current_sample
                        total_mask = cond_mask
                    else:
                        
                        if self.is_ignnk:
                            cond_obs = (cond_mask * observed_data) #.unsqueeze(1)
                            noisy_target = ((1 - cond_mask) * current_sample) #.unsqueeze(1)
                            diff_input = cond_obs + noisy_target
                            total_mask = cond_mask
                        else:
                            
                            cond_obs = (cond_mask * observed_data).unsqueeze(1)
                            noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                            diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

                            temp_mask = cond_mask.unsqueeze(dim=1)
                            total_mask = torch.cat([temp_mask, (1 - temp_mask)], dim=1)
                    if self.is_ignnk:
                        predicted = self.diffmodel(diff_input, total_mask, A_q, A_h, torch.tensor([t]).to(self.device)) #.cuda()) #.to(self.device))
                    elif self.is_dit_ca2:
                        inputs = {
                                'X_input': cond_obs,
                                'spatial_context': spatial_info,
                                'missing_mask': total_mask,
                                'X_target': noisy_target,
                                'missing_loc': missing_location,
                                'missing_data_mask': missing_data_mask,
                                'A_q': A_q,
                                'A_h': A_h
                            }
                        predicted, attn_spat = self.diffmodel(inputs, torch.tensor([t] * B).to(self.device)) #.cuda()) #.to(self.device))
                        # print(f"attn spat: {attn_spat.shape}")
                        if attn_spat is not None:
                            avg_attn_spat += attn_spat
                    elif self.is_pristi:
                        total_input = diff_input.reshape(B, 2, -1, L)
                        predicted = self.diffmodel(total_input, side_info, t, None)
                        avg_attn_spat = 0
                    elif self.is_dit:
                        if self.is_separate:
                            inputs = {
                                'X_input': cond_obs,
                                'spatial_context': spatial_info,
                                'missing_mask': total_mask,
                                'X_target': noisy_target,
                                'missing_loc': missing_location
                                # 'adj': adj
                            }
                        else:
                            inputs = {
                                'X': diff_input,
                                'spatial_context': spatial_info,
                                'missing_mask': total_mask,
                                # 'adj': adj
                            }
                        predicted = self.diffmodel(inputs, torch.tensor([t]).to(self.device)) #.cuda()) #.to(self.device)) 
                        # print(f"current_sample 1: {current_sample}")
                        # print(f"predicted: {predicted}\n")
                    else:
                        B, _, N, K, L = diff_input.shape
                        diff_input = diff_input.reshape(B, 2, -1, L)
                        total_mask = total_mask.reshape(B, 2, -1, L)
                        inputs = {
                            'X': diff_input,
                            'missing_mask': total_mask
                        }
                        _, _, predicted = self.diffmodel(inputs, torch.tensor([t]).to(self.device)) #.cuda()) #.to(self.device))              
                    if self.is_separate and (self.is_dit or self.is_dit_ca2):
                        if self.is_multi:
                            predicted = predicted.reshape(B, missing_dims, K, L)
                        else:
                            predicted = predicted.reshape(B, 1, K, L)
                    else:
                        predicted = predicted.reshape(B, N, K, L)

                    # Predict x0
                    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
                    x0_theta = (current_sample - torch.sqrt(1 - alpha_bar_t) * predicted) / sqrt_alpha_bar_t

                    # Compute sigma for DDIM (eta controls stochasticity)
                    sigma_t = eta * torch.sqrt((1 - alpha_bar_t_next) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_next)
                    sigma_t = sigma_t.clamp(0)

                    # DDIM update
                    current_sample = (torch.sqrt(alpha_bar_t_next) * x0_theta +
                        torch.sqrt(1 - alpha_bar_t_next - sigma_t**2) * predicted +
                        sigma_t * torch.randn_like(current_sample))
                    
                    # coeff1 = 1 / self.alphas[t] ** 0.5
                    # coeff2 = (1 - self.alphas[t]) / (1 - self.alpha_hats[t]) ** 0.5
                    # current_sample = coeff1 * (current_sample - coeff2 * predicted)
                    # print(f"current sample 2: {current_sample}")
                    # if t > 0:
                    #     noise = torch.randn_like(current_sample)
                    #     sigma = (
                    #         (1.0 - self.alpha_hats[t - 1]) / (1.0 - self.alpha_hats[t]) * self.betas[t]
                    #     ) ** 0.5
                    #     # print(f"noise: {noise}")
                    #     # print(f"sigma: {sigma}")
                    #     current_sample += sigma * noise
            else:
                for t in range(num_steps - 1, -1, -1):
                    if self.is_separate and (self.is_dit or self.is_dit_ca2):
                        cond_obs = observed_data
                        noisy_target = current_sample
                        total_mask = cond_mask
                    else:
                        
                        if self.is_ignnk:
                            cond_obs = (cond_mask * observed_data) #.unsqueeze(1)
                            noisy_target = ((1 - cond_mask) * current_sample) #.unsqueeze(1)
                            diff_input = cond_obs + noisy_target
                            total_mask = cond_mask
                        else:
                            cond_obs = (cond_mask * observed_data).unsqueeze(1)
                            noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                            diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

                            temp_mask = cond_mask.unsqueeze(dim=1)
                            total_mask = torch.cat([temp_mask, (1 - temp_mask)], dim=1)
                    if self.is_ignnk:
                        predicted = self.diffmodel(diff_input, total_mask, A_q, A_h, torch.tensor([t]).to(self.device)) #.cuda()) #.to(self.device))
                    elif self.is_dit_ca2:
                        inputs = {
                                'X_input': cond_obs,
                                'spatial_context': spatial_info,
                                'missing_mask': total_mask,
                                'X_target': noisy_target,
                                'missing_loc': missing_location,
                                'missing_data_mask': missing_data_mask,
                                'A_q': A_q,
                                'A_h': A_h
                            }
                        predicted, attn_spat = self.diffmodel(inputs, torch.tensor([t]).to(self.device)) #.cuda()) #.to(self.device))
                        # print(f"attn spat: {attn_spat.shape}")
                        if attn_spat is not None:
                            avg_attn_spat += attn_spat
                    elif self.is_pristi:
                        total_input = diff_input.reshape(B, 2, -1, L)
                        predicted = self.diffmodel(total_input, side_info, t, None)
                        avg_attn_spat = 0
                    elif self.is_dit:
                        if self.is_separate:
                            inputs = {
                                'X_input': cond_obs,
                                'spatial_context': spatial_info,
                                'missing_mask': total_mask,
                                'X_target': noisy_target,
                                'missing_loc': missing_location
                                # 'adj': adj
                            }
                        else:
                            inputs = {
                                'X': diff_input,
                                'spatial_context': spatial_info,
                                'missing_mask': total_mask,
                                # 'adj': adj
                            }
                        predicted = self.diffmodel(inputs, torch.tensor([t]).to(self.device)) #.cuda()) #.to(self.device)) 
                        # print(f"current_sample 1: {current_sample}")
                        # print(f"predicted: {predicted}\n")
                    else:
                        B, _, N, K, L = diff_input.shape
                        diff_input = diff_input.reshape(B, 2, -1, L)
                        total_mask = total_mask.reshape(B, 2, -1, L)
                        inputs = {
                            'X': diff_input,
                            'missing_mask': total_mask
                        }
                        _, _, predicted = self.diffmodel(inputs, torch.tensor([t]).to(self.device)) #.cuda()) #.to(self.device))              
                    if self.is_separate and (self.is_dit or self.is_dit_ca2 or self.is_dit_ca3):
                        if self.is_multi:
                            predicted = predicted.reshape(B, missing_dims, K, L)
                        else:
                            predicted = predicted.reshape(B, 1, K, L)
                    else:
                        predicted = predicted.reshape(B, N, K, L)

                    
                    coeff1 = 1 / self.alphas[t] ** 0.5
                    coeff2 = (1 - self.alphas[t]) / (1 - self.alpha_hats[t]) ** 0.5
                    # print(f"curr: {current_sample.shape} and predicted: {predicted.shape}")
                    # print(f"coeff1: {coeff1} and coeff2: {coeff2}")
                    current_sample = coeff1 * (current_sample - coeff2 * predicted)
                    # print(f"current sample 2: {current_sample}")
                    if t > 0:
                        noise = torch.randn_like(current_sample)
                        sigma = (
                            (1.0 - self.alpha_hats[t - 1]) / (1.0 - self.alpha_hats[t]) * self.betas[t]
                        ) ** 0.5
                        # print(f"noise: {noise}")
                        # print(f"sigma: {sigma}")
                        current_sample += sigma * noise
                # print(f"current sample: {current_sample.shape}")
            # current_sample = (1 - cond_mask) * current_sample + cond_mask * observed_data
            
            if not isinstance(avg_attn_spat, int):
                avg_attn_spat /= num_steps
                if i == 0:
                    all_samples_attn_spat = avg_attn_spat.unsqueeze(0)
                else:
                    all_samples_attn_spat = torch.cat([all_samples_attn_spat, avg_attn_spat.unsqueeze(0)], dim=0)
            if self.is_separate and (self.is_dit or self.is_dit_ca2):
                if self.is_multi:
                    imputed_samples[:, i] = current_sample.detach().reshape(B, missing_dims * K, L)
                else:
                    imputed_samples[:, i] = current_sample.detach().reshape(B, K, L)
            else:
                imputed_samples[:, i] = current_sample.detach().reshape(B, N*K, L)
        if len(all_samples_attn_spat) != 0:
            attn_spat_mean = all_samples_attn_spat.mean(0)
            attn_spat_std = all_samples_attn_spat.std(0)
        else:
            attn_spat_mean, attn_spat_std = None, None
        return imputed_samples, attn_spat_mean, attn_spat_std


    def forward(self, batch, is_train=1, is_spat=False):
        if self.is_ignnk:
            (
                observed_data,
                spatial_info,
                observed_mask,
                observed_tp,
                gt_mask,
                neighbor_location,
                _,
                _,
                mean_loc,
                std_loc
            ) = self.process_data(batch)
        elif self.is_separate:
            if self.is_dit_ca2:
                (
                observed_data,
                spatial_info,
                observed_mask,
                observed_tp,
                gt_mask,
                neighbor_location,
                _,
                _,
                missing_data,
                missing_data_mask,
                missing_location,
                mean_loc,
                std_loc
            ) = self.process_data(batch)
            else:
                (
                    observed_data,
                    spatial_info,
                    observed_mask,
                    observed_tp,
                    gt_mask,
                    neighbor_location,
                    _,
                    _,
                    missing_data,
                    missing_data_mask,
                    missing_location,
                    mean_loc,
                    std_loc
                ) = self.process_data(batch)
        else:
            (
                observed_data,
                spatial_info,
                observed_mask,
                observed_tp,
                gt_mask,
                neighbor_location,
                _,
                _
            ) = self.process_data(batch)
        
        
        if is_train == 0:
            if not self.is_separate:
                missing_data = None
                missing_data_mask = None
                missing_location = None
                cond_mask = gt_mask
            else:
                # print(f"missing data in forward: {missing_location.shape}")
                cond_mask = torch.zeros_like(missing_data_mask)
        elif self.target_strategy == "mix":
            # observed_mask = self.get_spatial_nodes(observed_mask, train_indices)
            cond_mask = self.get_hist_mask(
                observed_mask
            )
            missing_data = None
        elif self.target_strategy == 'blackout':
            cond_mask = self.get_bm_mask(
                observed_mask
            )
            missing_data = None
        elif self.target_strategy == 'spatial':
            # observed_mask = self.get_spatial_nodes(observed_mask, train_indices)
            if self.is_neighbor:
                observed_data, observed_mask, cond_mask, spatial_info, missing_data, missing_location, missing_data_mask = self.get_spatial_nodes(observed_data, observed_mask, spatial_info, neighbor_location)
            else:
                observed_data, observed_mask, spatial_info, cond_mask, missing_data, missing_location, missing_data_mask = self.get_spatial_mask_separate(observed_data, observed_mask, spatial_info)
        else:
            cond_mask = self.get_randmask(observed_mask)
            missing_data = None

        A_q = None
        A_h = None

        if self.is_pristi:
            side_info = self.get_side_info(observed_tp, cond_mask)
        else:
            side_info = None

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        if self.is_separate:
            missing_location = (missing_location - mean_loc) / std_loc
        spatial_info = (spatial_info - mean_loc) / std_loc
        return loss_func(observed_data, spatial_info, cond_mask, observed_mask, is_train, side_info=side_info, is_spat=is_spat, missing_data=missing_data, A_q=A_q, A_h=A_h, missing_data_mask=missing_data_mask, missing_location=missing_location)#, adj=adj)

    def evaluate(self, batch, n_samples, missing_dims=10):
        if self.is_ignnk:
            (
                observed_data,
                spatial_info,
                observed_mask,
                observed_tp,
                gt_mask,
                _,
                cut_length,
                gt_intact,
                mean_loc,
                std_loc
            ) = self.process_data(batch)
        elif self.is_separate:
            if self.is_dit_ca2:
                (
                    observed_data,
                    spatial_info,
                    observed_mask,
                    observed_tp,
                    gt_mask,
                    _,
                    cut_length,
                    gt_intact,
                    missing_data,
                    missing_data_mask,
                    missing_location,
                    mean_loc,
                    std_loc
                ) = self.process_data(batch)
            else:
                (
                    observed_data,
                    spatial_info,
                    observed_mask,
                    observed_tp,
                    gt_mask,
                    _,
                    cut_length,
                    gt_intact,
                    missing_data,
                    missing_data_mask,
                    missing_location,
                    mean_loc,
                    std_loc
                ) = self.process_data(batch)
        else:
            (
                observed_data,
                spatial_info,
                observed_mask,
                observed_tp,
                gt_mask,
                _,
                cut_length,
                gt_intact
            ) = self.process_data(batch)

        with torch.no_grad():
            
            if self.is_separate:
                cond_mask = observed_mask
                target_mask = torch.logical_xor(missing_data_mask, gt_mask).float()
            else:
                cond_mask = gt_mask
                target_mask = observed_mask - cond_mask
            
         
            A_q = None
            A_h = None

            if self.is_pristi:
                side_info = self.get_side_info(observed_tp, cond_mask)
            else:
                side_info = None

            if self.is_separate:
                # print(f"missing loc: {missing_location.shape}, mean_loc: {mean_loc.shape}")
                missing_location = (missing_location - mean_loc) / std_loc
                # max_loc: mean_loc
                # min_loc: std_loc
                # missing_location = -1 + (2 * (missing_location - std_loc) / (mean_loc - std_loc))
            spatial_info = (spatial_info - mean_loc) / std_loc
            samples, attn_spat_mean, attn_spat_std = self.impute(observed_data, spatial_info, cond_mask, n_samples, side_info=side_info, A_q=A_q, A_h=A_h, missing_location=missing_location, missing_data_mask=gt_mask, missing_data=missing_data, missing_dims=missing_dims)

            for i in range(len(cut_length)):
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        if not self.is_pristi:
        
            B, N, K, L = observed_data.shape
            observed_data = observed_data.reshape(B, N*K, L)

            if self.is_multi:
                target_mask = target_mask.reshape(B, missing_dims * K, L)
            else:
                target_mask = target_mask.reshape(B, K, L)

            observed_mask = observed_mask.reshape(B, N*K, L)
        if self.is_separate:
            # missing_data_mask = 1.0 - missing_data_mask
            return samples, observed_data, target_mask, observed_mask, observed_tp, gt_intact, missing_data, missing_data_mask, attn_spat_mean, attn_spat_std
        else:
            return samples, observed_data, target_mask, observed_mask, observed_tp, gt_intact, None, None, None, None




class DynaSTI_PM25(Diffusion_base):
    def __init__(self, config, device, target_dim=36):
        super(DynaSTI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["cond_mask"].to(self.device).float()
        # gt_intact = batch["gt_intact"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 3, 1)
        observed_mask = observed_mask.permute(0, 2, 3, 1)
        gt_mask = gt_mask.permute(0, 2, 3, 1)
        # gt_intact = gt_intact.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 3, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            None,
            None
        )
    
class DynaSTI_NASCE(Diffusion_base):
    def __init__(self, config, device, n_spatial):
        super(DynaSTI_NASCE, self).__init__(config, device, n_spatial)

    def process_data(self, batch):
        if self.is_pristi:
            observed_data = batch["observed_data_pristi"].to(self.device).float() #.cuda().float()
            observed_mask = batch["observed_mask_pristi"].to(self.device).float() #.cuda().float()
            gt_mask = batch["gt_mask_pristi"].to(self.device).float() #.cuda().float()
        else:
            observed_data = batch["observed_data"].to(self.device).float() #.cuda().float()
            observed_mask = batch["observed_mask"].to(self.device).float() #.cuda().float()
            gt_mask = batch["gt_mask"].to(self.device).float() #.cuda().float()

        spatial_info = batch["spatial_info"].to(self.device).float() #.cuda().float()
        if self.is_separate:
            missing_data = batch["missing_data"].to(self.device).float() if "missing_data" in batch.keys() else None
            # print(f"missing_data in data: {missing_data}")
            missing_data_mask = batch["missing_data_mask"].to(self.device).float() if "missing_data_mask" in batch.keys() else None
            missing_data_loc = batch['missing_data_loc'].to(self.device).float() if "missing_data_loc" in batch .keys() else None
            # print(f"missing loc: {missing_data_loc}")
            if missing_data is not None and missing_data_mask is not None and missing_data_loc is not None:
                missing_data = missing_data.permute(0, 2, 3, 1)
                missing_data_mask = missing_data_mask.permute(0, 2, 3, 1)
                # missing_data_loc = missing_data_loc.permute(0, 2, 3, 1)
        
        observed_tp = batch["timepoints"].to(self.device).float()
        if batch["gt_intact"] is not None and not isinstance(batch["gt_intact"], list) and len(batch["gt_intact"]) != 0:
            gt_intact = batch["gt_intact"].to(self.device).float() #.cuda().float()
        else:
            gt_intact = None

        # if self.is_ignnk or self.is_dit_ca2:
        mean_loc = batch["mean_loc"].to(self.device).float() #.cuda().float()
        std_loc = batch['std_loc'].to(self.device).float() #.cuda().float()
        # max_loc = batch['max_loc'].cuda().float()
        # min_loc = batch['min_loc'].cuda().float()
        observed_data = observed_data.permute(0, 2, 3, 1) # B, N, K, L
        # spatial_info = spatial_info.permute(0, 2, 3, 1)
        observed_mask = observed_mask.permute(0, 2, 3, 1) # B, N, K, L
        gt_mask = gt_mask.permute(0, 2, 3, 1) # B, N, K, L


        cut_length = torch.zeros(len(observed_data)).long().to(self.device) #.cuda() #to(self.device)
        if self.is_ignnk:
            return (
                observed_data,
                spatial_info,
                observed_mask,
                observed_tp,
                gt_mask,
                None,
                cut_length,
                gt_intact,
                mean_loc,
                std_loc
            )
        # elif self.is_pristi:
        #     return (
        #             observed_data,
        #             spatial_info,
        #             observed_mask,
        #             observed_tp,
        #             gt_mask,
        #             None,
        #             cut_length,
        #             gt_intact,
        #             missing_data,
        #             missing_data_mask,
        #             missing_data_loc,
        #             mean_loc,
        #             std_loc
        #             # max_loc,
        #             # min_loc
        #         )
        elif self.is_separate:
            if self.is_dit_ca2:
                return (
                    observed_data,
                    spatial_info,
                    observed_mask,
                    observed_tp,
                    gt_mask,
                    None,
                    cut_length,
                    gt_intact,
                    missing_data,
                    missing_data_mask,
                    missing_data_loc,
                    mean_loc,
                    std_loc
                    # max_loc,
                    # min_loc
                )
            else:
                return (
                    observed_data,
                    spatial_info,
                    observed_mask,
                    observed_tp,
                    gt_mask,
                    None,
                    cut_length,
                    gt_intact,
                    missing_data,
                    missing_data_mask,
                    missing_data_loc,
                    mean_loc,
                    std_loc
                )
        else:
            return (
                observed_data,
                spatial_info,
                observed_mask,
                observed_tp,
                gt_mask,
                None,
                cut_length,
                gt_intact
            )
        
class DynaSTI_AWN(Diffusion_base):
    def __init__(self, config, device, n_spatial):
        super(DynaSTI_AWN, self).__init__(config, device, n_spatial)

    def process_data(self, batch):
        if self.is_pristi:
            observed_data = batch["observed_data_pristi"].to(self.device).float() #.cuda().float()
            observed_mask = batch["observed_mask_pristi"].to(self.device).float() #.cuda().float()
            gt_mask = batch["gt_mask_pristi"].to(self.device).float() #.cuda().float()
        else:
            observed_data = batch["observed_data"].cuda().float() #.to(self.device).float()
            observed_mask = batch["observed_mask"].cuda().float() #.to(self.device).float()
            gt_mask = batch["gt_mask"].cuda().float() #.to(self.device).float()

        spatial_info = batch["spatial_info"].cuda().float() #.to(self.device).float()
        if self.is_separate:
            missing_data = batch["missing_data"].cuda().float() if "missing_data" in batch.keys() else None
            # print(f"missing_data in data: {missing_data}")
            missing_data_mask = batch["missing_data_mask"].cuda().float() if "missing_data_mask" in batch.keys() else None
            missing_data_loc = batch['missing_data_loc'].cuda().float() if "missing_data_loc" in batch .keys() else None
            # print(f"missing loc: {missing_data_loc}")
            if missing_data is not None and missing_data_mask is not None and missing_data_loc is not None:
                missing_data = missing_data.permute(0, 2, 3, 1)
                missing_data_mask = missing_data_mask.permute(0, 2, 3, 1)
                # missing_data_loc = missing_data_loc.permute(0, 2, 3, 1)
        
        observed_tp = batch["timepoints"].cuda().float()
        if "neighbor_location" in batch.keys() and batch["neighbor_location"] is not None:
            neighbor_location = batch["neighbor_location"][0]
        else:
            neighbor_location = None
        if batch["gt_intact"] is not None and not isinstance(batch["gt_intact"], list) and len(batch["gt_intact"]) != 0:
            gt_intact = batch["gt_intact"].cuda().float() #.cuda().float() #.to(self.device).float()
        else:
            gt_intact = None

        # if self.is_ignnk or self.is_dit_ca2:
        mean_loc = batch["mean_loc"].cuda().float() #.cuda().float() #.to(self.device).float()
        std_loc = batch['std_loc'].cuda().float() #.cuda().float() #.to(self.device).float()

        observed_data = observed_data.permute(0, 2, 3, 1) # B, N, K, L
        # spatial_info = spatial_info.permute(0, 2, 3, 1)
        observed_mask = observed_mask.permute(0, 2, 3, 1) # B, N, K, L
        gt_mask = gt_mask.permute(0, 2, 3, 1) # B, N, K, L


        cut_length = torch.zeros(len(observed_data)).long().cuda()
        if self.is_ignnk:
            return (
                observed_data,
                spatial_info,
                observed_mask,
                observed_tp,
                gt_mask,
                neighbor_location,
                cut_length,
                gt_intact,
                mean_loc,
                std_loc
            )
        elif self.is_separate:
            if self.is_dit_ca2:
                return (
                    observed_data,
                    spatial_info,
                    observed_mask,
                    observed_tp,
                    gt_mask,
                    neighbor_location,
                    cut_length,
                    gt_intact,
                    missing_data,
                    missing_data_mask,
                    missing_data_loc,
                    mean_loc,
                    std_loc
                    
                )
            else:
                return (
                    observed_data,
                    spatial_info,
                    observed_mask,
                    observed_tp,
                    gt_mask,
                    neighbor_location,
                    cut_length,
                    gt_intact,
                    missing_data,
                    missing_data_mask,
                    missing_data_loc,
                    mean_loc,
                    std_loc
                )
        else:
            return (
                observed_data,
                spatial_info,
                observed_mask,
                observed_tp,
                gt_mask,
                neighbor_location,
                cut_length,
                gt_intact
            )
        
class DynaSTI_METRLA(Diffusion_base):
    def __init__(self, config, device, n_spatial):
        super(DynaSTI_METRLA, self).__init__(config, device, n_spatial)

    def process_data(self, batch):
        if self.is_pristi:
            observed_data = batch["observed_data_pristi"].cuda().float() #.to(self.device).float()
            observed_mask = batch["observed_mask_pristi"].cuda().float() #.to(self.device).float()
            gt_mask = batch["gt_mask_pristi"].cuda().float() #.to(self.device).float()
        else:
            observed_data = batch["observed_data"].cuda().float() #.to(self.device).float()
            observed_mask = batch["observed_mask"].cuda().float() #.to(self.device).float()
            gt_mask = batch["gt_mask"].cuda().float() #.to(self.device).float()

        spatial_info = batch["spatial_info"].cuda().float() #.to(self.device).float()
        if self.is_separate:
            missing_data = batch["missing_data"].cuda().float() if "missing_data" in batch.keys() else None
            # print(f"missing_data in data: {missing_data}")
            missing_data_mask = batch["missing_data_mask"].cuda().float() if "missing_data_mask" in batch.keys() else None
            missing_data_loc = batch['missing_data_loc'].cuda().float() if "missing_data_loc" in batch .keys() else None
            # print(f"missing loc process: {missing_data_loc}")
            if missing_data is not None and missing_data_mask is not None and missing_data_loc is not None:
                missing_data = missing_data.permute(0, 2, 3, 1)
                missing_data_mask = missing_data_mask.permute(0, 2, 3, 1)
                # missing_data_loc = missing_data_loc.permute(0, 2, 3, 1)
        
        observed_tp = batch["timepoints"].cuda().float()
        if "neighbor_location" in batch.keys() and batch["neighbor_location"] is not None:
            neighbor_location = batch["neighbor_location"][0]
        else:
            neighbor_location = None
        if batch["gt_intact"] is not None and not isinstance(batch["gt_intact"], list) and len(batch["gt_intact"]) != 0:
            gt_intact = batch["gt_intact"].cuda().float() #.cuda().float() #.to(self.device).float()
        else:
            gt_intact = None

        # if self.is_ignnk or self.is_dit_ca2:
        mean_loc = batch["mean_loc"].cuda().float() #.cuda().float() #.to(self.device).float()
        std_loc = batch['std_loc'].cuda().float() #.cuda().float() #.to(self.device).float()

        observed_data = observed_data.permute(0, 2, 3, 1) # B, N, K, L
        # spatial_info = spatial_info.permute(0, 2, 3, 1)
        observed_mask = observed_mask.permute(0, 2, 3, 1) # B, N, K, L
        gt_mask = gt_mask.permute(0, 2, 3, 1) # B, N, K, L


        cut_length = torch.zeros(len(observed_data)).long().cuda()
        if self.is_ignnk:
            return (
                observed_data,
                spatial_info,
                observed_mask,
                observed_tp,
                gt_mask,
                neighbor_location,
                cut_length,
                gt_intact,
                mean_loc,
                std_loc
            )
        elif self.is_separate:
            if self.is_dit_ca2:
                return (
                    observed_data,
                    spatial_info,
                    observed_mask,
                    observed_tp,
                    gt_mask,
                    neighbor_location,
                    cut_length,
                    gt_intact,
                    missing_data,
                    missing_data_mask,
                    missing_data_loc,
                    mean_loc,
                    std_loc
                )
            else:
                return (
                    observed_data,
                    spatial_info,
                    observed_mask,
                    observed_tp,
                    gt_mask,
                    neighbor_location,
                    cut_length,
                    gt_intact,
                    missing_data,
                    missing_data_mask,
                    missing_data_loc,
                    mean_loc,
                    std_loc
                )
        else:
            return (
                observed_data,
                spatial_info,
                observed_mask,
                observed_tp,
                gt_mask,
                neighbor_location,
                cut_length,
                gt_intact
            )

class DynaSTI_Synth(Diffusion_base):
    def __init__(self, config, device, n_spatial):
        super(DynaSTI_Synth, self).__init__(config, device, n_spatial)

    def process_data(self, batch):
        
        observed_data = batch["observed_data"].cuda().float() #.to(self.device).float()
        observed_mask = batch["observed_mask"].cuda().float() #.to(self.device).float()
        gt_mask = batch["gt_mask"].cuda().float() #.to(self.device).float()

        spatial_info = batch["spatial_info"].cuda().float() #.to(self.device).float()
        if self.is_separate:
            missing_data = batch["missing_data"].cuda().float() if "missing_data" in batch.keys() else None
            # print(f"missing_data in data: {missing_data}")
            missing_data_mask = batch["missing_data_mask"].cuda().float() if "missing_data_mask" in batch.keys() else None
            missing_data_loc = batch['missing_data_loc'].cuda().float() if "missing_data_loc" in batch .keys() else None
            # print(f"missing loc process: {missing_data_loc}")
            if missing_data is not None and missing_data_mask is not None and missing_data_loc is not None:
                missing_data = missing_data.permute(0, 2, 3, 1)
                missing_data_mask = missing_data_mask.permute(0, 2, 3, 1)
                # missing_data_loc = missing_data_loc.permute(0, 2, 3, 1)
        
        observed_tp = batch["timepoints"].cuda().float()
        if "neighbor_location" in batch.keys() and batch["neighbor_location"] is not None:
            neighbor_location = batch["neighbor_location"][0]
        else:
            neighbor_location = None
        if batch["gt_intact"] is not None and not isinstance(batch["gt_intact"], list) and len(batch["gt_intact"]) != 0:
            gt_intact = batch["gt_intact"].cuda().float() #.cuda().float() #.to(self.device).float()
        else:
            gt_intact = None

        # if self.is_ignnk or self.is_dit_ca2:
        mean_loc = batch["mean_loc"].cuda().float() #.cuda().float() #.to(self.device).float()
        std_loc = batch['std_loc'].cuda().float() #.cuda().float() #.to(self.device).float()

        observed_data = observed_data.permute(0, 2, 3, 1) # B, N, K, L
        # spatial_info = spatial_info.permute(0, 2, 3, 1)
        observed_mask = observed_mask.permute(0, 2, 3, 1) # B, N, K, L
        gt_mask = gt_mask.permute(0, 2, 3, 1) # B, N, K, L


        cut_length = torch.zeros(len(observed_data)).long().cuda()
        if self.is_ignnk:
            return (
                observed_data,
                spatial_info,
                observed_mask,
                observed_tp,
                gt_mask,
                neighbor_location,
                cut_length,
                gt_intact,
                mean_loc,
                std_loc
            )
        elif self.is_separate:
            if self.is_dit_ca2:
                return (
                    observed_data,
                    spatial_info,
                    observed_mask,
                    observed_tp,
                    gt_mask,
                    neighbor_location,
                    cut_length,
                    gt_intact,
                    missing_data,
                    missing_data_mask,
                    missing_data_loc,
                    mean_loc,
                    std_loc
                )
            else:
                return (
                    observed_data,
                    spatial_info,
                    observed_mask,
                    observed_tp,
                    gt_mask,
                    neighbor_location,
                    cut_length,
                    gt_intact,
                    missing_data,
                    missing_data_mask,
                    missing_data_loc,
                    mean_loc,
                    std_loc
                )
        else:
            return (
                observed_data,
                spatial_info,
                observed_mask,
                observed_tp,
                gt_mask,
                neighbor_location,
                cut_length,
                gt_intact
            )
        
class DynaSTI_PEMSBAY(Diffusion_base):
    def __init__(self, config, device, n_spatial):
        super(DynaSTI_PEMSBAY, self).__init__(config, device, n_spatial)

    def process_data(self, batch):
        if self.is_pristi:
            observed_data = batch["observed_data_pristi"].cuda().float() #.to(self.device).float()
            observed_mask = batch["observed_mask_pristi"].cuda().float() #.to(self.device).float()
            gt_mask = batch["gt_mask_pristi"].cuda().float() #.to(self.device).float()
        else:
            observed_data = batch["observed_data"].cuda().float() #.to(self.device).float()
            observed_mask = batch["observed_mask"].cuda().float() #.to(self.device).float()
            gt_mask = batch["gt_mask"].cuda().float() #.to(self.device).float()

        spatial_info = batch["spatial_info"].cuda().float() #.to(self.device).float()
        if self.is_separate:
            missing_data = batch["missing_data"].cuda().float() if "missing_data" in batch.keys() else None
            # print(f"missing_data in data: {missing_data}")
            missing_data_mask = batch["missing_data_mask"].cuda().float() if "missing_data_mask" in batch.keys() else None
            missing_data_loc = batch['missing_data_loc'].cuda().float() if "missing_data_loc" in batch .keys() else None
            # print(f"missing loc process: {missing_data_loc}")
            if missing_data is not None and missing_data_mask is not None and missing_data_loc is not None:
                missing_data = missing_data.permute(0, 2, 3, 1)
                missing_data_mask = missing_data_mask.permute(0, 2, 3, 1)
                # missing_data_loc = missing_data_loc.permute(0, 2, 3, 1)
        
        observed_tp = batch["timepoints"].cuda().float()
        if "neighbor_location" in batch.keys() and batch["neighbor_location"] is not None:
            neighbor_location = batch["neighbor_location"][0]
        else:
            neighbor_location = None
        if batch["gt_intact"] is not None and not isinstance(batch["gt_intact"], list) and len(batch["gt_intact"]) != 0:
            gt_intact = batch["gt_intact"].cuda().float() #.cuda().float() #.to(self.device).float()
        else:
            gt_intact = None

        # if self.is_ignnk or self.is_dit_ca2:
        mean_loc = batch["mean_loc"].cuda().float() #.cuda().float() #.to(self.device).float()
        std_loc = batch['std_loc'].cuda().float() #.cuda().float() #.to(self.device).float()

        observed_data = observed_data.permute(0, 2, 3, 1) # B, N, K, L
        # spatial_info = spatial_info.permute(0, 2, 3, 1)
        observed_mask = observed_mask.permute(0, 2, 3, 1) # B, N, K, L
        gt_mask = gt_mask.permute(0, 2, 3, 1) # B, N, K, L


        cut_length = torch.zeros(len(observed_data)).long().cuda()
        if self.is_ignnk:
            return (
                observed_data,
                spatial_info,
                observed_mask,
                observed_tp,
                gt_mask,
                neighbor_location,
                cut_length,
                gt_intact,
                mean_loc,
                std_loc
            )
        elif self.is_separate:
            if self.is_dit_ca2:
                return (
                    observed_data,
                    spatial_info,
                    observed_mask,
                    observed_tp,
                    gt_mask,
                    neighbor_location,
                    cut_length,
                    gt_intact,
                    missing_data,
                    missing_data_mask,
                    missing_data_loc,
                    mean_loc,
                    std_loc
                )
            else:
                return (
                    observed_data,
                    spatial_info,
                    observed_mask,
                    observed_tp,
                    gt_mask,
                    neighbor_location,
                    cut_length,
                    gt_intact,
                    missing_data,
                    missing_data_mask,
                    missing_data_loc,
                    mean_loc,
                    std_loc
                )
        else:
            return (
                observed_data,
                spatial_info,
                observed_mask,
                observed_tp,
                gt_mask,
                neighbor_location,
                cut_length,
                gt_intact
            )