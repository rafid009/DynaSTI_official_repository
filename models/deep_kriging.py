import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QuantileLoss(nn.Module):
    """
    Quantile loss for multi-feature, multi-quantile outputs.
    Computes the pinball loss across features and quantile levels.

    Args:
        quantiles (list or torch.Tensor): List or tensor of quantile levels (values between 0 and 1).
    """
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float64)
        if self.quantiles.dim() == 0:
            self.quantiles = self.quantiles.unsqueeze(0)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds (torch.Tensor): Predicted quantiles, shape [batch_size, num_features, num_quantiles].
            target (torch.Tensor): True target values, shape [batch_size, num_features].

        Returns:
            torch.Tensor: Scalar loss (mean over batch, features, and quantiles).
        """
        if preds.dim() != 3:
            raise ValueError(f"Expected preds to be 3D [batch, K, Q], got {preds.shape}")

        # Move quantiles to the same device as preds
        quantiles = self.quantiles.to(preds.device)

        # Expand target to [batch, K, Q] for broadcasting
        target_expanded = target.unsqueeze(-1)  # [batch, K, 1]

        # Compute errors: difference between predicted quantiles and true values
        # print(f"preds: {preds.shape}, target: {target.shape}")
        errors = preds - target_expanded       # [batch, K, Q]

        # Compute pinball loss: max(q * error, (q - 1) * error)
        # quantiles shape [Q] -> reshape [1, 1, Q] for broadcasting
        q = quantiles.view(1, 1, -1)
        loss = torch.max(q * errors, (q - 1) * errors)

        # Return mean loss over batch, features, and quantiles
        return loss.mean()

class SpatioTemporalDataset(Dataset):
    """
    PyTorch Dataset for spatio-temporal interpolation tasks.

    Args:
        coords (torch.Tensor): Tensor of shape (N, 2) with spatial coordinates.
        times (torch.Tensor): Tensor of shape (N,) with temporal coordinates.
        values (torch.Tensor): Tensor of shape (N,) with target values.
        embedder (nn.Module): A module that takes (coords, times) and returns
                             embedded features of shape (batch, K).
        covariates (torch.Tensor, optional): Tensor of shape (N, P) with additional
                                           covariates per sample.
        precompute (bool): If True, precompute all embeddings in memory for fast access.
    """
    def __init__(self, coords: torch.Tensor, times: torch.Tensor, values: torch.Tensor,
                 embedder: torch.nn.Module, covariates: torch.Tensor = None,
                 precompute: bool = True):
        super().__init__()
        assert coords.size(0) == times.size(0) == values.size(0), \
            "coords, times, and values must have the same number of samples"
        if covariates is not None:
            assert covariates.size(0) == coords.size(0), \
                "covariates must have the same number of samples as coords/times"

        self.coords = coords
        self.times = times
        self.values = values
        self.covariates = covariates
        self.embedder = embedder
        self.precompute = precompute

        if self.precompute:
            # Precompute embeddings for all samples
            with torch.no_grad():
                embedded = self.embedder(self.coords, self.times)

                if self.covariates is not None:
                    embedded = torch.cat([embedded, self.covariates], dim=1)
                self.features = embedded  # shape: (N, K_total)

    def __len__(self):
        return self.coords.size(0)

    def __getitem__(self, idx: int):
        # Get embedded features (precomputed or on-the-fly)
        if self.precompute:
            feat = self.features[idx]
        else:
            coord = self.coords[idx].unsqueeze(0)  # shape: (1, 2)
            time = self.times[idx].unsqueeze(0)     # shape: (1,)
            feat = self.embedder(coord, time).squeeze(0)
            # print(f"feat: {feat.shape}\nfeat_one_hot: {self.feat_one_hot.shape}")
            # feat = torch.cat([feat, self.feat_one_hot[idx]], dim=0)

            if self.covariates is not None:
                feat = torch.cat([feat, self.covariates[idx]], dim=0)

        # Retrieve target value
        target = self.values[idx]
        return feat, target


class WendlandBasis(nn.Module):
    """Spatial radial basis functions using Wendland kernel."""
    def __init__(self, knots: torch.Tensor, theta: float):
        super().__init__()
        self.register_buffer('knots', knots)    # shape [K_s, 2]
        self.theta = theta
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [batch, 2]
        # Compute pairwise distance to each knot
        diff = coords.unsqueeze(1) - self.knots.unsqueeze(0)   # [batch, K_s, 2]
        dist = torch.linalg.norm(diff, dim=2) / self.theta     # [batch, K_s]
        # Wendland kernel: (1 - r)^6 * (35r^2 + 18r + 3) / 3 for r <= 1
        r = torch.clamp(dist, max=1.0)
        wendland_vals = ((1 - r)**6) * (35*r**2 + 18*r + 3) / 3.0
        # values outside support (dist>theta) are already zeroed by clamp
        return wendland_vals

class GaussianTimeBasis(nn.Module):
    """Temporal radial basis functions using Gaussian kernels."""
    def __init__(self, centers: torch.Tensor, scales: torch.Tensor):
        super().__init__()
        self.register_buffer('centers', centers)  # [K_t]
        self.register_buffer('scales', scales)    # [K_t]
    def forward(self, times: torch.Tensor) -> torch.Tensor:
        # times: [batch] or [batch, 1]
        diff = times.unsqueeze(1) - self.centers.unsqueeze(0)  # [batch, K_t]
        # Gaussian RBF
        vals = torch.exp(-0.5 * (diff / self.scales.unsqueeze(0))**2)
        return vals

class SpatioTemporalEmbedding(nn.Module):
    """Combines spatial and temporal RBF embeddings."""
    def __init__(self, spatial_basis: WendlandBasis, temporal_basis: GaussianTimeBasis):
        super().__init__()
        self.spatial = spatial_basis
        self.temporal = temporal_basis
    def forward(self, coords: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        # coords: [batch, 2], times: [batch]
        phi_space = self.spatial(coords)    # [batch, K_s]
        phi_time = self.temporal(times)     # [batch, K_t]
        return torch.cat([phi_space, phi_time], dim=1)  # [batch, K_s+K_t]

class DeepKrigingModel(nn.Module):
    """
    DeepKrigingModel supporting multi-output interpolation with multiple quantiles per output.

    Args:
        input_dim (int): Dimension of input features (K_s + K_t + optional covariates/feature embed dims).
        num_features (int): Number of target features (K).
        quantiles (list[float]): List of quantile levels (e.g., [0.1, 0.5, 0.9]).
        hidden_dims (list[int]): Sizes of hidden layers.
        lambda_val (float): Scaling for non-crossing quantile transformation.
    """
    def __init__(self, input_dim: int, num_features: int, quantiles: list, hidden_dims: list, lambda_val: float):
        super().__init__()
        self.num_features = num_features
        self.quantiles = quantiles
        self.lambda_val = lambda_val
        self.num_quantiles = len(quantiles)
        
        # Build feed-forward network
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        # Final layer outputs num_features * num_quantiles
        layers.append(nn.Linear(prev_dim, num_features * self.num_quantiles))
        self.net = nn.Sequential(*layers)

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight)
        self.apply(_basic_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input features, shape [batch, input_dim].
        
        Returns:
            torch.Tensor: Quantile predictions, shape [batch, num_features, num_quantiles].
        """
        # Raw network output: [batch, K * Q]
        # print(f"x type: {x.dtype}, x shape: {x.shape}")
        raw = self.net(x)  
        # Reshape to [batch, num_features, num_quantiles]
        batch_size = raw.shape[0]
        raw = raw.view(batch_size, self.num_features, self.num_quantiles)
        
        # Apply non-crossing Ψ transform across quantiles for each feature
        # Extract median index (expected to be in quantiles list)
        median_idx = self.quantiles.index(0.5)
        med = raw[:, :, median_idx:median_idx+1]  # [batch, K, 1]
        
        # Prepare output tensor
        out = torch.zeros_like(raw)
        for q_idx, tau in enumerate(self.quantiles):
            if tau == 0.5:
                # Median: identity
                out[:, :, q_idx] = med.squeeze(-1)
            elif tau > 0.5:
                # Upper quantiles
                offset = raw[:, :, q_idx:q_idx+1]
                out[:, :, q_idx] = (med + (self.lambda_val * (tau - 0.5))
                                   / (1 + torch.exp(-offset))).squeeze(-1)
            else:
                # Lower quantiles
                offset = raw[:, :, q_idx:q_idx+1]
                out[:, :, q_idx] = (med - (self.lambda_val * (0.5 - tau))
                                   / (1 + torch.exp(-offset))).squeeze(-1)
        return out
    
    
    
def create_basis_embeddings():
    # Define spatial and temporal basis parameters (knots and scales)
    spatial_knots = torch.stack(torch.meshgrid(
        torch.linspace(0,1,5), torch.linspace(0,1,5)
    )).reshape(2, -1).T  # e.g., 5x5 grid = 25 knots
    spatial_theta = 0.5  # support radius
    W_basis = WendlandBasis(spatial_knots, spatial_theta)

    time_centers = torch.linspace(0,1,50)  # e.g., 50 time knots
    time_scales = torch.full((50,), 0.1)   # Gaussian width for each (could vary)
    T_basis = GaussianTimeBasis(time_centers, time_scales)

    embed_layer = SpatioTemporalEmbedding(W_basis, T_basis)
    return embed_layer, W_basis, T_basis

def get_model(num_features):
    embed_layer, W_basis, T_basis = create_basis_embeddings()
    # 3. Initialize model and loss
    quantiles = [0.1, 0.5, 0.9]
    model = DeepKrigingModel(input_dim = W_basis.knots.shape[0] + T_basis.centers.shape[0], num_features=num_features,
                            quantiles = quantiles,
                            hidden_dims = [100, 100, 100, 100, 100, 100, 100, 100, 50, 50, 50, 50],
                            lambda_val = 1.0)  # lambda can be chosen ~ half data range
    model = model.double().to(device)
    return model

def train_deep_kriging(lr, num_epochs, coords_tensor, times_tensor, values_tensor, num_features, model_file_path):
    embed_layer, W_basis, T_basis = create_basis_embeddings()
    # 2. Create dataset and dataloader
    dataset = SpatioTemporalDataset(coords_tensor, times_tensor, values_tensor, embed_layer)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, pin_memory=True)

    # 3. Initialize model and loss
    quantiles = [0.1, 0.5, 0.9]
    model = DeepKrigingModel(input_dim = W_basis.knots.shape[0] + T_basis.centers.shape[0], num_features=num_features,
                            quantiles = quantiles,
                            hidden_dims = [100, 100, 100, 100, 100, 100, 100, 100, 50, 50, 50, 50],
                            lambda_val = 1.0)  # lambda can be chosen ~ half data range
    model = model.double().to(device)
    loss_fn = QuantileLoss(quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. Training loop (simplified)
    model.train()
    # for epoch in range(num_epochs):
    #     for embedded_x, y in loader:  # dataset returns already embedded features if precomputed
    #         embedded_x = embedded_x.to(device)
    #         y = y.to(device)
    #         optimizer.zero_grad()
    #         q_preds = model(embedded_x)            # forward pass to get quantile predictions
    #         loss = loss_fn(q_preds, y)             # compute sum of quantile losses
    #         loss.backward()                       
    #         optimizer.step()
        # (validate on a held-out set, etc., as needed)

    for epoch in range(1, num_epochs + 1):
        # model.train()
        running_loss = 0.0
        # Initialize tqdm progress bar for this epoch
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar, start=1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            preds = model(inputs)
            # print(f"train loop preds: {preds.shape}, target: {targets.shape}")
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            avg_loss = running_loss / batch_idx
            # Update progress bar postfix with current average loss
            pbar.set_postfix(loss=f"{avg_loss:.4f}", refresh=False)
    torch.save(model.state_dict(), model_file_path)
    return model

def evaluate(model, embed_layer, new_coords_tensor, new_times_tensor):
    # 5. Inference on new data
    model.eval()
    with torch.no_grad():
        new_feats = embed_layer(new_coords_tensor, new_times_tensor)  # embed new locations
        pred_quantiles = model(new_feats)  # get quantile predictions
        # pred_quantiles[:,0] is 10th percentile, [:,1] median, [:,2] 90th percentile
    return pred_quantiles[:,1]


def prepare_data(data_loader):
    # 1. Collect all batches
    observed_data_list = []
    observed_mask_list = []
    spatial_info_list = []
    for batch in data_loader:
        observed_data_list.append(batch['observed_data'])   # shape (b_i, L, N, K)
        observed_mask_list.append(batch['observed_mask'])   # shape (b_i, L, N, K)
        spatial_info_list.append(batch['spatial_info'])     # shape (b_i, N, d)
    
    observed_data = torch.cat(observed_data_list, dim=0)   # (B, L, N, K)
    observed_mask = torch.cat(observed_mask_list, dim=0)   # (B, L, N, K)
    spatial_info  = torch.cat(spatial_info_list,  dim=0)   # (B, N, d)
    
    B, L, N, K = observed_data.shape
    print(f"B: {B}, L: {L}, N: {N}, K: {K}")
    
    # 2. Identify which (b, t, n) have *all* K features observed
    #    observed_mask.sum(dim=-1) == K  would also work
    valid_mask = observed_mask.all(dim=-1)   # shape (B, L, N), True if all K features present
    
    # 3. Flatten out the valid (b, t, n) indices
    #    idx will be of shape (num_valid, 3) with columns [b_idx, t_idx, n_idx]
    idx = valid_mask.nonzero(as_tuple=False)
    b_idx, t_idx, n_idx = idx[:, 0], idx[:, 1], idx[:, 2]
    
    # 4. Gather the coordinates, time, and targets
    #    spatial_info[b, n, :] → (num_valid, d)
    spatial_points = spatial_info[b_idx, n_idx, :]               # (num_valid, d)
    
    #    normalize time if you like, or just keep as integer index
    time_points = t_idx.float() / (L - 1)                        # (num_valid,)
    
    #    observed_data[b, t, n, :] → (num_valid, K)
    targets = observed_data[b_idx, t_idx, n_idx, :]          # (num_valid, K)
    
    return spatial_points, time_points, targets, K




def prepare_test_dataloader_multi(coords: torch.Tensor,
                                  num_time_steps: int,
                                  embedder: torch.nn.Module,
                                  batch_size: int = 1024,
                                  device: torch.device = torch.device('cpu')):
    """
    Prepare a DataLoader for test-time interpolation for multi-output DeepKriging model.

    Args:
        coords (torch.Tensor): Tensor of shape (B, N, d) with spatial coordinates.
        num_time_steps (int): Number of time steps (L).
        embedder (nn.Module): SpatioTemporalEmbedding module.
        batch_size (int): Batch size for DataLoader.
        device (torch.device): Device to perform embedding on (CPU/CUDA).

    Returns:
        DataLoader yielding embedded features of shape (batch, K_total), and a tuple
        (B, N, L) for reshaping outputs back to (B, N, L, K, Q).
    """
    B, N, d = coords.shape
    L = num_time_steps

    if d > 2:
        d = 2
        coords = coords[:,:,:2]
    # 1. Normalize time indices to [0,1]
    times = torch.linspace(0.0, 1.0, steps=L, device=device)  # [L]

    # 2. Expand coords into (B*N*L, d) and times into (B*N*L)
    coords_exp = coords.unsqueeze(2).expand(B, N, L, d).reshape(-1, d)  # [B*N*L, d]
    times_exp = times.view(1, 1, L).expand(B, N, L).reshape(-1)        # [B*N*L]

    # 3. Compute embeddings in batch on device
    embedder = embedder.to(device)
    with torch.no_grad():
        spatial_embed = embedder.spatial(coords_exp.to(device))   # [B*N*L, K_s]
        temporal_embed = embedder.temporal(times_exp.to(device))  # [B*N*L, K_t]
        embedded = torch.cat([spatial_embed, temporal_embed], dim=1)  # [B*N*L, K_s + K_t]

    # Move embeddings to CPU for DataLoader
    embedded = embedded.cpu()

    # 4. Create DataLoader (inputs only)
    dataset = TensorDataset(embedded)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return loader, (B, N, L)


def test_deepkriging_model_multi(model: torch.nn.Module,
                                 dataloader: DataLoader,
                                 reshape_dims: tuple,
                                 device: torch.device = torch.device('cpu')):
    """
    Run the trained multi-output DeepKriging model on test data and reshape predictions.

    Args:
        model (nn.Module): Trained DeepKrigingModelMultiOutput (outputs shape [batch, K, Q]).
        dataloader (DataLoader): DataLoader from prepare_test_dataloader_multi.
        reshape_dims (tuple): Tuple (B, N, L) to reshape flat outputs.
        device (torch.device): Device for inference.

    Returns:
        torch.Tensor: Predictions reshaped to (B, N, L, K, Q), where K = num_features,
                      Q = number of quantiles per feature.
    """
    model = model.to(device).eval()  # ensure on correct device
    all_preds = []

    with torch.no_grad():
        for (batch_embed,) in dataloader:
            batch_embed = batch_embed.to(device)
            preds = model(batch_embed)  # [batch, K, Q]
            all_preds.append(preds.detach())

    # Concatenate batch outputs
    all_preds = torch.cat(all_preds, dim=0)  # [B*N*L, K, Q]

    B, N, L = reshape_dims
    K, Q = all_preds.shape[1], all_preds.shape[2]

    # Reshape to (B, N, L, K, Q)
    predictions = all_preds.view(B, N, L, K, Q)[:, :, :, :, 1]
    return predictions