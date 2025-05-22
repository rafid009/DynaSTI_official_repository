import numpy as np
import torch
import gpytorch
from tqdm import trange
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler


# def GP(X, Y):
#     kernel = ConstantKernel(1.0, (1e-3, 1e3)) * (
#         RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), active_dims=[0, 1]) *
#         RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), active_dims=[2])
#     )

#     gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=10)
#     multi_gp = MultiOutputRegressor(gpr)
#     multi_gp.fit(X, Y)
#     return multi_gp

class MultitaskSparseGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_tasks, d=3):

        print(f"in class inducing points: {inducing_points.shape}")

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0),
            batch_shape=torch.Size([num_tasks])
        )
        # Base VariationalStrategy
        base_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        # Wrap it in a MultitaskVariationalStrategy
        multitask_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            base_strategy,
            num_tasks=num_tasks
        )
        super().__init__(multitask_strategy)

        # Mean & kernel modules
         # Task-batched mean and kernel modules
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        # self.mean_module = gpytorch.means.MultitaskMean(
        #     gpytorch.means.ConstantMean(), num_tasks=num_tasks
        # )
        spatial_kernel = gpytorch.kernels.RBFKernel(active_dims=[i for i in range(d)])
        temporal_kernel = gpytorch.kernels.RBFKernel(active_dims=[d])
        base_kernel = spatial_kernel * temporal_kernel
        scaled_kernel = gpytorch.kernels.ScaleKernel(base_kernel)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            scaled_kernel, num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks

        # Multitask mean: each task has its own constant mean.
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        
        # Define the spatial RBF kernel (on x and y, i.e. dims [0, 1])
        spatial_kernel = gpytorch.kernels.RBFKernel(active_dims=[0, 1, 2])
        # Define the temporal RBF kernel (on time, i.e. dim [2])
        temporal_kernel = gpytorch.kernels.RBFKernel(active_dims=[3])
        # The composite kernel: product of spatial and temporal kernels
        base_kernel = spatial_kernel * temporal_kernel
        # Scale the kernel by an overall constant.
        base_kernel = gpytorch.kernels.ScaleKernel(base_kernel)
        # Wrap the base kernel with a multitask kernel to couple outputs.
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            base_kernel, num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        # Compute the mean and covariance for inputs x.
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # Return a MultitaskMultivariateNormal object.
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def trainGP(train_x, train_y, d=3):
    # Define the multitask Gaussian likelihood.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = train_x.float().to(device)
    train_y = train_y.float().to(device)

    # Normalize to [-1, 1] range (better for GP stability)
    x_scaler = StandardScaler().fit(train_x.cpu().numpy())

    train_x = torch.tensor(x_scaler.transform(train_x.cpu().numpy()), dtype=torch.float32, device=device)
    # train_y = torch.tensor(y_scaler.transform(train_y.cpu().numpy()), dtype=torch.float32, device=device)

    # Create data loader with persistent workers
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=1024,
        shuffle=True
    )




    num_tasks = train_y.shape[-1]
    # print(f"num tasks: {num_tasks}")
    N = train_x.shape[0]
    # loader = DataLoader(
    #     TensorDataset(train_x, train_y),
    #     batch_size=512,
    #     shuffle=True
    # )

    # -----------------------------
    # 2. Inducing points (M ≪ N)
    # -----------------------------
    M = 1000
    # assert M <= N, f"{M} and {N}"
    inds = torch.randperm(train_x.shape[0])[:M]
    inducing_points = train_x[inds].clone().to(device)
    # print(f"train y: {train_y.shape}")
    # print(f"inducing_points: {inducing_points}")
    model = MultitaskSparseGP(inducing_points, num_tasks=num_tasks, d=d).to(device)
    # likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(device)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks,
        noise_constraint=gpytorch.constraints.GreaterThan(1e-4)  # Prevent noise collapse
    ).to(device)


    # Use gradient-clipped Adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=0.01)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Stochastic Weight Averaging

    # Use KL-weighted ELBO for better numerical stability
    mll = gpytorch.mlls.PredictiveLogLikelihood(
        likelihood, 
        model, 
        num_data=len(train_y),
        beta=0.1  # KL annealing factor
    )

    likelihood = likelihood.float()
    # Initialize the model.
    # model = MultitaskGPModel(train_x, train_y, likelihood, num_tasks)

    model = model.float()
    
    # -----------------------------
    # 3. Train the model
    # -----------------------------
    # Set the model and likelihood into training mode.
    
    model = model.to(device)
    likelihood = likelihood.to(device)
    model.train()
    likelihood.train()

    # Use the Adam optimizer.
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # # Marginal log likelihood as our loss.
    # # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    # mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_x.shape[0])


    training_iter = 300
    pbar = trange(training_iter, desc="Training", unit="iteration")
    with gpytorch.settings.cholesky_jitter(1e-3):
        for i in pbar:
            total_loss = 0.0
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                print(f"x_batch: {x_batch.shape}")
                output = model(x_batch)
                loss = -mll(output, y_batch)

                if torch.isnan(loss):
                    raise RuntimeError("NaN detected in loss")
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            pbar.set_postfix(avg_ELBO=avg_loss, epoch=i)
    return model, likelihood, x_scaler #, y_scaler

def testGP(model: MultitaskGPModel, likelihood, test_x, x_scaler): #, y_scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

     # 1. Scale test inputs using training data scaler
    test_x_scaled = x_scaler.transform(test_x.cpu().numpy())
    test_tensor = torch.tensor(test_x_scaled, dtype=torch.float32).to(device)

    # 2. Make predictions on scaled data
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(test_tensor))
    #     scaled_mean = preds.mean.detach().clone().cpu().numpy()
    #     scaled_var = preds.variance.detach().clone().cpu().numpy()

    # # 3. Inverse transform predictions
    # # For mean: use inverse_transform
    # original_mean = y_scaler.inverse_transform(scaled_mean)
    
    # # For variance: scale by (y_std)^2 since Var(aX + b) = a²Var(X)
    # scale_sq = np.square(y_scaler.scale_)
    # original_var = scaled_var * scale_sq

    # return torch.tensor(original_mean).to(device), torch.tensor(original_var).to(device)

    # test_x = torch.tensor(test_x).float().to(device)
    # model.eval()
    # likelihood.eval()
    # with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # preds = likelihood(model(test_x))
        # Get the predictive mean and full covariance.
        pred_mean = preds.mean.detach().clone() # .cpu().numpy() if preds.mean.device.type == 'cuda' else preds.mean.detach().numpy() # shape: [n_test, num_tasks]
        pred_cov = preds.covariance_matrix.detach().clone() #.cpu().numpy() if preds.covariance_matrix.device.type == 'cuda' else preds.covariance_matrix.detach().numpy()  # shape: [n_test, num_tasks, n_test, num_tasks]
    return pred_mean, pred_cov