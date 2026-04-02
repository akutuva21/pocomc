import torch
import numpy as np

# Test Gamma
n_dim = 2
nu = 3.0
n_walkers = 5
diff = np.random.randn(n_walkers, n_dim)
inv_cov = np.eye(n_dim)

quad_form = np.einsum('ki,ij,kj->k', diff, inv_cov, diff)
scale_gamma = 2.0 / (nu + quad_form)

# numpy
np.random.seed(0)
s_np = 1.0 / np.random.gamma((n_dim + nu) / 2, scale_gamma, size=n_walkers)

# torch
torch.manual_seed(0)
scale_gamma_t = torch.as_tensor(scale_gamma)
gamma_dist = torch.distributions.Gamma((n_dim + nu) / 2, 1.0 / scale_gamma_t)
s_t = 1.0 / gamma_dist.sample()

print("numpy", s_np)
print("torch", s_t.numpy())
