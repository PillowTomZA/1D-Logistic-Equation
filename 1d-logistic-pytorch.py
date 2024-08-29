import torch
from torch import nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import cm

# Define a simple neural network for regression
class simple_NN(nn.Module):
    def __init__(self):
        super(simple_NN,self).__init__()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(1,16),
            nn.Tanh(),
            nn.Linear(16,32),
            nn.Tanh(),
            nn.Linear(32,16),
            nn.Tanh(),
            nn.Linear(16,1),
        )
    def forward(self,x):
        out = self.linear_tanh_stack(x)
        return out
# Define dataset
x_train = torch.tensor([[1.1437e-04],
        [1.4676e-01],
        [3.0233e-01],
        [4.1702e-01],
        [7.2032e-01]], dtype=torch.float32)
y_train = torch.tensor([[1.0000],
        [1.0141],
        [1.0456],
        [1.0753],
        [1.1565]], dtype=torch.float32)

# PINN
domain = [0.0, 1.5]
x_eval = torch.linspace(domain[0], domain[1], steps=100).reshape(-1, 1)
R = 1.0
ft0 = 1.0

def df(f: simple_NN, x: torch.Tensor = None, order: int = 1) -> torch.Tensor: # type: ignore
    """ compute neural network derivative with respect to input features using PyTorch autograd engine"""
    df_value = f(x)
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            x,
            grad_outputs=torch.ones_like(x),
            create_graph=True,
            retain_graph=True,
        )[0]

    return df_value
t = torch.linspace(domain[0],domain[1], steps=10, requires_grad=True).reshape(-1,1)
def compute_loss(nn: simple_NN,
                 t: torch.Tensor = None, # type: ignore
                 x: torch.Tensor = None, # type: ignore
                 y: torch.Tensor = None, # type: ignore
                 ) -> torch.float: # type: ignore
    """ Compute the full loss function as pde loss + boundary loss
    theis custom loss function is fully define with differentiable tensors therefore
    the backboard method can be applied to it
    """

    pde_loss = df(nn,t) - R * t * (1 - t)
    pde_loss = pde_loss.pow(2).mean()

    boundary = torch.Tensor([0.0])
    boundary.requires_grad = True
    bc_loss = nn(boundary) - ft0
    bc_loss = bc_loss.pow(2)

    mse_loss = torch.nn.MSELoss()(nn(x),y)

    tot_loss = pde_loss + bc_loss + mse_loss

    return tot_loss

model = simple_NN()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)

# Train
for ep in range(5000):

    loss = compute_loss(model,t, x_train, y_train)

    #Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if ep % 200 == 0:
        print(f"Epoch: {ep}, loss: {loss.item():>7f}")

# numeric solution
def logistic_eq_fn(x, y):
    return R * x * (1 - x)

numeric_solution = solve_ivp(
    logistic_eq_fn, domain, [ft0], t_eval=x_eval.squeeze().detach().numpy()
)

f_colloc = solve_ivp(
    logistic_eq_fn, domain, [ft0], t_eval=t.squeeze().detach().numpy()
).y.T

# evaluation on the domain [0, 1.5]
f_eval = model(x_eval)

# plotting
fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(t.detach().numpy(), f_colloc, label="Collocation points", color="magenta", alpha=0.75)
ax.scatter(x_train.detach().numpy(), y_train.detach().numpy(), label="Observation data", color="blue")
ax.plot(x_eval.detach().numpy(), f_eval.detach().numpy(), label="NN solution", color="black")
ax.plot(x_eval.detach().numpy(), numeric_solution.y.T,
        label="Analytic solution", color="magenta", alpha=0.75)
ax.set(title="Logistic equation solved with NNs", xlabel="t", ylabel="f(t)")
ax.legend();
plt.show()