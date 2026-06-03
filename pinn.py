from xml.parsers.expat import model

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import os

if not os.path.exists("model_checkpoints"):
    os.makedirs("model_checkpoints")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")
torch.set_default_dtype(torch.float32)


class ODEData:
    def __init__(self, m, c, k, t_start, t_end, n_points, x0, noise=0.1):
        self.m = m
        self.c = c
        self.k = k
        self.t_start = t_start
        self.t_end = t_end
        self.n_points = n_points
        self.x0 = x0
        self.noise = noise

        self.t = self._generate_time()

        if self.noise is None or self.noise == 0:
            self.solution, self.velocity = self._solve_ode()
        else:
            self.solution, self.velocity = self._add_noise(self._solve_ode())


    def _generate_time(self):
        return np.linspace(self.t_start, self.t_end, self.n_points)

    def _solve_ode(self):
        from scipy.integrate import solve_ivp

        def ode(t, x):
            y1, y2 = x
            dy1_dt = y2
            dy2_dt = -(self.c * y2 + self.k * y1) / self.m
            return [dy1_dt, dy2_dt]

        sol = solve_ivp(ode, (self.t_start, self.t_end), self.x0, t_eval=self.t)
        return sol.y

    def _add_noise(self, x):
        if self.noise is not None:
            return x + np.random.normal(0, self.noise, size=x.shape)
        else:
            return x


class PINN(nn.Module):
    def __init__(self, neurons=64, m=1.0, c_init=0.34, k_init=5.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, neurons),
            nn.Tanh(),
            nn.Linear(neurons, neurons),
            nn.Tanh(),
            nn.Linear(neurons, 1),
        )
        self.m = m
        self.c_train = torch.nn.Parameter(torch.tensor([c_init]), requires_grad=True)
        self.k_train = torch.nn.Parameter(torch.tensor([k_init]), requires_grad=True)

    def forward(self, t):
        return self.net(t)

    def derivatives(self, t):
        x = self(t)
        x_t = torch.autograd.grad(outputs=x, inputs=t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        x_tt = torch.autograd.grad(outputs=x_t, inputs=t, grad_outputs=torch.ones_like(x_t), create_graph=True)[0]
        return x, x_t, x_tt

    def physics_residual(self, t):
        x, x_t, x_tt = self.derivatives(t)
        return self.m * x_tt + self.c_train * x_t + self.k_train * x


def train_pinn(
    model,
    data,
    epochs=50000,
    lr_model=1e-3,
    lr_phys_params=1e-4,
    lambda_physics=2.0,
    lambda_data=5.0,
    lambda_ic=6.0,
):
    time = torch.tensor(data.t, dtype=torch.float32, device=device).unsqueeze(1)
    time.requires_grad_(True)

    reference = torch.tensor(
        data.solution, dtype=torch.float32, device=device
    ).unsqueeze(1)
    t0 = torch.tensor([[0.0]], dtype=torch.float32, device=device)
    v0 = torch.tensor(data.velocity[0], dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(
        [
            {"params": model.net.parameters(), "lr": lr_model},
            {"params": [model.c_train, model.k_train], "lr": lr_phys_params},
        ]
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.8)

    history = {"c": [], "k": [], "loss": [], "physics": [], "data": [], "ic": []}

    for epoch in range(epochs):
        optimizer.zero_grad()
        t0.requires_grad_(True)
        
        x, x_t, x_tt = model.derivatives(time)
        x0_pred, x_t0, _ = model.derivatives(t0)

        physics_residual = model.m * x_tt + model.c_train * x_t + model.k_train * x
        physics_loss = torch.mean(physics_residual ** 2)

        data_loss = torch.mean((x - reference) ** 2)
        ic_loss = (x0_pred - 1) ** 2 + (x_t0[0] - v0) ** 2
        loss = (
            lambda_physics * physics_loss
            + lambda_data * data_loss
            + lambda_ic * ic_loss
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            history["c"].append(model.c_train.item())
            history["k"].append(model.k_train.item())
            history["loss"].append(loss.item())
            history["physics"].append(physics_loss.item())
            history["data"].append(data_loss.item())
            history["ic"].append(ic_loss.item())

        if epoch % 5000 == 0:
            print(
                f"Epoch {epoch}, Physics Loss: {physics_loss.item():.4f}, "
                f"Data Loss: {data_loss.item():.4f}, IC Loss: {ic_loss.item():.4f}, "
                f"Learning Rate: {optimizer.param_groups[0]['lr'] * 100:.6f}%, "
                f"c: {model.c_train.item():.4f}, k: {model.k_train.item():.4f}"
            )

    return model, history, time, x.detach()


def plot_results(data, prediction, history, filename="results.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=450)
    # Linker subplot
    ax1.plot(data.t, data.solution, label="Reference Data", color="blue")
    ax1.plot(
        data.t, prediction, label="PINN Prediction", color="orange", linestyle="dashed"
    )
    ax1.set_title("Reference Data vs. PINN Prediction")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Extension (m)")
    ax1.legend()
    # Rechter subplot
    ax2.plot(history["c"], label="c", color="red")
    ax2.plot(history["k"], label="k", color="green")
    ax2.set_title("Parameter values over epochs")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Parameter Value")
    ax2.legend()

    plt.savefig(filename, dpi=450, bbox_inches="tight")


def main(noise, filename):
    np.random.seed(42)
    torch.manual_seed(42)

    data = ODEData(
        m=1.0,
        c=0.4,
        k=4.0,
        t_start=0,
        t_end=10,
        n_points=1000,
        x0=[1.0, 0.0],
        noise=noise,
    )

    model = PINN(neurons=128, m=data.m, c_init=data.c * 0.85, k_init=data.k * 1.3).to(device)

    model, history, _, prediction = train_pinn(model, data, epochs=50000)

    # Save the model checkpoint including the optimized parameters
    torch.save({
    "model_state_dict": model.state_dict(),
    "c": model.c_train.detach().cpu(),
    "k": model.k_train.detach().cpu(),
    }, f"model_checkpoints/pinn_checkpoint_{filename.split('.')[0]}.pth")

    print(
        f"Optimized parameters: c = {model.c_train.item():.4f}, k = {model.k_train.item():.4f}"
    )
    plot_results(data, prediction.cpu().numpy(), history, filename=filename)


if __name__ == "__main__":
    main(noise = 0, filename="results.png")
    # main(noise= 0.05, filename="results_with_05_noise.png")
    # main(noise = 0.1, filename="results_with_noise.png")
