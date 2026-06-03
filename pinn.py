import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Parameter ---
m, c, k = 1.0, 0.4, 4.0
lambda_physics = 2
lambda_data = 5.0
lambda_ic = 6

# --- Netzwerk ---
neurons = 64
np.random.seed(42)
torch.manual_seed(42)

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, neurons), nn.Tanh(),
            nn.Linear(neurons, neurons), nn.Tanh(),
            nn.Linear(neurons, 1)
        )
    
    def forward(self, t):
        return self.net(t)

c_train = torch.nn.Parameter(torch.tensor([c*0.85]), requires_grad=True).to(device)
k_train = torch.nn.Parameter(torch.tensor([k*1.3]), requires_grad=True).to(device)
model = PINN().to(device)

optimizer = torch.optim.Adam(
    [{"params": model.parameters()},
     {"params": [c_train, k_train], "lr": 1e-4}],
     lr=1e-3
    )

class ODEData:
    def __init__(self, t, x):
        self.t = t
        self.x = x

        result = self.solver((t[0], t[-1]), x)
        self.solution = result[0]
        self.velocity = result[1]

    def solver(self, t_span, x0):
        from scipy.integrate import solve_ivp
        def ode(t, x):
            y1, y2 = x
            dy1_dt = y2
            dy2_dt = -(c*y2 + k*y1) / m
            return [dy1_dt, dy2_dt]
        sol = solve_ivp(ode, t_span, x0, t_eval=self.t)
        return sol.y
    
if __name__ == "__main__":
    
    # 1. Referenzlösung generieren
    time = np.linspace(0, 10, 500)
    x0 = [1.0, 0.0]  # Anfangsbedingungen
    data = ODEData(t = time, x = x0)
   
    # 2. Physics Loss implementieren
    reference = torch.tensor(data.solution, dtype=torch.float32).unsqueeze(1).to(device)

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=2000, 
        gamma=0.8
        )

    # sammeln der trainigs iterationen
    c_history = []
    k_history = []

    # Training Loop
    for epoch in range(50000):
        optimizer.zero_grad()
        # time wird hier als torch tensor neu definiert 
        time = torch.linspace(0,10,500).unsqueeze(1).to(device)
        time.requires_grad_(True)

        x = model(time)
        x_t = torch.autograd.grad(
            outputs = x,
            inputs = time,
            grad_outputs = torch.ones_like(x),
            create_graph = True
        )
        x_tt = torch.autograd.grad(
            outputs = x_t[0],
            inputs = time,
            grad_outputs = torch.ones_like(x_t[0]),
            create_graph = True
        ) 
        # Anfangsbedingungen
        t0 = torch.tensor([[0.0]], requires_grad=True).to(device)
        x0_pred = model(t0)
        x_t0 = torch.autograd.grad(
            outputs = x0_pred,
            inputs = t0,
            grad_outputs = torch.ones_like(x0_pred),
            create_graph = True
        )
        v0 = torch.tensor(data.velocity[0], dtype=torch.float32).to(device)

        
        # ODE-Residuum    
        residuum = m*x_tt[0] + c_train*x_t[0] + k_train*x
        physics_loss = torch.mean(residuum**2)

        # Data Loss
        data_loss = torch.mean((x - reference)**2)

        # IC Loss
        ic_loss = (x0_pred - 1)**2 + (x_t0[0] - v0)**2

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Physics Loss: {physics_loss.item():.4f}, Data Loss: {data_loss.item():.4f}, IC Loss: {ic_loss.item():.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]*100:.6f}%, c: {c_train.item():.4f}, k: {k_train.item():.4f}')  

        # Gesamtverlust mit Gewichtung
        loss = lambda_physics * physics_loss + lambda_data * data_loss + lambda_ic * ic_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        c_history.append(c_train.item())
        k_history.append(k_train.item())
        

    print(f'Optimierte Parameter: c = {c_train.item():.4f}, k = {k_train.item():.4f}')
    # Beide Ergebnisse in einem Plot vergleichen. 
    # Linker subplot: Referenzlösung vs. Vorhersage des PINN
    # Rechter subplot: Verlauf der Parameter c und k über die Epochen
    fig, (ax1, ax2) = plt.subplots(1, 2,
        figsize=(12, 5),
        dpi=450
    )
    # Linker subplot
    ax1.plot(data.t, data.solution, label='Reference Data', color='blue')
    ax1.plot(time.detach().numpy(), x.detach().numpy(), label='PINN Prediction', color='orange', linestyle='dashed')
    ax1.set_title('Reference Data vs. PINN Prediction')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Extension (m)')
    ax1.legend()
    # Rechter subplot
    ax2.plot(c_history, label='c', color='red')
    ax2.plot(k_history, label='k', color='green')
    ax2.set_title('Parameter values over epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Parameter Value')
    ax2.legend()
    plt.savefig('results.png', dpi=450, bbox_inches='tight')
    