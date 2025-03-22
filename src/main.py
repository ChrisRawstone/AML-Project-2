import torch
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import torch.distributions as td
import torch
from torch import optim
import matplotlib.pyplot as plt

import pdb

class PiecewiseLinearCurve(torch.nn.Module):
    def __init__(self,c_0,c_1,G, n_intervals=10):
        super(PiecewiseLinearCurve, self).__init__()
        self.n_intervals = n_intervals
        self.t_space = torch.linspace(0,1,n_intervals+1)[None,:]
        pertubation = torch.randn(c_0.shape[0],n_intervals-1)*0.1*torch.norm(c_1-c_0,2)
        self.c_free = torch.nn.Parameter((c_1-c_0)*self.t_space[:,1:-1] + c_0+pertubation) # initial value of c as a linear interpolation between c_0 and c_1
        self.c_0 = c_0
        self.c_1 = c_1
        self.G = G
        self.initial_c = self.c().detach().clone()
    
    def c(self):
        return torch.cat([self.c_0, self.c_free,self.c_1],dim=1)

    def curve_grad(self):
        c = self.c()
        diffs = c[:,1:]-c[:,:-1] 
        dt = 1/self.n_intervals
        return diffs/dt
    
    def norm(self,x,y):
        return x.T @ self.G(y) @ x

    def forward(self):
        c = self.c()
        c_dot = self.curve_grad()
        norm2 = 0
        for i in range(self.n_intervals):
            norm2 += self.norm(c_dot[:,i].unsqueeze(1),c[:,i])
        return norm2 / self.n_intervals
    


class PolynomialCurve(torch.nn.Module):
    def __init__(self,c_0,c_1,G, degree=2,n_intervals=10):
        super(PolynomialCurve, self).__init__()
        self.degree = degree
        self.t_points = n_intervals+1
        self.t_space = torch.linspace(0,1,self.t_points)[None,:]
        self.t_tensor = torch.vstack([self.t_space[0]**i for i in range(degree+1)])
        pertubation = torch.randn(c_0.shape[0],degree-1)*3*torch.norm(c_1-c_0,2)
        self.w = torch.nn.Parameter(torch.zeros(c_0.shape[0],degree-1)+pertubation) # initial value of c as a linear interpolation between c_0 and c_1
        
        self.c_0 = c_0
        self.c_1 = c_1
        self.G = G

        self.initial_c = self.c().detach().clone()

    
    def c(self):
        w_0 = torch.zeros(self.w.shape[0],1)
        w_K = -torch.sum(self.w,dim=1)[:,None]
        w = torch.cat([w_0,self.w,w_K],dim=1)
        return w @ self.t_tensor+(1-self.t_space)*self.c_0+self.t_space*self.c_1

    def curve_grad(self):
        t_tensor_grad = self.t_tensor[:-1,:]
        w_K = -torch.sum(self.w,dim=1)[:,None]
        degrees = torch.arange(1,self.degree)
        w = torch.cat([degrees*self.w,self.degree*w_K],dim=1)
        return  w @ t_tensor_grad+self.c_1-self.c_0
    
    def norm(self,x,y):
        return x.T @ self.G(y) @ x
    
    def forward(self):
        c = self.c()
        c_dot = self.curve_grad()
        norm2 = 0
        for i in range(c_dot.shape[1]):
            norm2 += c_dot[:,i] @ self.G(c[:,i]) @ c_dot[:,i]
        return norm2 / c_dot.shape[1]
    
def optimize(curve,n_steps=20):
   # L-BFGS
    def closure():
        objective = curve()
        objective.backward()
        return objective
    optimizer = optim.LBFGS(curve.parameters(),
                        history_size=10, 
                        max_iter=4, 
                        line_search_fn="strong_wolfe")
    #optimizer = optim.Adam(curve.parameters(), lr=0.01)  
    history_lbfgs = []
    for i in range(n_steps):
        history_lbfgs.append(curve().item())
        optimizer.zero_grad()
        
        optimizer.step(closure)

    plt.figure()
    plt.plot(history_lbfgs, label=f'{curve.__class__.__name__} Optimization History')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Optimization History')
    plt.legend()
    plt.savefig(f'{curve.__class__.__name__}_optimization_history.pdf')
    plt.show()
    
def plot_2D(curve, title, data):
    c_values_before = curve.initial_c
    c_values_after = curve.c().detach().clone().numpy()
    plt.figure()
    plt.plot(c_values_before[0], c_values_before[1], label='Before optimization')
    plt.plot(c_values_after[0], c_values_after[1], label='After optimization')
    plt.scatter([c_values_after[0, 0], c_values_after[0, -1]], [c_values_after[1, 0], c_values_after[1, -1]], color='blue', marker='x', label='Start/End Points')
    plt.scatter(data[:, 0], data[:, 1], color='green', marker='o', label='Data Points')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.title(f'{title}')
    plt.legend()
    plt.savefig(f'2d_{title}.pdf')
    plt.show()

if __name__ == '__main__':
    # Define the curve

    data = np.load('toybanana.npy')
    data = torch.tensor(data, dtype=torch.float32)

    # def G(x, epsilon=1e-4, sigma=0.1):
    #     N = data.shape[0]
    #     p_x = 0
    #     for n in range(N):
    #         p_x += td.MultivariateNormal(data[n], sigma**2 * torch.eye(2)).log_prob(x).exp()
    #     p_x /= N
    #     return 1/(p_x + epsilon)*torch.eye(2)

    def G(x, epsilon=1e-4, sigma=0.1):
        N = data.shape[0]
        cov = sigma**2 * torch.eye(2, device=data.device)  # Precompute covariance matrix
        mvn = td.MultivariateNormal(data, cov)  # Create a batched distribution
        log_probs = mvn.log_prob(x)  # Compute log probabilities in parallel
        p_x = log_probs.exp().mean()  # Compute the mean probability
        return (1 / (p_x + epsilon)) * torch.eye(2, device=data.device)  # Return scaled identity

    c_0 = data[0].unsqueeze(1)
    c_1 = data[10].unsqueeze(1)
    
    n_intervals = 10
    curve_pl = PiecewiseLinearCurve(c_0, c_1, G, n_intervals)
    curve_poly = PolynomialCurve(c_0, c_1, G, 3, n_intervals=30)

    optimize(curve_pl, n_steps=50)
    optimize(curve_poly, n_steps=50)

    print([curve_pl.norm(curve_velocity, curve_position).item() for curve_velocity, curve_position in zip(curve_pl.curve_grad().T, curve_pl.c().T)])
    print([curve_poly.norm(curve_velocity, curve_position).item() for curve_velocity, curve_position in zip(curve_poly.curve_grad().T, curve_poly.c().T)])

    plot_2D(curve_pl, 'Piecewise Linear Curve', data)
    plot_2D(curve_poly, 'Polynomial Curve', data)

    plt.figure()
    plt.plot(curve_pl.c().detach().clone().numpy()[0], curve_pl.c().detach().clone().numpy()[1], label='Piecewise Linear Curve')
    plt.plot(curve_poly.c().detach().clone().numpy()[0], curve_poly.c().detach().clone().numpy()[1], label='Polynomial Curve')
    plt.scatter([curve_pl.c().detach().clone().numpy()[0, 0], curve_pl.c().detach().clone().numpy()[0, -1]], [curve_pl.c().detach().clone().numpy()[1, 0], curve_pl.c().detach().clone().numpy()[1, -1]], color='red', marker='x', label='PL Start/End Points')
    plt.scatter([curve_poly.c().detach().clone().numpy()[0, 0], curve_poly.c().detach().clone().numpy()[0, -1]], [curve_poly.c().detach().clone().numpy()[1, 0], curve_poly.c().detach().clone().numpy()[1, -1]], color='blue', marker='x', label='Poly Start/End Points')
    plt.scatter(data[:, 0], data[:, 1], color='green', marker='o', label='Data Points')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.title('Comparison of Curves After Optimization')
    plt.legend()
    plt.savefig('2d_Comparison_of_Curves.pdf')
    plt.show()
