#run simulation and save data

import numpy as np
import scipy
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from tqdm import tqdm


#simulation parameters
dx = 0.05  # descretization parameters smaller = more accurate but longer compute time      
dt = 0.01  # ^^
Omega = 0.05
D = 0.11# set to optimal value 1/(4*np.log(np.sqrt(2)/(np.pi*Omega))) by default
A = 0.13      


x_min, x_max = -2.5, 2.5  
t_max = 300 #simulation length


def potential(t, x):
    return -(x**2)/2 + (x**4)/4 - np.outer(A*np.sin(Omega*t),x)

def evolution_step(p, x, t, A, Omega, drift_coeff, diff_coeff):

    p_new = np.copy(p)
    drift = (x - x**3 + A*np.sin(Omega * t)) * p

    drift_term = -drift_coeff * (np.roll(drift, -1) - np.roll(drift, 1))
    diffusion_term = diff_coeff * (np.roll(p, -1) - 2 * p + np.roll(p, 1))

    p_new += drift_term + diffusion_term
    p_new = np.maximum(p_new, 0)  
    p_new /= np.sum(p_new)*dx

    return p_new

def simulate(x_min, x_max, t_max, dx, A, Omega, D, dt):
    
    x = np.arange(x_min, x_max, dx)
    t = np.arange(0, t_max, dt)
    Nx = len(x)
    Nt = len(t) 
    

    p = np.zeros(Nx)
    p[int(3*Nx/4)] = 1.0 / dx  

    diff_coeff = D * dt / dx**2
    
    drift_coeff = dt / (2 * dx)
    p_arr = []

    for i in tqdm(range(Nt), desc="Processing"):
        p_arr.append(p)
        p = evolution_step(p, x, t[i], A, Omega, drift_coeff, diff_coeff)
    
    p_arr = np.array(p_arr)
    return x, t, p_arr

x, t, p_arr = simulate(x_min, x_max, t_max, dx, A, Omega, D, dt)
p_arr_slice = p_arr[-1] 

def two_gaussians(x, A1, mu1, sigma1, A2, mu2, sigma2):
    g1 = A1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))
    g2 = A2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))
    return g1 + g2

initial_guess = [ 1.30627554, 1.00201903, 0.23655985, 0.25047879, -0.87585937, 0.3040868 ]

params, covariance = curve_fit(two_gaussians, x, p_arr_slice, p0=initial_guess)

A1_fit, mu1_fit, sigma1_fit, A2_fit, mu2_fit, sigma2_fit = params

y_fit = two_gaussians(x, *params)
print(params)
plt.scatter(x, p_arr_slice, label="Data", color='gray', alpha=0.5)
plt.plot(x, y_fit, label="Fitted Sum of Gaussians", color='red')
plt.legend()
plt.show()
plt.plot(x, p_arr_slice-y_fit, label="Fitted Sum of Gaussians", color='red')
plt.show()
    