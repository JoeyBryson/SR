import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

         
matplotlib.use("Qt5Agg") 

#Uses Fokker-Planck Equation to model evolution of pdf then visualizes, takes a few seconds to compute before showing results
#Uses some notation from https://pure.coventry.ac.uk/ws/portalfiles/portal/28741000/Binder3.pdf

#simulation parameters
dx = 0.05  # descretization parameters smaller = more accurate but longer compute time      
dt = 0.001  # ^^
Omega = 0.05
D = 0.11# set to optimal value 1/(4*np.log(np.sqrt(2)/(np.pi*Omega))) by default
A = 0.13      
speed_up = 50# the rate at which the simulation is visualized compared to normal time 
fps = 30 #frames per second

x_min, x_max = -2.5, 2.5  
t_max = 40 #simulation length

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

    p = np.ones(Nx)/((x_max-x_min)/dx)


    diff_coeff = D * dt / dx**2
    
    drift_coeff = dt / (2 * dx)
    p_arr = []

    for i in range(Nt):
        p_arr.append(p)
        p = evolution_step(p, x, t[i], A, Omega, drift_coeff, diff_coeff)
    
    p_arr = np.array(p_arr)
    return x, t, p_arr


def large_number_of_sims(D_arr):
    
    p_rhs_max_arr = []
    
    for i in tqdm(range(len(D_arr)), desc="Processing"):
        x, t, p_arr = simulate(x_min, x_max, t_max, dx, A, Omega, D_arr[i], dt)
        p_rhs = np.sum(p_arr[:, p_arr.shape[1] // 2:], axis=1)*dx
        p_rhs_max = np.max(p_rhs)
        p_rhs_max_arr.append(p_rhs_max)
    
    p_rhs_max_arr = np.array(p_rhs_max_arr)

    return p_rhs_max_arr

D_arr = np.linspace(0.015,0.5, 200)

p_rhs_max_arr = large_number_of_sims(D_arr)

plt.xlabel("D")
plt.ylabel(rf"$max(P(x>0))$")
plt.plot(D_arr, p_rhs_max_arr, color='black', linewidth=1)
plt.show()


