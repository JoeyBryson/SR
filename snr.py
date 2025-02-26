import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import scipy
from statsmodels.tsa.stattools import acf

matplotlib.use("Qt5Agg") 

#Uses Fokker-Planck Equation to model evolution of pdf then visualizes, takes a few seconds to compute before showing results
#Uses some notation from https://pure.coventry.ac.uk/ws/portalfiles/portal/28741000/Binder3.pdf

#simulation parameters
dx = 0.5  # descretization parameters smaller = more accurate but longer compute time      
dt = 0.1  # ^^
Omega = 0.05
D = 0.11# set to optimal value 1/(4*np.log(np.sqrt(2)/(np.pi*Omega))) by default
A = 0.13      
speed_up = 50# the rate at which the simulation is visualized compared to normal time 
fps = 30 #frames per second

x_min, x_max = -2.5, 2.5  
t_max = 3000 #simulation length

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

    for i in range(Nt):
        p_arr.append(p)
        p = evolution_step(p, x, t[i], A, Omega, drift_coeff, diff_coeff)
    
    p_arr = np.array(p_arr)
    return x, t, p_arr

#run simulation and save data
x, t, p_arr = simulate(x_min, x_max, t_max, dx, A, Omega, D, dt)

correlation = []

for i in range(len(x)):
    mean = np.mean(p_arr[:,i])
    correlation.append(np.correlate(p_arr[:,i]-mean, p_arr[:,i]-mean, mode='full'))

plt.plot(p_arr[:,int(1.5/dx)])
plt.plot(correlation[int(1.5/dx)])
plt.plot(correlation[int(3.1/dx)])
plt.show()

psd = np.fft.fft([x, correlation[int(1.5/dx)]])**2
freqs = np.fft.fftfreq([x, len(correlation[int(1.5/dx)])])

plt.plot(freqs[:len(freqs)//2], psd[:len(psd)//2])
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.title("Power Spectral Density")
plt.show()

