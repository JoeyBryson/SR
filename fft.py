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
dt = 0.01  # ^^
Omega = 0.05
D = 0.11# set to optimal value 1/(4*np.log(np.sqrt(2)/(np.pi*Omega))) by default
A = 0.13      
speed_up = 50# the rate at which the simulation is visualized compared to normal time 
fps = 30 #frames per second

x_min, x_max = -2.5, 2.5  
t_max = 1000 #simulation length

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
        print(Omega)
        x, t, p_arr = simulate(x_min, x_max, t_max, dx, A, Omega, D_arr[i], dt)
        p_fft = np.fft.fft2(p_arr)

        p_fft_shifted = np.fft.fftshift(p_fft)
        magnitude_spectrum = np.abs(p_fft_shifted)

        
        freq_t = np.fft.fftshift(np.fft.fftfreq(int(t_max/dt), d=dt))  # Temporal frequency axis (ω)
        freq_x = np.fft.fftshift(np.fft.fftfreq(int((x_max-x_min)/dx), d=dx))  # Spatial frequency axis (kx)
        FREQ_X, FREQ_T = np.meshgrid(freq_x, freq_t)
        plt.figure(figsize=(10, 5))
        plt.pcolormesh(FREQ_X, FREQ_T, np.log1p(magnitude_spectrum), cmap='inferno', shading='auto')
        print(np.shape(magnitude_spectrum))
        plt.colorbar(label="Log Magnitude")
        plt.title("2D FFT Magnitude Spectrum")
        plt.show()
        intensity_vs_freq = np.sum(magnitude_spectrum, axis=1)
        plt.plot(freq_t, intensity_vs_freq)
        plt.show()
        
    
    p_rhs_max_arr = np.array(p_rhs_max_arr)

    return p_rhs_max_arr

D_arr = np.linspace(0.1,0.2, 2)

p_rhs_max_arr = large_number_of_sims(D_arr)


plt.plot(D_arr, p_rhs_max_arr)
plt.show()


