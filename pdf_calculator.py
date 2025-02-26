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
dt = 0.005  # ^^
Omega = 0.05
D = 0.2# set to optimal value 1/(4*np.log(np.sqrt(2)/(np.pi*Omega))) by default
A = 0.13      
speed_up = 50# the rate at which the simulation is visualized compared to normal time 
fps = 30 #frames per second

x_min, x_max = -2.5, 2.5  
t_max = 300 #simulation length

def potential(t, x):
    return -(x**2)/2 + (x**4)/4 - np.outer(A*np.sin(-Omega*t),x)

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
    p[int(3*Nx/4)] = 0.5 / dx  
    #p[int(Nx/4)] = 0.5 / dx  

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
#get potential function
V = potential(t, x)

fig, ax = plt.subplots()
ax2 = ax.twinx()
line1, = ax.plot([], [], lw=2, label = "Probability density function $p(x, t)$")
line2, = ax2.plot([], [], lw=2, label=r"Potential Function $V(x,t) = -\frac{x^2}{2} + \frac{x^4}{4} - A \sin(\Omega t) x$", color="orange")
ax2.axvline(0, color='black', lw=1)  
ax2.axhline(0, color='black', lw=1)  

def init():
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-3, 3)  
    ax.set_xlabel("x")
    ax.set_ylabel("p(x, t)")
    eq = (r"$\frac{\partial}{\partial t} p(x,t) = $"
         r"$- \frac{\partial}{\partial x} \left( (x - x^3 + A \sin(\omega t)) p(x,t) \right) $"
         r"$+ D \frac{\partial^2}{\partial x^2} p(x,t)$")
    ax.set_title("Evolution of Probability Density Function $p(x,t)$ with\n"
            +eq+"\n"
            rf"$\Omega = {Omega:.2g}, \ A = {A:.2g}, \ D = {D:.2g}$")

    

    ax2.set_ylabel("V(x, t)")
    ax2.set_ylim(-1, 1)  
    


    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="upper left")
    line1.set_data(x, p_arr[0, :])
    line2.set_data(x, V[0 :])

    return line1, line2


def update(frame):
    line1.set_ydata(p_arr[frame, :])
    line2.set_ydata(V[frame, :])
    time_text.set_text(f"t = {frame * dt:.3g}")
    return line1, line2, time_text

plt.tight_layout()
time_text = ax.text(0.85, 0.05, "", transform=ax.transAxes, animated = True)
fig.subplots_adjust(top=0.85)  
fig.set_size_inches(6, 6)
ani = animation.FuncAnimation(fig, update, frames=range(0, len(t), np.round(speed_up/(dt*fps)).astype(int)), init_func=init, blit=True, interval=1000/fps)

plt.show()