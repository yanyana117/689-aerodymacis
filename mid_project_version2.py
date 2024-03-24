# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 05:59:03 2023
- Second Problem: Write a program to solve the Burgers' equation: ∂u/∂t + ∂/∂x (u^2/2) = 0.
Use the upwind scheme with different initial conditions and provide a graphical output.

odd-even model (odd-even De-coupling)
Koren limiter,. Van Leer limiter 齿轮

a rarefaction wave
upwind sceme reverse shocl wave

Rarefaction Wave: 稀疏波（一维流动中，因为速度差异导致的流体逐渐分离的区域,通常出现在一维流动中，当流体从高压区域突然进入低压区域时。）
 shows N-wave,shock wave
 https://www.scirp.org/journal/paperinformation.aspx?paperid=110205
图片：
"Numerical" 意味着这条线代表的是使用数值方法（一阶迎风格式）计算得到的 Burgers 方程的解。
"Analytical" 表明这条线代表的是通过解析方法（即精确的数学公式）得到的 Burgers 方程的解。

@author: yanyan
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the initial condition with a sine function
def initial_condition_sin_sin(x):
    return np.sin(np.pi * x)

# Define the Lax-Friedrichs numerical scheme
def lax_friedrichs(u, dx, dt):
    u_next = np.zeros_like(u)
    for i in range(1, len(u) - 1):
        f_plus = 0.5 * u[i+1]**2
        f_minus = 0.5 * u[i-1]**2
        u_next[i] = 0.5 * (u[i+1] + u[i-1]) - dt/(2*dx) * (f_plus - f_minus)
    return u_next

# Function to initialize the half ramp
def initialize_half_ramp(x, L):
    """
    Initialize the half ramp condition for Burgers' equation.
    Args:
    x: Array of spatial coordinates.
    L: Length of the domain where the ramp goes to zero.
    Returns:
    u: Array of velocity values initialized with the half ramp condition.
    """
    u = np.maximum(1 - x/L, np.zeros_like(x))
    return u


# Function to initialize the sine wave
def initialize_sine_wave(x, L):
    """
    Initialize a sine wave condition for Burgers' equation.
    Args:
    x: Array of spatial coordinates.
    L: Length of the domain over which the sine wave is defined.
    Returns:
    u: Array of velocity values initialized with the sine wave condition.
    """
    u = np.sin(2 * np.pi * x / L)
    return u


# Function to compute the upwind scheme step
def upwind_step(u, dx, dt):
    """
    Compute the next time step in the upwind scheme for Burgers' equation.
    Args:
    u: Array of current velocity values.
    dx: Spatial step size.
    dt: Time step size.
    Returns:
    u_next: Array of velocity values at the next time step.
    """
    u_next = u.copy()
    for i in range(1, len(u)):
        u_next[i] = u[i] - dt / dx * (u[i]**2 / 2 - u[i-1]**2 / 2)
    return u_next


########################################################################################
# Set computational parameters
x_min, x_max = -1.0, 1.0
t_final = 0.5
Nx = 400
Nt = 200
dx = (x_max - x_min) / (Nx - 1)
dt = t_final / Nt

CFL = 0.5
dt = CFL * dx

# Initialize the computational grid and solution
x = np.linspace(x_min, x_max, Nx)
u = initial_condition_sin_sin(x)

# Prepare the plot
plt.figure(figsize=(12, 8))
plt.plot(x, u, 'k-', label='Initial Condition')

# Select time steps at which to plot the solution
# plot_times = [0, t_final * 0.25, t_final * 0.5, t_final * 0.75, t_final]
# plot_indices = [int(t / dt) for t in plot_times]

time_steps = np.arange(0, t_final + dt, 0.05)
plot_indices = (time_steps / dt).astype(int)

# Solve the Burgers' equation and plot at selected times
for n in range(1, Nt+1):
    u_n = u.copy()
    u = lax_friedrichs(u_n, dx, dt)
    u[0] = u[-2]  # Apply periodic boundary conditions
    u[-1] = u[1]
    
    if n in plot_indices:
        plt.plot(x, u,'--', label=f'Numerical t={n*dt:.2f}')

# Finalize the plot
plt.title("Burgers' Equation with Sine Wave Initial Condition")
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.ylim(-1.1, 1.1)
plt.xlim(x_min, x_max)
plt.grid(True)
plt.show()

########################################################################################

# Define a sharp sawtooth initial condition with 3-4 teeth across the domain
def initial_condition_sharp_sawtooth(x):
    # Scale the pattern to have 3-4 teeth over the interval [-1, 1]
    # Each tooth will be a triangle wave with a base width equal to the period
    period = (x_max - x_min) / 3.5  # Adjust the period to get 3-4 teeth
    return 2 * (2 * np.abs((x / period) % 1 - 0.5) - 1)

# Initialize the computational grid and solution with the new initial condition
u_sharp_sawtooth = initial_condition_sharp_sawtooth(x)

# Prepare the plot for the sharp sawtooth initial condition
plt.figure(figsize=(12, 8))
plt.plot(x, u_sharp_sawtooth, 'k-', label='Initial Condition')

# Solve the Burgers' equation and plot at selected times
for n in range(1, Nt+1):
    u_n = u_sharp_sawtooth.copy()
    u_sharp_sawtooth = lax_friedrichs(u_n, dx, dt)
    u_sharp_sawtooth[0] = u_sharp_sawtooth[-2]  # Apply periodic boundary conditions
    u_sharp_sawtooth[-1] = u_sharp_sawtooth[1]
    
    if n % (Nt // 10) == 0 or n == 1 in plot_indices:
        plt.plot(x, u_sharp_sawtooth,'--', label=f'Numerical t={n*dt:.2f}')

# Finalize the plot for the sharp sawtooth initial condition
plt.title("Burgers' Equation with Sharp Sawtooth Initial Condition")
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.ylim(-2, 2)
plt.xlim(x_min, x_max)
plt.grid(True)
plt.show()

########################################################################################

# Parameters
L = 1.8            # Length of the domain
nx = 90            # Number of spatial steps
dx = L / (nx - 1)  # Spatial step size
dt = 0.01          # Time step size
nt = 110           # Number of time steps to reach t=1.0

# Define the grid
x = np.linspace(0, L, nx)
x_sine = np.linspace(0, L, nx)

# Initial conditions: half ramp
u = initialize_half_ramp(x, 1)  # Assuming the ramp goes to zero at x=1
u_sine = initialize_sine_wave(x_sine, L)

# Initialize the plot
plt.figure(figsize=(10, 5))
plt.title('Problem 2 half ramp with upwind scheme')
plt.xlabel('X domain')
plt.ylabel('u')

# Time-stepping loop using the upwind scheme
for n in range(nt):
    u = upwind_step(u, dx, dt)
    
    # Plot at specific intervals to observe evolution
    if n % 5 == 0:  # Plot every 5 steps
        plt.plot(x, u, label=f"t={n*dt:.2f}")

# Finalize plot
plt.legend()
plt.show()

########################################################################################

def initial_condition_jump(x, t=0):
    return np.where(x < 0, -1, 1)

x_min, x_max = -1.0, 1.0
t_final = 1.0
Nx = 400  # Number of spatial points
Nt = 400  # Number of time points
x = np.linspace(x_min, x_max, Nx)
dx = (x_max - x_min) / (Nx - 1)
CFL = 0.8
dt = CFL * dx  # Time step size calculated based on the CFL condition

# Prepare for plotting
plt.figure(figsize=(12, 8))

# Store analytical solutions to avoid recomputation
analytical_solutions = []

# Time points to plot, including the initial and final times
t_values = np.arange(0, t_final + dt, 0.05)  # Now includes more time steps

# Calculate and plot the analytical solutions
for t_val in t_values:
    # Calculate the analytical solution
    u_analytical = np.piecewise(x, [x < -t_val, (x >= -t_val) & (x <= t_val), x > t_val], [-1, lambda x: x/t_val, 1])
    analytical_solutions.append(u_analytical)
    # Plot the analytical solution
    plt.plot(x, u_analytical, '--', label=f'Analytical t={t_val:.2f}') if t_val > 0 else plt.plot(x, u_analytical, 'k-', label='Initial Condition')

# Calculate and plot the numerical solutions
u = initial_condition_jump(x)
u_n = np.zeros(Nx)

for n in range(1, Nt):
    u_n[:] = u[:]
    # First-order upwind scheme for solving conservation laws
    for i in range(1, Nx - 1):
        u[i] = u_n[i] - CFL * (u_n[i] ** 2 / 2 - u_n[i - 1] ** 2 / 2)
    u[0], u[-1] = initial_condition_jump(np.array([x_min, x_max]), n*dt)
    
    # Plot the numerical solution at specific time points
    if np.isclose(n*dt, t_values).any():
        plt.plot(x, u, label=f'Numerical t={n*dt:.2f}')

# Set the plot formatting
plt.title("Burgers' Equation with Rarefaction Wave at Various Times")
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.ylim(-1.1, 1.1)
plt.xlim(x_min, x_max)
plt.grid(True)

# Display the plot
plt.show()
