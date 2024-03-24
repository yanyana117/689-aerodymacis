# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:39:36 2023

@author: Xingran Huang
Midterm proejct

- First Problem: Write a program to solve the equation: ∂u/∂t + a ∂u/∂x = 0 (One-dimensional linear advection equation)对流方程
Try both the upwind scheme and the central difference scheme with implicit time stepping. (隐式时间步进)
Propagate(传播) a step and a pulse for different CFL values (1 and less than 1).
Test the upwind scheme with CFL 1.1 on approximately 32 mesh cells.

Hits:
    a: convection speed,对流速度 , if alpha >0 , the wave propagates in the positive direction of x-axis
    u: dependent variable,依赖变量,代表通过对流过程被对流或传输的物理量
    x: spatial coordinate,空间坐标,代表 posotion in which the advection is taking place.
    t: time
    CFL: Courant–Friedrichs–Lewy
    

- Second Problem: Write a program to solve the Burgers' equation: ∂u/∂t + ∂/∂x (u^2/2) = 0.
Use the upwind scheme with different initial conditions and provide a graphical output.


- Final Problem: Write a program to solve 2D subsonic and transonic flow past an airfoil, applying it to the following test cases:
    NACA0012
        Mach = 0.50 at 3° angle of attack
        Mach = 0.80 at 0° angle of attack
        Mach = 0.80 at 1.25° angle of attack.
"""
import numpy as np
import matplotlib.pyplot as plt

# Problem 1:
# Initial conditions
L = 30.0  
nx = 96  
dx = L / (nx-1)
nt = 100
dt = 0.02
a = 1.0

def simulate(CFL):
    # step function(模拟初始条件)
    u_step = np.ones(nx) 
    u_step[int(0.25*L/dx):int(0.5*L/dx)] = 2
    # u_step[int(2*L/dx)] = 2
    
    # pulse function
    u_pulse = np.ones(nx)
    u_pulse[int(0.45*L/dx):int(0.55*L/dx)] = 2
    # u_pulse[int(0.5*L/dx)] = 2


    def implicit_upwind(u, nt):     # 隐式上风方案
        u_history = [u.copy()]

        for n in range(nt):
            A = np.eye(nx)  # 单位矩阵
            b = u.copy()

            for i in range(1, nx):      # 隐式上风差分方案
                A[i, i] = 1 + CFL
                A[i, i-1] = -CFL 

            u = np.linalg.solve(A, b)   # 线性代数库来解决线性系统Ax = b，得到下一个时间步的解
            u_history.append(u.copy())

        return u_history


    def implicit_central_diff(u, nt):       # 隐式中心差分方案
        u_history = [u.copy()]

        for n in range(nt):
            A = np.eye(nx)  # 单位矩阵
            b = u.copy()

            for i in range(1, nx-1):
                A[i, i-1] = -0.5 * CFL
                A[i, i+1] = 0.5 * CFL

            u = np.linalg.solve(A, b)
            u_history.append(u.copy())

        return u_history

    u_step_upwind = implicit_upwind(u_step.copy(), nt)
    u_pulse_upwind = implicit_upwind(u_pulse.copy(), nt)
    u_step_central = implicit_central_diff(u_step.copy(), nt)
    u_pulse_central = implicit_central_diff(u_pulse.copy(), nt)

    return u_step_upwind, u_pulse_upwind, u_step_central, u_pulse_central


# Plot
def plot_results_subplot(u_step_upwind, u_pulse_upwind, u_step_central, u_pulse_central, CFL, fig_num):
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plot_results(u_step_upwind, f"Implicit Upwind Scheme - Step Function (CFL={CFL})")

    plt.subplot(2, 2, 2)
    plot_results(u_pulse_upwind, f"Implicit Upwind Scheme - Pulse Function (CFL={CFL})")

    plt.subplot(2, 2, 3)
    plot_results(u_step_central, f"Implicit Central Difference Scheme - Step Function (CFL={CFL})")

    plt.subplot(2, 2, 4)
    plot_results(u_pulse_central, f"Implicit Central Difference Scheme - Pulse Function (CFL={CFL})")

    plt.tight_layout()
    plt.savefig(f"Simulation_Results_{fig_num}.png")
    plt.show()


def plot_results(u, title):
    for i, ui in enumerate(u):
        if i % 10 == 0:  # 仅绘制部分时间步骤
            plt.plot(np.linspace(0, L, nx), ui, label=f"t = {i*dt:.2f}")
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('u')
    plt.title(title)


# Define CFL values:
CFL = 1.1
u_step_upwind, u_pulse_upwind, u_step_central, u_pulse_central = simulate(CFL)
plot_results_subplot(u_step_upwind, u_pulse_upwind, u_step_central, u_pulse_central, CFL, fig_num=1)


CFL = 0.89
u_step_upwind, u_pulse_upwind, u_step_central, u_pulse_central = simulate(CFL)
plot_results_subplot(u_step_upwind, u_pulse_upwind, u_step_central, u_pulse_central, CFL, fig_num=2)



"""
- Second Problem: Write a program to solve the Burgers' equation: ∂u/∂t + ∂/∂x (u^2/2) = 0.
Use the upwind scheme with different initial conditions and provide a graphical output.
"""


class FVGrid:   # Define Finite Volume Grid
                # 定义汽车，def1有颜色、品牌、型号、速度等属性，还有def2启动、def3停止、def4加速、def5减速等功能

    def __init__(self, nx, ng, bc="outflow", xmin=0.0, xmax=1.0):        
        
        """
        __init__: initializer constructor 为新对象设置初始状态或属性
        nx: Number of grid nodes 设置网格节点数
        ng: Ghost cells,帮助处理边界处的流体
        bc: Boundary conditions
        outlow: 在bc条件中，假设流体只能流出不能流入
        xmin & xmax : range of x-axis
        """
      
        self.initialize_grid(nx, ng, xmin, xmax)    # 初始化一个网格
        self.initialize_data(bc)                    # 初始化数据
        
        
    def initialize_grid(self, nx, ng, xmin, xmax):
        
        """
        Initialize the grid parameters.
        self.ilow（Index Lower）:代表网格中实际计算区域的最低索引
        self.ihigh（Index Higher）:代表网格中实际计算区域的最高索引
        Index:数组或数据结构中元素的位置编号
        self.dx:这行代码计算网格点之间的间距
        self.x: 这行代码计算每个网格点的中心位置 
        self.xl // self.xr:计算每个网格单元的左/右边界位置
        """

        self.nx, self.ng, self.xmin, self.xmax = nx, ng, xmin, xmax   
       
        # lowest/highest index of the actual computational region in the grid
        self.ilow, self.ihigh = ng, ng + nx - 1 #       
        
        # spacing between grid points
        self.dx = (xmax - xmin) / nx 
        
        # center position of each grid point
        self.x = xmin + (np.arange(2 * ng + nx) - ng + 1/2) * self.dx
        
        # left/right boundary position of each grid cell
        indices = np.arange(nx + 2 * ng) - ng
        self.xl, self.xr = xmin + indices * self.dx, xmin + (indices + 1) * self.dx

        
    def initialize_data(self, bc):  # 用于初始化数据和边界条件   
        """Initialize the data and boundary conditions."""       
        self.bc = bc       
        self.u = self.scratch_array()        
        self.uinit = self.scratch_array()


    def scratch_array(self):       
        """Return a zero array for grid operations.""" 
        # 通常用于表示整个数组的大小，包括实际计算区域中的网格点和两侧的幽灵单元                   
        return np.zeros(( 2 * self.ng + self.nx), dtype=np.float64)  # 提供非常高的精度
                   
    
    def fill_BCs(self, atmp):
        """
        Fill boundary conditions
        atmp: Array temporary 临时数组
        填充边界条件: 检查 边界条件类型 是否为周期性 
        """
        if self.bc == "periodic": # 周期性 
            atmp[   :   self.ng ] = atmp[ -2*self.ng  :  -self.ng ]
            atmp[ -self.ng  :   ] = atmp[ self.ng : 2*self.ng ]
            
        elif self.bc == "outflow":           
            atmp[     : self.ng ] = atmp[ self.ng ]      
            atmp[ -self.ng :    ] = atmp[ -self.ng - 1 ]
        else:
            raise ValueError("Invalid BC")


    def norm(self, e):
        """Calculate the norm of a vector"""
        return np.sqrt(self.dx * np.sum(e[self.ilow : 1 + self.ihigh]**2))


def flux_update(gr, u):
    """compute -div{F} for linear advection"""

    del_l = gr.scratch_array()
    del_l[gr.ilow-1:gr.ihigh+2] = u[gr.ilow-1:gr.ihigh+2] - u[gr.ilow-2:gr.ihigh+1]

    del_r = gr.scratch_array()
    del_r[gr.ilow-1:gr.ihigh+2] = u[gr.ilow:gr.ihigh+3] - u[gr.ilow-1:gr.ihigh+2]

    del_1 = np.where(np.fabs(del_l) < np.fabs(del_r), del_l, del_r)
    del_u = np.where(del_l*del_r > 0.0, del_1, 0.0)

    u_left = gr.scratch_array()
    u_right = gr.scratch_array()
    
    u_right[gr.ilow:gr.ihigh+2] = u[gr.ilow:gr.ihigh+2] - 0.5 * del_u[gr.ilow:gr.ihigh+2]
 
    u_left[gr.ilow:gr.ihigh+2] = u[gr.ilow-1:gr.ihigh+1] + 0.5 * del_u[gr.ilow-1:gr.ihigh+1]  

    # Speed of the wave
    S = 0.5 * (u_left + u_right)
    u_shock = np.where(S > 0.0, u_left, u_right)
    u_shock = np.where(S == 0.0, 0.0, u_shock)  

    # Rarefaction solution
    u_rightare = np.where(u_right <= 0.0, u_right, 0.0)
    u_rightare = np.where(u_left >= 0.0, u_left, u_rightare)   

    us = np.where(u_left > u_right, u_shock, u_rightare)
    
    # Flux difference
    flux_diff = gr.scratch_array()
    flux_diff[gr.ilow:gr.ihigh+1] = (0.5 * us[gr.ilow:gr.ihigh+1]**2 - 0.5 * us[gr.ilow+1:gr.ihigh+2]**2) / gr.dx

    return flux_diff


def burgers_mol(nx, C, tmax, init_cond=None):
    """Solve Burgers' equation using method of lines."""
    
    grid = FVGrid(nx, ng=2)
    
    init_cond(grid)
    
    grid.uinit[:] = grid.u[:]
    
    t = 0.0
    
    while t < tmax:
        
        dt = C * grid.dx / np.abs(grid.u).max()
        
        dt = min(dt, tmax - t)
        
        grid.fill_BCs(grid.u)
        
        k1 = flux_update(grid, grid.u)

        utmp = grid.scratch_array()
        
        utmp[:] = grid.u[:] + 1/2 * dt * k1
        
        grid.fill_BCs(utmp)
        
        k2 = flux_update(grid, utmp)
        
        grid.u[:] += dt * k2
        
        t += dt

    return grid


def jump(g):
    """Initial condition: rarefaction."""
    g.u[:] = 1.0
    g.u[g.x > 0.5] = 2.0



def sin_wave(g):
    """Initial condition: sine wave."""
    g.u[:] = np.sin(2 * np.pi * g.x)


def odd(g):
    """Initial condition: complex sine wave."""
    g.u[:] = np.sin(8 * np.pi * g.x)


def half_sharp(g):
    """Initial condition: half sharp wave."""
    for i in range(len(g.x)):
        if g.x[i] < 0.5:
            g.u[i] = 1 - 2 * g.x[i]
        else:
            g.u[i] = 0


def plot(self):
    fig, ax = plt.subplots()
    ax.plot(self.x, self.uinit, label="initial conditions")
    ax.plot(self.x, self.u)
    ax.legend()
    return fig


if __name__ == "__main__":
    nx, C, tmax = 128, 0.5, 0.2
    
    fig, axarr = plt.subplots(2, 2, figsize=(10, 6))

    # 使用 rarefaction 初始条件
    g_rare = burgers_mol(nx, C, tmax, init_cond=jump)
    axarr[0, 0].plot(g_rare.x, g_rare.uinit, label="initial conditions")
    axarr[0, 0].plot(g_rare.x, g_rare.u,linestyle='--')
    axarr[0, 0].set_title("Rarefaction Initial Condition")
    #axarr[0, 0].set_xlim(-10, 10)  # 设置x轴范围为-50到50
    axarr[0, 0].legend()

    # 使用 sine wave 初始条件
    g_sin = burgers_mol(nx, C, tmax, init_cond=sin_wave)
    axarr[0, 1].plot(g_sin.x, g_sin.uinit, label="initial conditions")
    axarr[0, 1].plot(g_sin.x, g_sin.u,linestyle='--')
    axarr[0, 1].set_title("Sine Wave Initial Condition")
    #axarr[0, 1].set_xlim(-10, 10)
    axarr[0, 1].legend()

    # 使用 complex sine wave 初始条件
    g_complex_sin = burgers_mol(nx, C, tmax, init_cond=odd)
    axarr[1, 0].plot(g_complex_sin.x, g_complex_sin.uinit, label="initial conditions",linestyle='--')
    axarr[1, 0].plot(g_complex_sin.x, g_complex_sin.u)
    axarr[1, 0].set_title("Odd even Initial Condition")
   # axarr[1, 0].set_xlim(-10, 10)
    axarr[1, 0].legend()

    # 使用 half ramp 初始条件
    g_half_sharp = burgers_mol(nx, C, tmax, init_cond=half_sharp)
    axarr[1, 1].plot(g_half_sharp.x, g_half_sharp.uinit, label="initial conditions")
    axarr[1, 1].plot(g_half_sharp.x, g_half_sharp.u,linestyle='--')
    axarr[1, 1].set_title("Half ramp Initial Condition")
   # axarr[1, 1].set_xlim(-10, 10)
    axarr[1, 1].legend()

    # 调整子图的间距

    plt.tight_layout()
    plt.show()

    