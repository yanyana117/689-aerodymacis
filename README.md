pre.py file:
Write a 2D Euler Solver to calculate flow past an airfoil. By using a triangular mesh file.

Test Case:

NACA0012 at Mach 0.80 and AOA 1.25 degree (Transonic).

NACA0012 at Mach 0.50 and AOA 3.00 degree (Subsonic, 𝑪𝑫=𝟎).

NACA0012 at Mach 0.85 and AOA 0.00 degree (Subsonic, 𝑪𝑳=𝟎).

thought.py file: Simple analysis of an O-mesh grid

---------------------------------------------------------------------------------------------------
Midterm project:

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
