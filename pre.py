# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 22:50:27 2023

@author: yanyan
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import copy


rk4_coeff = np.array([1/4, 1/3, 1/2, 1])
# 定义RK4（四阶龙格-库塔方法）的系数数组。这些系数用于在时间积分过程中权衡不同阶段的影响。

# inf：来流
# k：单元中间点上的直
# R:黎曼 R_p： R_plus； R_m： minus

# 有限体积数值方法
# 来流初始化,设置来流条件，马赫数自己设置
class Case:
    def __init__(self, Ma_inf, alpha):
        # Setting up inflow conditions and basic parameters for numerical simulation
        
        self.Ma_inf = Ma_inf                # Set Mach number of the inflow 设置来流的马赫数
        self.alpha = alpha * np.pi / 180    # AOA 攻角，将度转换为弧度
        self.T_inf = 1.0                    # inflow temperature 来流温度
        self.C_inf = 1.0                    # speed of sound in the inflow 来流声速
        self.P_inf = 1.0                    # inflow pressure 来流压力
        self.rho_inf = 1.4                  # inflow density 来流密度
        self.R = 1/1.4                      # gas constant 气体常数R
        self.gamma = 1.4                    # specific heat ratio 比热比，气体的物理性质
        self.Cv = self.R/(self.gamma - 1.0) # specific heat at constant volume 比容
        self.calculate_conserve_inf()       # 调用方法计算来流的守恒变量
        self.k2 = 0.8                       # 2nd 2阶人工粘性系数
        self.k4 = 3e-3                      # 4阶人工粘性系数
        self.CFL = 2.0                      # 库朗数，用于稳定性条件的计算
        self.error = 1e-5                   # 收敛误差阈值,
        self.STEP = 1000                    # 模拟的迭代步数,1000~10000; 注意1459步左右就nan了 1h55min33s part 3  可以改成5000，1436 
        self.data_path = ".//data//"        # 存数据的路径，存储模拟数据的路径
        self.image_path = ".//image//"      # 存图片的路径，存储生成的图像的路径
        pass

    
    # project里的q    
    def calculate_conserve_inf(self):  # 来流条件下的守恒变量
                                       # 最后输出是一个包含四个元素的数组 w_inf：密度、动量的x分量、动量的y分量和总能量。
        
        U_inf = self.Ma_inf * self.C_inf * np.cos(self.alpha)   # 计算来流的U分量（x方向速度）
        V_inf = self.Ma_inf * self.C_inf * np.sin(self.alpha)   # 计算来流的V分量（y方向速度）
        E_inf = self.P_inf /(self.gamma - 1) / self.rho_inf + (U_inf**2 + V_inf**2) / 2   
                                                                # 计算来流的总能量   
        w_inf = np.zeros(4)                                     # 创建一个包含四个元素的零数组[0. 0. 0. 0.]，用于存储守恒变量
        w_inf[0] = self.rho_inf                                 # 第一个守恒变量是密度
        w_inf[1] = self.rho_inf * U_inf                         # 第二个守恒变量是动量的x分量
        w_inf[2] = self.rho_inf * V_inf                         # 第三个守恒变量是动量的y分量
        w_inf[3] = self.rho_inf * E_inf                         # 第四个守恒变量是总能量
        
        self.w_inf = w_inf                                      # 将计算得到的守恒变量数组存储在类的属性中
        self.w_inf_all = self.conserve_to_all(w_inf)            # 将守恒变量转换为其他形式，并存储
        pass
    
    
    def conserve_to_all(self,w): # 守恒变量转换，输入是w_inf，输出是一个包含了更多流体物理性质的数组
        rho = w[0]               # 主要定义流体物理性质和基础计算
        U = w[1] / rho    # Velocity in x-direction x方向速度
        V = w[2] / rho    # Velocity in y-directiony方向速度
        E = w[3] / rho    # Specific energy
        P = E -(U**2 + V**2) /2 * rho * (self.gamma - 1) # pressure 压力
        H = E + P / rho   # Specific enthalpy 比焓
        C = np.sqrt(self.gamma * P / rho)    #  Speed of sound 声速
        return np.array([rho, U, V, E, P, H, C])
        pass
    
    
    def calculate_flux(self, w_edge, dx, dy): # Flux calculation 通量计算  
        rho, U, V, E, P, H, C = self.conserve_to_all(w_edge)
        Q = np.zeros(4)                 # flux 通量
        Z = U*dy - V*dx                 # Auxiliary variable Z 辅助变量Z
        Q[0] = Z * rho                  # Mass flux
        Q[1] = Z * rho * U + P * dy     # Momentum flux in x-direction
        Q[2] = Z * rho * V - P * dx     # Momentum flux in y-direction
        Q[3] = Z * rho * H              # Energy flux
        return Q                        # Return flux array
        pass     
    pass
     

###############################################################################################
class Grid: # 读取网格
            # 责从一个文件中读取网格信息，并将这些信息作为其属性进行存储。
            # 这些属性包括网格的节点、边、单元及其相关的数据，如坐标、体积用于存储和处理网格数据
    def __init__(self, grid_file): # __init__：初始化方法，构造函数
        nnodes, nedges, ncells = np.loadtxt(grid_file, max_rows=1, dtype=int)
        # 从网格文件的第一行读取节点数、边数和单元数。这些值被存储在nnodes, nedges, ncells变量中。
        
        xy = np.loadtxt(grid_file, skiprows=1, max_rows=nnodes, dtype=float)
        # 从网格文件读取节点的坐标。跳过第一行，读取nnodes行，存储每个节点的x和y坐标(2列数)
        
        iedge = np.loadtxt(grid_file, skiprows=1+nnodes, max_rows=nedges, dtype=int)-1
        # iedge = np.loadtxt(grid_file, skiprows=1+nnodes, max_rows=nedges, dtype=int)
        # iedge = np.loadtxt(grid_file, skiprows=1+nnodes, max_rows=nedges, dtype=int, ndmin=2)
        # 读取边的信息。跳过前面的节点信息行，读取nedges行，存储边的信息。(4列数i, j, k, p )
        # i,j: 第k条边连接的起始点与终点节点编号
        # k,p: 单元左右两侧单元号 若该号为-1，则表明为物面(wallboundary)，若为-2，则表明为近场(farfeld boundary)


        icell = np.loadtxt(grid_file, skiprows=1+nnodes+nedges, max_rows=ncells, dtype=int) - 1
        # 读取单元的信息。跳过节点和边的信息行，读取ncells行，存储单元的信息。（3列数,表示三角单元的3个节点编号）
        # 数组中的值减1，可能是因为原始数据是基于1的索引，而Python使用基于0的索引
        
        vol = np.loadtxt(grid_file, skiprows=1+nnodes+nedges+ncells, max_rows=ncells, dtype=float)
        # 读取每个单元的体积。跳过节点、边和单元信息行，读取ncells行，存储每个单元的体积。（1列数）
        
        self.nnodes = nnodes
        self.nedges = nedges
        self.ncells = ncells    # 单元数
        self.xy = xy
        self.iedge = iedge
        self.icell = icell
        self.vol = vol
        self.calcualte_dx_dy()
        # 将读取的数据保存为类的属性，以便在类的其他方法中使用
        print("xy array structure:", self.xy.shape)
        print("xy array sample:\n", self.xy[:5])
        pass


    def plot_grid(self, image_name):
        triangle = tri.Triangulation(self.xy[:,0], self.xy[:,1], self.icell)
        plt.figure(figsize=(12, 8), facecolor="white", dpi=500) #dpi 分辨率
        plt.gca().set_aspect(1) # 设置绘图的长宽比为1
        plt.triplot(triangle, lw=0.1, color="r") # line wide
        plt.xlim(-2, 3)
        plt.ylim(-2.5, 2.5)
        plt.savefig(image_name, bbox_inches="tight")
        if image_name:  # Only save the figure if image_name is given
            plt.savefig(image_name, bbox_inches="tight")
        plt.show()
        pass
    
    
    def calcualte_dx_dy(self):
        # 创建三个数组，分别用于存储网格边的dx, dy和长度ds
        dx = np.zeros(self.nedges)
        dy = np.zeros(self.nedges)
        ds = np.zeros(self.nedges)
        for edge in range(self.nedges):
            i, j, k, p = self.iedge[edge]
            # print("j index:", j, "Value at j:", self.xy[j])
            
            xj, yj = self.xy[j]
            xi, yi = self.xy[i]
            dx[edge] = xj - xi # dx这个边的长度
            dy[edge] = yj - yi
            ds[edge] = np.sqrt(dx[edge]**2 + dy[edge]**2)
            pass
        self.dx = dx
        self.dy = dy
        self.ds = ds
        pass


##############################################################################
class Solver:
    
    def __init__(self, case, grid): # 传进case和gird,这个class里有case和grid两个变量
        self.case = case
        self.grid = grid
        pass
    
    def initialize_field(self): # 定义初始化流场
        W_cur = np.zeros([self.grid.ncells, 4])
        # W是守恒量(公式有给),cur=current；创建了一个名为W_cur的二维numpy数组
        # 如果网格中有100个单元self.grid.ncells，这个数组就会有100行和4列，每行代表一个网格单元，每列代表一个守恒变量的分量
        for j in range(self.grid.ncells):
            W_cur[j] = self.case.w_inf # j行，i列？ # 将来流条件下的守恒变量赋值给每个网格单元
            pass
        self.W_cur = W_cur
        # 这行将局部变量W_cur的值赋给类的实例变量self.W_cur。这意味着W_cur的值在方法外部也能通过类的实例访问
        self.conserve_to_all()
        pass
    
    
    def conserve_to_all(self): # 后面7个变量，标注all # 主要用这些定义来进行整个流场的求解和模拟         
        W_cur_all = np.zeros([self.grid.ncells, 7]) # rho, U, V, E, P, H, C = self.conserve_to_all(w_edge)
        for j in range(self.grid.ncells):
            W_cur_all[j] = self.case.conserve_to_all(self.W_cur[j])
            pass
        self.W_cur_all = W_cur_all
        pass
   
    
    def calculate_flux(self, rho, U, V, P, H, dx, dy): # 通量计算Q
        Q = np.zeros(4)
        Z = U*dy - V*dx
        Q[0] = Z * rho
        Q[1] = Z * rho * U + P * dy
        Q[2] = Z * rho * V - P * dx
        Q[3] = Z * rho * H
        return Q
        pass 
    
    def calculate_laplace(self):
        W_cur_laplace = np.zeros([self.grid.ncells, 4])
        for edge in range(self.grid.nedges):
            i, j, k, p = self.grid.iedge[edge]
            if p > -1:      # 如果p大于-1，则表示边界上的单元
                w_k = self.W_cur[k]  # 获取k单元的守恒变量
                w_p = self.W_cur[p]  # 获取p单元的守恒变量
                value = w_p - w_k    # 差值
                # 更新拉普拉斯数组
                W_cur_laplace[k] = W_cur_laplace[k] + value
                W_cur_laplace[p] = W_cur_laplace[p] - value
                pass
            pass
        self.W_cur_laplace = W_cur_laplace
        pass
    
    
    def each_step(self):    # flux 通量计算
        self.conserve_to_all()
        self.calculate_laplace()
        Q = np.zeros([self.grid.ncells, 4])
        D2 = np.zeros([self.grid.ncells, 4])   # 二阶人工粘性 False diffusion  
        D4 = np.zeros([self.grid.ncells, 4])   # 四阶人工粘性
        t = np.zeros(self.grid.ncells)         # 每个单元的时间
        for edge in range(self.grid.nedges):   # 边连接矩阵？ 我的是ijxy，example是ijkp
            i,j,k,p = self.grid.iedge[edge] 
            dx = self.grid.dx[edge]   # 获取边在x，y方向的长度
            dy = self.grid.dy[edge]
            ds = self.grid.ds[edge]   # ds 代表边的实际长度，从 self.grid.ds 数组中获取索引为 edge 的边的长度 
            if p == -2:
                # wall boundary 地面条件/物面边界条件
                w_k = self.W_cur[k]
                # p = self.case.conserve_to_all(w_k)
                p = self.W_cur_all[k][4]
                flux = np.array([0, p*dy, -p*dx, 0])
                Q[k] = Q[k] + flux
                pass
            
            
            elif p ==-3: # 数字要调整
                # for filed boundary 边场边界
                w_k = self.W_cur[k]             # 当前边界单元的守恒变量值
                w_k_all = self.W_cur_all[k]     # 当前边界单元的全变量值
                w_inf =  self.case.w_inf_all    # 远场（无穷远处）条件下的全变量值
                w_inf_all = self.case.w_inf_all # 远场条件下的全变量值（再次赋值，可能有冗余）
                
                # 提取单元的物理量：单元的声速，rho,presure,xy方向速度
                C_k = w_k_all[6]
                rho_k = w_k_all[0]
                P_k = w_k_all[4]
                U_k = w_k_all[1]
                V_k = w_k_all[2]
                                
                # 提取远场的物理量：单元的声速，rho,presure,xy方向速度
                C_inf = w_inf_all[6]
                rho_inf = w_inf_all[0]
                P_inf = w_inf_all[4]
                U_inf = w_inf_all[1]
                V_inf = w_inf_all[2]
                
                # 计算边界上的法向量和切向量
                vec_tau = np.array([dx, dy]) / ds       # 单位切向量
                vec_norm = np.array([dy, -dx]) /ds      # 单位法向量
                
                
                # 计算单元在边界上的法线和切线速度分量
                Un_k = np.dot(vec_norm, w_k_all[1:3])   # 单元在边界上的法线分量
                Ut_k = np.dot(vec_tau, w_k_all[1:3])
                
                # 计算远场条件在边界上的法线和切线速度分量
                Un_inf = np.dot(vec_norm, w_inf_all[1:3]) # 无穷量来流在法线上的速度分量
                Ut_inf = np.dot(vec_tau, w_inf_all[1:3])
                
                # 根据特征值计算边界上的法线速度和声速
                R_p = Un_k + 2 * C_k / (self.case.gamma - 1)
                R_m = Un_inf - 2 * C_inf / (self.case.gamma - 1)
                
                Un_edge = (R_p + R_m) / 2
                C_edge = (R_p - R_m) * (self.case.gamma - 1) / 4
                Man_inf = np.abs(Un_inf / C_inf)  # 马赫数？
                
                '''
                法向量（vec_norm）: 边界上的法向量是垂直于边界表面的一个单位向量。
                在二维模拟中，如果边界是由一条线段定义的，那么这条线段的法向量就是垂直于这条线的向量。
                这个向量用于计算流体相对于边界的正交（法线）速度分量。

                切向量（vec_tau）: 切向量是沿着边界表面的一个单位向量。
                在二维模拟中，这通常是边界线段自身的方向，用于计算沿着边界的切线速度分量。
                
                单元在边界上的法线和切线速度分量（Un_k, Ut_k）: 
                这些速度分量是指流体在单元边界上的速度在法线和切线方向上的分量。
                这些分量可以帮助确定流体是否朝向边界进入或离开边界
                
                远场条件在边界上的法线和切线速度分量（Un_inf, Ut_inf）: 
                这些速度分量是指定义在边界上的远场流体状态的速度分量。
                "远场"通常指的是足够远离物体，以至于其存在不会显著影响流体速度分布的地方。
                这些远场速度分量通常用于定义开放边界条件
                '''
                
                if Un_edge <= 0: # 入流 inflow
                    if Man_inf <= 1: # subsonic
                        s = P_inf / rho_inf ** self.case.gamma
                        rho_edge = (C_edge ** 2 / s / self.case.gamma) ** (1 / (self.case.gamma - 1))
                        P_edge = s * rho_edge ** self.case.gamma 
                        U_edge = U_inf + (Un_edge - Un_inf) * vec_tau[1]
                        V_edge = V_inf + (Un_edge - Un_inf) * vec_tau[0]
                        Z = Un_edge * ds
                        flux = np.array([rho_edge * Z,
                                      Z * rho_edge * U_edge + P_edge * dy,
                                      Z * rho_edge * V_edge - P_edge * dx,
                                      Z * rho_edge * (self.case.gamma * P_edge / (self.case.gamma -1)/rho_edge + (U_edge**2 + V_edge**2)/2)
                                      ])
                        Q[k] = Q[k] + flux
                    else:
                        # 超声速入流 supersonic inflow
                        rho_edge, U_edge, V_edge, E_edge, P_edge, H_edge, C_edge = self.case.w_inf_all
                        flux = self.case.calcualte_flux(rho_edge, U_edge, V_edge, P_edge, H_edge, dx, dy)
                        Q[k] = Q[k] + flux
                        pass
                else: # 出流 压声速
                    if Man_inf <= 1:
                        s = P_k / rho_k ** self.case.gamma
                        rho_edge = (C_edge ** 2 / s / self.case.gamma) ** (1 / (self.case.gamma - 1))
                        P_edge = s * rho_edge ** self.case.gamma
                        U_edge = U_k + (Un_edge - Un_k) * vec_tau[1]
                        V_edge = V_k + (Un_edge - Un_k) * vec_tau[0]
                        Z = Un_edge * ds
                        flux = np.array([rho_edge * Z,
                                      Z * rho_edge * U_edge + P_edge * dy,
                                      Z * rho_edge * V_edge - P_edge * dx,
                                      Z * rho_edge * (self.case.gamma * P_edge / (self.case.gamma -1)/rho_edge + (U_edge**2 + V_edge**2)/2)
                                      ])
                        Q[k] = Q[k] + flux
                        pass                        
                    else: # 超声速出流 supersonic outflow
                        rho_edge, U_edge, V_edge, E_edge, P_edge, H_edge, C_edge = self.w_inf_all[k]
                        flux = self.case.calcualte_flux(rho_edge, U_edge, V_edge, P_edge, H_edge, dx, dy)
                        Q[k] = Q[k] + flux
                        pass          
                    pass                   
                pass
            
            else: # k单元w的值
                # 通量计算 flux calculate
                w_k = self.W_cur[k]
                w_p = self.W_cur[p]
                w_edge = (w_k + w_p) / 2 # 加权平均
                rho_edge, U_edge, V_edge, E_edge, P_edge, H_edge, C_edge = self.case.conserve_to_all(w_edge)
                flux = self.calculate_flux(rho_edge, U_edge, V_edge, P_edge, H_edge, dx, dy)
                Q[k] = Q[k] + flux
                Q[p] = Q[p] - flux
                
                # 2阶 & 4阶 粘性计算
                alpha_edge = np.abs(U_edge * dy - V_edge * dx) + C_edge * ds
                # k是左单元，p是右单元
                P_p = self.W_cur_all[p][4] #边连接矩阵？ 我的是ijxy，example是ijkp
                P_k = self.W_cur_all[k][4]
                nu_edge = np.abs((P_p - P_k) /(P_p + P_k))
                epsilon_edge2 = nu_edge * self.case.k2
                d2_edge = alpha_edge * epsilon_edge2 *(w_p - w_k) ### 代码核对
                D2[k] = D2[k] + d2_edge
                D2[p] = D2[p] - d2_edge
                
                epsilon_edge4 = np.max([0, self.case.k4 - epsilon_edge2])
                d4_edge = -alpha_edge * epsilon_edge4 *(self.W_cur_laplace[p] - self.W_cur_laplace[k])
                D4[k] = D4[k] + d4_edge
                D4[p] = D4[p] - d4_edge

                t[k] = t[k] + alpha_edge
                t[p] = t[p] + alpha_edge
                pass
            pass
        t = self.case.CFL * self.grid.vol/t
        self.Q = Q
        self.D = D2 + D4
        self.t = t
        pass


    def each_step_non_vis(self):    # flux 通量计算
        self.conserve_to_all()
        self.calculate_laplace()
        Q = np.zeros([self.grid.ncells, 4])

        for edge in range(self.grid.nedges): #边连接矩阵？ 我的是ijxy，example是ijkp
            i,j,k,p = self.grid.iedge[edge]
            dx = self.grid.dx[edge]
            dy = self.grid.dy[edge]
            ds = self.grid.ds[edge]
            if p == -2:
                # wall boundary 地面条件/物面边界条件:不允许流体通过的边界，比如固体表面。在这种情况下，流体的法线速度分量将是零
                w_k = self.W_cur[k]
                
                # p = self.case.conserve_to_all(w_k)
                p = self.W_cur_all[k][4]
                flux = np.array([0, p*dy, -p*dx, 0])
                Q[k] = Q[k] + flux
                pass
            
            
            elif p == -3: # 数字要调整
                # far-filed boundary 边场边界: 这种边界允许流体自由进出。远场边界条件通常用于模拟流体在足够远处的自由流动状态，不受物体影响的区域
                w_k = self.W_cur[k]
                w_k_all = self.W_cur_all[k]
                w_inf =  self.case.w_inf_all
                w_inf_all = self.case.w_inf_all
                
                C_k = w_k_all[6]
                rho_k = w_k_all[0]
                P_k = w_k_all[4]
                U_k = w_k_all[1]
                V_k = w_k_all[2]
                                
                C_inf = w_inf_all[6]
                rho_inf = w_inf_all[0]
                P_inf = w_inf_all[4]
                U_inf = w_inf_all[1]
                V_inf = w_inf_all[2]
                
                vec_tau = np.array([dx, dy]) / ds   # 那条边的切向方向
                vec_norm = np.array([dy, -dx]) /ds  # 法向方向
                Un_k = np.dot(vec_norm, w_k_all[1:3])   # 单元在边界上的法线分量
                Ut_k = np.dot(vec_tau, w_k_all[1:3])
                Un_inf = np.dot(vec_norm, w_inf_all[1:3]) # 无穷量来流在法线上的速度分量
                Ut_inf = np.dot(vec_tau, w_inf_all[1:3])
                
                R_p = Un_k + 2 * C_k / (self.case.gamma - 1)
                R_m = Un_inf - 2 * C_inf / (self.case.gamma - 1)
                
                Un_edge = (R_p + R_m) / 2
                C_edge = (R_p - R_m) * (self.case.gamma - 1) / 4
                Man_inf = np.abs(Un_inf / C_inf)  # 马赫数？
                
                if Un_edge <= 0: # 入流 inflow
                    if Man_inf <= 1: # subsonic
                        s = P_inf / rho_inf ** self.case.gamma
                        rho_edge = (C_edge ** 2 / s / self.case.gamma) ** (1 / (self.case.gamma - 1))
                        P_edge = s * rho_edge ** self.case.gamma 
                        U_edge = U_inf + (Un_edge - Un_inf) * vec_tau[1]
                        V_edge = V_inf + (Un_edge - Un_inf) * vec_tau[0]
                        Z = Un_edge * ds
                        flux = np.array([rho_edge * Z,
                                      Z * rho_edge * U_edge + P_edge * dy,
                                      Z * rho_edge * V_edge - P_edge * dx,
                                      Z * rho_edge * (self.case.gamma * P_edge / (self.case.gamma -1)/rho_edge + (U_edge**2 + V_edge**2)/2)
                                      ])
                        Q[k] = Q[k] + flux
                    else:
                        # 超声速入流 supersonic inflow
                        rho_edge, U_edge, V_edge, E_edge, P_edge, H_edge, C_edge = self.case.w_inf_all
                        flux = self.case.calcualte_flux(rho_edge, U_edge, V_edge, P_edge, H_edge, dx, dy)
                        Q[k] = Q[k] + flux
                        pass
                else: # 出流 压声速
                    if Man_inf <= 1:
                        s = P_k / rho_k ** self.case.gamma
                        rho_edge = (C_edge ** 2 / s / self.case.gamma) ** (1 / (self.case.gamma - 1))
                        P_edge = s * rho_edge ** self.case.gamma
                        U_edge = U_k + (Un_edge - Un_k) * vec_tau[1]
                        V_edge = V_k + (Un_edge - Un_k) * vec_tau[0]
                        Z = Un_edge * ds
                        flux = np.array([rho_edge * Z,
                                      Z * rho_edge * U_edge + P_edge * dy,
                                      Z * rho_edge * V_edge - P_edge * dx,
                                      Z * rho_edge * (self.case.gamma * P_edge / (self.case.gamma -1)/rho_edge + (U_edge**2 + V_edge**2)/2)
                                      ])
                        Q[k] = Q[k] + flux
                        pass                        
                    else: # 超声速出流 supersonic outflow
                        rho_edge, U_edge, V_edge, E_edge, P_edge, H_edge, C_edge = self.w_inf_all[k]
                        flux = self.case.calcualte_flux(rho_edge, U_edge, V_edge, P_edge, H_edge, dx, dy)
                        Q[k] = Q[k] + flux
                        pass          
                    pass                   
                pass
        
            else: # k单元w的值 # 处理内部单元边界的情况
                # 通量计算 flux calculate
                # 从当前时间步的网格单元k和p中获取守恒变量
                
                w_k = self.W_cur[k]      # 获取网格单元k,p的守恒变量
                w_p = self.W_cur[p]
                w_edge = (w_k + w_p) / 2 # 加权平均
                
                # 将边界守恒变量转换为全变量，如密度、速度、能量等
                rho_edge, U_edge, V_edge, E_edge, P_edge, H_edge, C_edge = self.case.conserve_to_all(w_edge)
                # 转换后的全变量和边界的几何尺寸来计算通量
                flux = self.calculate_flux(rho_edge, U_edge, V_edge, P_edge, H_edge, dx, dy)
                Q[k] = Q[k] + flux   # 将计算出的通量加到网格单元k的通量总和中
                Q[p] = Q[p] - flux   # 并从网格单元p的通量总和中减去，保持通量守恒           
                pass
            pass       
        self.Q = Q
        pass   
    

    # def rk4_iteration(self):
    #     self.each_step()
    #     dt = np.min(self.t)
    #     W0 = copy.copy(self.W_cur)
    #     for step in range(4):
    #         self.W_cur = (W0 - (dt * rk4_coeff[step] * (self.Q - self.D)).T / self.grid.vol).T
    #         self.each_step_non_vis()
    #         pass
    #     pass


    def rk4_iteration(self):
        self.each_step()
        dt = np.min(self.t)
        W0 = copy.copy(self.W_cur)
        for step in range(4):
            self.W_cur = W0 - ((dt * rk4_coeff[step] * (self.Q - self.D)).T / self.grid.vol).T
            self.each_step_non_vis()
            pass
        residual = np.max(np.abs(self.W_cur - W0))
        return residual / self.residual_base
        pass
    

    # def post_process(self): # 如何调整这一块画出pressure云图
    #     rho_node = np.zeros(self.grid.nnodes)
    #     vol_add_node = np.zeros(self.grid.nnodes)
    #     for k in range(self.grid.ncells):
    #         for i in range(3):
    #             ji = self.grid.icell[k, i]
    #             rho_node[ji] = rho_node[ji] + self.W_cur[k][0] * self.grid.vol[k]
    #             vol_add_node[ji] = vol_add_node[ji] + self.grid.vol[k]
    #             pass
    #         pass
    #     self.rho_result = rho_node / vol_add_node
    #     pass
    
    def post_process(self):
        rho_node = np.zeros(self.grid.nnodes)
        pressure_node = np.zeros(self.grid.nnodes)
        vol_add_node = np.zeros(self.grid.nnodes)
        for k in range(self.grid.ncells):
            for i in range(3):
                ji = self.grid.icell[k, i]
                rho_node[ji] += self.W_cur[k][0] * self.grid.vol[k]
                pressure_node[ji] += self.W_cur_all[k][4] * self.grid.vol[k]
                vol_add_node[ji] += self.grid.vol[k]
            pass
        self.rho_result = rho_node / vol_add_node
        self.pressure_result = pressure_node / vol_add_node
        pass

    
    
    def start_simulate(self):
        # ncell = self.grid.ncells  # 网格单元的数量
        
        # Initialize
        self.initialize_field()       
        self.residual_base = 1
        self.rk4_iteration()
        self.residual = np.zeros(self.case.STEP)
        self.residual_observer = np.zeros(self.case.STEP)  # 创建用于存储 Observer Residual 的数组
        
        # Set the initial residual
        self.residual[0] = self.rk4_iteration()
        self.residual_base = np.max(np.abs(self.W_cur - self.case.w_inf))
        self.residual_observer[0] = np.log10(np.sum(self.residual[0]**2) / self.grid.ncells)
        
        # Perform the simulation iterations
        for step in range(1, self.case.STEP):
            # Update the current residual
            self.residual[step] = self.rk4_iteration()
            
            # Calculate and update the observer residual
            # self.residual_observer[step] = np.log10(np.sum(self.residual[step]**2) / ncell)
            self.residual_observer[step] = np.log10(np.sum(self.residual[step]**2) / self.grid.ncells)
            
            print("-----------------------------")
            print("step: " + str(step))
            print("residual: " + str(self.residual[step]))
            print("residual_observer: " + str(self.residual_observer[step]))
            print("-----------------------------")
            # importnat residual
            pass
        
        self.post_process() # 流场的后处理 post-processing
        
        # 绘制 Observer Residual 曲线图
        plt.plot(self.residual_observer)
        plt.xlabel('Iteration Steps')
        plt.ylabel('Observer Residual')
        plt.title('Observer Residual vs. Iteration Steps \n Mach 0.80 and AOA 1.25 degree — Transonic')
        plt.show()
        
        pass

    # def start_simulate(self):
    #     self.initialize_field()
    #     self.residual_base = 1
    #     self.rk4_iteration()
    #     self.residual = np.zeros(self.case.STEP)
    #     residual_observer = np.zeros(self.case.STEP)  # 创建用于存储 Observer Residual 的数组
    #     self.residual[0] = 1
    #     self.residual_base = np.max(np.abs(self.W_cur - self.case.w_inf))
    #     ncell = self.grid.ncells  # 网格单元的数量
    #     for step in range(1, self.case.STEP):
    #         self.residual[step] = self.rk4_iteration()
    #         # 计算 Observer Residual
    #         residual_observer[step] = np.log10(np.sum(self.residual[step]**2) / ncell)
    #         print("-----------------------------")
    #         print("step: " + str(step))
    #         print("residual: " + str(self.residual[step]))
    #         print("residual_observer: " + str(residual_observer[step]))
    #         print("-----------------------------")
    #         pass
    #     self.post_process()
    
    #     # 绘制 Observer Residual 曲线图
    #     plt.plot(residual_observer)
    #     plt.xlabel('Iteration Steps')
    #     plt.ylabel('Observer Residual')
    #     plt.title('Observer Residual vs. Iteration Steps')
    #     plt.show()    




    # def start_simulate(self):
    #     self.initialize_field()
    #     for step in range(self.case.STEP):
    #         self.rk4_iteration()
    #         print(step)
    #         pass
    #     pass
    # pass



##################################

# def main():
#     # grid_file = "C:\Users\yanya\OneDrive\Desktop\Aero 689\final proj\naca0012_80x16_O-mesh"
#     grid_file = r"C:\Users\yanya\OneDrive\Desktop\Aero 689\final proj\naca0012_nanjing\computation\new\naca0012.grd"

#     grid = Grid(grid_file)
#     case = Case(0.8,0) # 马赫，AOA
#     solver = Solver(case,grid)
#     solver.start_simulate()
#     pass


# main()
##########################################


grid_file = r"C:\Users\yanya\OneDrive\Desktop\Aero 689\final proj\naca0012_nanjing\computation\new\naca0012.grd"   
grid = Grid(grid_file)
# 从这里设置mach muner和AOA
case = Case(0.8, 1.25) #  Mach number & AOA in degree
solver = Solver(case, grid)
solver.start_simulate()

triangle = tri.Triangulation(solver.grid.xy.T[0], solver.grid.xy.T[1], solver.grid.icell)
#################################################################
# # 云图
# plt.figure(figsize=(6, 4), dpi=200, facecolor="white")
# plt.gca().set_aspect(1)

# plt.tricontourf(triangle, solver.rho_result, cmap="jet")
# plt.colorbar()

# plt.xlim(-1.5, 2.5)
# plt.ylim(-2, 2)

# plt.xlabel("$x$")
# plt.ylabel("$y$")
# plt.title(r"$\rho$ \\ Mach 0.80 and AOA 1.25 degree — Transonic")

# # plt.savefig("../image/rho_contour.png", bbox_inches="tight")
# # plt.savefig("../image/rho_contour.pdf", bbox_inches="tight")
# plt.show()

# 清理rho_result中的非有限值
solver.rho_result = np.nan_to_num(solver.rho_result)

# 清理pressure_result中的非有限值
solver.pressure_result = np.nan_to_num(solver.pressure_result)

# 然后绘制密度云图
plt.figure(figsize=(6, 4), dpi=200, facecolor="white")
plt.tricontourf(triangle, solver.rho_result, cmap="jet")
plt.colorbar()
plt.xlim(-1.5, 2.5)
plt.ylim(-2, 2)
plt.title("Density Contour \n Mach 0.80 and AOA 1.25 degree — Transonic")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()

# 接着绘制压力云图
plt.figure(figsize=(6, 4), dpi=200, facecolor="white")
plt.tricontourf(triangle, solver.pressure_result, cmap="jet")
plt.colorbar()
plt.xlim(-1.5, 2.5)
plt.ylim(-2, 2)
plt.title("Pressure Contou \n Mach 0.80 and AOA 1.25 degree — Transonic")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()


##############################################################################
print("RHO_result: \n",solver.rho_result)
print("Residual: \n",solver.residual)
print("Residual Observer: \n",solver.residual_observer)

# -8 是比较好的数据
# Residual:残差 是指实际观测值和模型或方程预测值之间的差异
# triangle mesh plot
g = Grid(grid_file)
g.plot_grid('')  # 传入一个空字符串作为 image_name 参数


# 5:35 pm 开始跑