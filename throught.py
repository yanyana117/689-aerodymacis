# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:06:31 2024

@author: yanyan
"""
import numpy as np
import matplotlib.pyplot as plt
#from numpy import ogrid
# import pymesh

file_path = r"C:\Users\yanya\OneDrive\Desktop\Aero 689\final proj\grid_44_20.dat"

# 初始化 x 和 y 坐标的列表
XMat_initial = []
YMat_initial = []

# 打开文件并读取每一行
with open(file_path, 'r') as file:
    for line in file:
        # 去掉空白字符（如空格和换行符），然后分割每一行为 x 和 y 坐标
        x, y = line.strip().split()
        # 将字符串坐标转换为浮点数，并添加到相应的列表中
        XMat_initial.append(float(x))
        YMat_initial.append(float(y))

# 检查读取的结果
print(f"读取了 {len(XMat_initial)} 个 x 坐标和 {len(YMat_initial)} 个 y 坐标。")

# 打印坐标测试
print("一些 x 坐标的示例:", XMat_initial[:5], "一些 y 坐标的示例:", YMat_initial[:5])
coordinate_pairs = [(x, y) for x, y in zip(XMat_initial[:5], YMat_initial[:5])]
print("一些坐标对的示例:", coordinate_pairs)

Nx = YMat_initial.index(YMat_initial[0], 1) - 2 
Ny = len(YMat_initial) // (Nx + 1) - 1 

print("估计的 Nx:", Nx)
print("估计的 Ny:", Ny)

Imax = Nx + 1 # total number!!
Jmax = Ny + 1

# 创建新的二维矩阵
XMat = []
YMat = []

# 将一维坐标列表转换为二维矩阵
for j in range(Jmax):
    start_index = j * Imax
    end_index = start_index + Imax
    XMat.append(XMat_initial[start_index:end_index])
    YMat.append(YMat_initial[start_index:end_index])

# 然后我们可以检查它们的尺寸
print("Number of rows (Jmax):", len(XMat),Jmax)
print("Number of columns (Imax):", len(XMat[0]),Imax)

# 绘制空气动力翼型的形状
num_circles = Nx
num_points_per_circle = Ny

# Draw airfoil shap to circle(j = 1 to Jma )
# 将点分组为每个形状
shapes = []  # 创建一个空列表用于存储所有形状的点
for i in range(Imax):  # 循环遍历每个形状
    start_index = i * Jmax  # 计算当前形状起始点的索引，每个图像开始的索引
    end_index = start_index + Jmax  # 计算当前形状结束点的索引
    # 通过切片获取当前形状的点坐标，并转换为元组的形式，添加到形状列表中
    shape = [(XMat_initial[j], YMat_initial[j]) for j in range(start_index, end_index)]
    shapes.append(shape)

# 创建一个图形窗口，并设置尺寸
plt.figure(figsize=(10, 8))
# 遍历绘制每个形状
for shape in shapes:
    x, y = zip(*shape)  # 将形状中的点坐标分离为 x 和 y 坐标
    plt.plot(x, y,'--', marker='.', color='#1f77b4') # 绘制形状的连线，点使用圆圈标记

# Drew circumference index
# 绘制形状的点之间的连线
for i in range(Jmax):
    x = [shape[i][0] for shape in shapes]  # 获取所有形状中第 i 个点的 x 坐标
    y = [shape[i][1] for shape in shapes]  # 获取所有形状中第 i 个点的 y 坐标
    plt.plot(x, y, linestyle='--', color='#1f77b4')  # 绘制所有形状中第 i 个点的连线，使用虚线


plt.title('NACA 0012: 44*20 grid')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
plt.axis([-0.5, 1.5, -0.5, 0.5])  # 设置坐标轴范围
plt.show()

##################################################################################
## Set up BC: NFarField, NAirfoil, and NPeriodic
# 绘制形状的连线
for shape in shapes:
    x, y = zip(*shape)  # 将形状中的点坐标分离为 x 和 y 坐标
    plt.axis('equal')
    plt.plot(x, y, '--', marker='.', color='#1f77b4', label='Shape')  # 添加 label 参数


# 绘制所有形状中第 i 个点的连线
for i in range(Jmax):
    x = [shape[i][0] for shape in shapes]
    y = [shape[i][1] for shape in shapes]
    plt.axis('equal')
    plt.plot(x, y, linestyle='--', color='#1f77b4', label=f'Line {i}')  # 添加 label 参数

# 这里证明最外圈是个圆形的图片

############################################################################################
NFarField = []

for i in range(1, Imax + 1):
    node_index_top = (Jmax -1) * Imax + i  # j = Jmax 时的索引
    NFarField.append(node_index_top)
    
for i in range(0, Imax + 2):
    node_index_top = (Jmax - 1) * Imax - i  # j = Jmax 时的索引
    NFarField.append(node_index_top)

print("NFarField:", NFarField)        
print("NFarField:", NFarField)
print("size NFarField:", np.size(NFarField))

# 绘制空气动力翼型的形状
plt.figure(figsize=(10, 8))

# 绘制网格线
for shape in shapes:
    x, y = zip(*shape)
    plt.plot(x, y, '--', marker='.', color='#1f77b4')

for i in range(Jmax):
    x = [shape[i][0] for shape in shapes]
    y = [shape[i][1] for shape in shapes]
    plt.plot(x, y, linestyle='--', color='#1f77b4')

# 标注NFarField的节点为红色
for node_index in NFarField:
    # 由于node_index是从1开始的，我们需要减1来转换为0-based索引
    j_index = (node_index - 1) // Imax
    i_index = (node_index - 1) % Imax
    plt.plot(XMat[j_index][i_index], YMat[j_index][i_index], 'o', color='red')
    
    
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.title('NFarField')

plt.axis('equal')
plt.show()

######################################################################
# NAirfoil 
NAirfoil = []
J_inner_top = 2
J_inner_bottom = 1

for i in range(1, Imax + 1):
    # 计算内围上半层的节点索引
    node_index_inner_top = (J_inner_top - 1) * Imax + i
    NAirfoil.append(node_index_inner_top)
    
    # 计算下半层的节点索引（假设 j = J_inner_bottom 是底部边界）
    node_index_inner_bottom = (J_inner_bottom - 1) * Imax + i
    NAirfoil.append(node_index_inner_bottom)

# 绘制空气动力翼型的形状
plt.figure(figsize=(10, 8))

# 绘制网格线
for shape in shapes:
    x, y = zip(*shape)
    plt.plot(x, y, '--', marker='.', color='#1f77b4')

for i in range(Jmax):
    x = [shape[i][0] for shape in shapes]
    y = [shape[i][1] for shape in shapes]
    plt.plot(x, y, linestyle='--', color='#1f77b4')

    
    
# 标注NAirfoil的上表面节点为红色
for node_index in NAirfoil:
    # 注意这里假设您的网格是从1开始索引的
    x = XMat_initial[node_index-1]
    y = YMat_initial[node_index-1]
    plt.plot(x, y, 'o', color='red')
    


# 设置图形标题和轴标签
plt.title('NAirfoil')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
plt.axis([-0.5, 1.5, -0.5, 0.5])
plt.show()

##############################################################################
NPeriodic = []

# 遍历所有节点，找到满足条件的节点索引
for i in range(len(XMat_initial)):
    if YMat_initial[i] == 0 and XMat_initial[i] > 1:
        NPeriodic.append(i)

# 绘制空气动力翼型的形状
plt.figure(figsize=(10, 8))

# 绘制网格线
for shape in shapes:
    x, y = zip(*shape)
    plt.plot(x, y, '--', marker='.', color='#1f77b4')

for i in range(Jmax):
    x = [shape[i][0] for shape in shapes]
    y = [shape[i][1] for shape in shapes]
    plt.plot(x, y, linestyle='--', color='#1f77b4')
    
# 标注NPeriodic的节点为红色
for node_index in NPeriodic:
    x = XMat_initial[node_index]
    y = YMat_initial[node_index]
    plt.plot(x, y, 'o', color='red')

# 设置标题和轴标签
plt.title('NPeriodic')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
plt.show()
















