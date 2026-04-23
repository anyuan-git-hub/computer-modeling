# -*- coding: utf-8 -*-
"""
题目：根据测试数据估计汽车刹车距离模型中的参数 c1 和 c2

模型形式：
    d = c1 * v + c2 * v^2

其中：
    v : 车速（单位：km/h）
    d : 刹车距离（单位：m）

方法：
    使用最小二乘法拟合参数 c1 和 c2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# 设置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']      # 黑体
plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示问题


# =========================
# 1. 输入原始测试数据
# =========================
# 车速数据 v（单位：km/h）
v = np.array([20, 40, 60, 80, 100, 120, 140], dtype=float)

# 刹车距离数据 d（单位：m）
d = np.array([6.5, 17.8, 33.6, 57.1, 83.4, 118.0, 153.5], dtype=float)


# =========================
# 2. 构造最小二乘法中的设计矩阵 X
# =========================
# 根据模型 d = c1*v + c2*v^2
# 每一行对应一组数据 [v_i, v_i^2]
X = np.column_stack((v, v**2))

# 输出设计矩阵，便于检查
print("设计矩阵 X =")
print(X)
print()


# =========================
# 3. 最小二乘拟合参数 c1, c2
# =========================
# 使用 numpy 的最小二乘函数：
# coeffs 为拟合得到的参数 [c1, c2]
# residuals 为残差平方和
# rank 为矩阵秩
# s 为奇异值
coeffs, residuals, rank, s = np.linalg.lstsq(X, d, rcond=None)

# 取出参数
c1, c2 = coeffs

print("拟合得到的参数为：")
print(f"c1 = {c1:.6f}")
print(f"c2 = {c2:.6f}")
print()


# =========================
# 4. 构造拟合函数并计算预测值
# =========================
# 根据拟合结果计算每个测试点上的预测刹车距离
d_pred = c1 * v + c2 * v**2

print("各测试点的实际值与预测值对比：")
print("车速v(km/h)\t实际距离d(m)\t预测距离d_hat(m)\t误差")
for i in range(len(v)):
    error = d[i] - d_pred[i]
    print(f"{v[i]:>8.1f}\t{d[i]:>10.2f}\t{d_pred[i]:>14.2f}\t{error:>8.2f}")
print()


# =========================
# 5. 计算误差指标
# =========================
# 残差：实际值 - 预测值
errors = d - d_pred

# 均方误差 MSE
mse = np.mean(errors**2)

# 均方根误差 RMSE
rmse = np.sqrt(mse)

# 决定系数 R^2
# R^2 = 1 - SSE/SST
# SSE: 残差平方和
# SST: 总离差平方和
sse = np.sum(errors**2)
sst = np.sum((d - np.mean(d))**2)
r2 = 1 - sse / sst

print("误差评价指标：")
print(f"残差平方和 SSE = {sse:.6f}")
print(f"均方误差 MSE   = {mse:.6f}")
print(f"均方根误差 RMSE = {rmse:.6f}")
print(f"决定系数 R^2    = {r2:.6f}")
print()


# =========================
# 6. 输出最终模型
# =========================
print("最终拟合模型为：")
print(f"d = {c1:.6f} * v + {c2:.6f} * v^2")
print()


# =========================
# 7. 绘制原始数据点和拟合曲线
# =========================
# 为了让曲线更平滑，生成更密集的速度点
v_smooth = np.linspace(0, 150, 300)
d_smooth = c1 * v_smooth + c2 * v_smooth**2

plt.figure(figsize=(8, 6))

# 绘制原始数据散点
plt.scatter(v, d, label='测试数据')

# 绘制拟合曲线
plt.plot(v_smooth, d_smooth, label='最小二乘拟合曲线')

# 添加标题和坐标轴标签
plt.title('汽车刹车距离模型拟合')
plt.xlabel('车速 v (km/h)')
plt.ylabel('刹车距离 d (m)')

# 显示网格与图例
plt.grid(True)
plt.legend()

# 显示图像
plt.show()