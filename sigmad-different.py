import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import time

# %% 1. 固定所有基础参数（不变，确保与此前模型连贯）
# 1.1 生命史与感染基础参数
α1 = 0.25;
α2 = 0.08;
α3 = 0.25;
μ1 = 0.005;
μ2 = 0.015;
μ3 = 0.01;
μ4 = 0.15;
b = 0.6;
K = 0.1113;
μ5 = 0.12; σ_fixed = 0.3; d_fixed = 0.3  # σ和d固定为0.3，避免干扰β分析
# 1.2 时变参数相关
t1, t2, t3, t4 = 7, 14, 19, 27
base_γ = 0.5823
base_α4 = 0.3461
γ_coeff = [0.01, 2.0, 7.8, 0.3, 0.001];
α4_coeff = [1.0, 0.2, 0.05, 2.0, 30.0]
# 1.3 固定初始条件
A0 = 0.0004; E0 = A0*10; Q0 = A0*8; P0 = A0*5; I0_fixed = 3
t_full = np.arange(1, 32)  # 1-31天

# %% 2. 含σ、d、β的EQPA模型（兼容三类参数分析）
def eqpa_param_model(t, y, sigma, delta, beta):
    E, Q, P, A, I = y
    # 计算时变参数γ和α4
    if t <= t1:
        γ = base_γ * γ_coeff[0]; α4 = base_α4 * α4_coeff[0]
    elif t1 < t <= t2:
        γ = base_γ * γ_coeff[1]; α4 = base_α4 * α4_coeff[1]
    elif t2 < t <= t3:
        γ = base_γ * γ_coeff[2]; α4 = base_α4 * α4_coeff[2]
    elif t3 < t <= t4:
        γ = base_γ * γ_coeff[3]; α4 = base_α4 * α4_coeff[3]
    else:
        γ = base_γ * γ_coeff[4]; α4 = base_α4 * α4_coeff[4]
    # 微分方程组
    dE_dt = b*A - (α1+μ1)*E
    dQ_dt = α1*E - (α2+μ2)*Q
    dP_dt = α2*Q - (α3+μ3)*P
    dA_dt = γ*A*(1-A/K) + α3*P - (α4+μ4)*A - beta*A*I
    dI_dt = beta*A*I - (μ5 + sigma * delta) * I
    return [dE_dt, dQ_dt, dP_dt, dA_dt, dI_dt]

# %% 3. 定义三类参数梯度（各5个值，保持一致性）
sigma_list = [0.1, 0.3, 0.5, 0.7, 1.0]
delta_list = [0.1, 0.3, 0.5, 0.7, 1.0]
beta_list = [0.1, 0.3, 0.5, 0.7, 1.0]  # 新增β梯度，符合生物意义

# %% 4. 批量计算函数（兼容σ、d、β三类参数）
def calculate_A_I_for_param(param_list, param_type):
    """
    计算不同参数下的A(t)和I(t)
    param_type: 'sigma'/'delta'/'beta'
    返回：(A_results列表, I_results列表)
    """
    A_results = []
    I_results = []
    start_time = time.time()
    for param in param_list:
        y0 = [E0, Q0, P0, A0, I0_fixed]
        if param_type == 'sigma':
            sol = solve_ivp(
                fun=lambda t,y: eqpa_param_model(t, y, sigma=param, delta=d_fixed, beta=0.8),
                t_span=(1, 31), y0=y0, t_eval=t_full,
                method='RK45', max_step=0.05, atol=1e-12, rtol=1e-10
            )
        elif param_type == 'delta':
            sol = solve_ivp(
                fun=lambda t,y: eqpa_param_model(t, y, sigma=σ_fixed, delta=param, beta=0.8),
                t_span=(1, 31), y0=y0, t_eval=t_full,
                method='RK45', max_step=0.05, atol=1e-12, rtol=1e-10
            )
        elif param_type == 'beta':
            sol = solve_ivp(
                fun=lambda t,y: eqpa_param_model(t, y, sigma=σ_fixed, delta=d_fixed, beta=param),
                t_span=(1, 31), y0=y0, t_eval=t_full,
                method='RK45', max_step=0.05, atol=1e-12, rtol=1e-10
            )
        # 提取结果
        A_fit = interp1d(sol.t, sol.y[3], kind='linear')(t_full)
        I_fit = interp1d(sol.t, sol.y[4], kind='linear')(t_full)
        A_results.append(A_fit)
        I_results.append(I_fit)
    print(f"所有{param_type}参数的A(t)和I(t)计算完成，运行时间：{time.time() - start_time:.2f}秒")
    return A_results, I_results

# 执行三类参数计算
A_sigma, I_sigma = calculate_A_I_for_param(sigma_list, 'sigma')
A_delta, I_delta = calculate_A_I_for_param(delta_list, 'delta')
A_beta, I_beta = calculate_A_I_for_param(beta_list, 'beta')  # 新增β结果

# %% 5. 3行2列组图可视化（核心优化：仅第二行保留Y轴文字）
plt.rcParams['font.family'] = ['Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['lines.linewidth'] = 2.2
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.95
plt.rcParams['legend.edgecolor'] = 'black'

# 1. 创建3行2列布局（宽16，高15，更紧凑美观）
fig, axes = plt.subplots(3, 2, figsize=(16, 15))
plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08, wspace=0.25, hspace=0.4)

# 2. 核心优化：统一前文拟合图配色（与Figure4保持一致）+ 差异化线型
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 前文拟合图配色（蓝、橙、绿、红、紫）
linestyles = ['-', '--', '-.', ':', '-']  # 差异化线型（实线、虚线、点划线、点线、粗实线）
linewidths = [2.2, 2.2, 2.2, 2.5, 2.8]  # 粗细区分，增强可读性
labels_sigma = [f'$\sigma = {s}$' for s in sigma_list]
labels_delta = [f'$d = {d}$' for d in delta_list]
labels_beta = [f'$β= {b}$' for b in beta_list]
y_label_A = 'Susceptible Adult Density (individuals/m²)'
y_label_I = 'Infected Adult Density (individuals/m²)'

# 3. 第1行：σ的影响（隐藏Y轴文字）
ax1 = axes[0, 0]  # σ对A(t)
for A_fit, color, ls, lw, label in zip(A_sigma, colors, linestyles, linewidths, labels_sigma):
    ax1.plot(t_full, A_fit, color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.95)
ax1.set_xlabel('Time (day)', fontsize=13, fontweight='bold')
# 隐藏Y轴文字
ax1.set_title('(A) Effect of $\sigma$ on Susceptible Adults ($A(t)$)', fontweight='bold', loc='left', pad=15)
ax1.legend(loc='upper left', ncol=1)
ax1.set_xticks(t_full[::4])  # 每4天一个刻度，更简洁
ax1.set_xlim(1, 31); ax1.set_ylim(bottom=0)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
ax1.grid(True, alpha=0.15, linestyle='-', linewidth=0.8)
ax1.set_yticklabels([])  # 取消Y轴刻度文字

ax2 = axes[0, 1]  # σ对I(t)
for I_fit, color, ls, lw, label in zip(I_sigma, colors, linestyles, linewidths, labels_sigma):
    ax2.plot(t_full, I_fit, color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.95)
ax2.set_xlabel('Time (day)', fontsize=13, fontweight='bold')
# 隐藏Y轴文字
ax2.set_title('(B) Effect of $\sigma$ on Infected Adults ($I(t)$)', fontweight='bold', loc='left', pad=15)
ax2.legend(loc='upper right', ncol=1)
ax2.set_xticks(t_full[::4])
ax2.set_xlim(1, 31); ax2.set_ylim(bottom=0)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
ax2.grid(True, alpha=0.15, linestyle='-', linewidth=0.8)
ax2.set_yticklabels([])  # 取消Y轴刻度文字

# 4. 第2行：d的影响（保留Y轴文字）
ax3 = axes[1, 0]  # d对A(t)
for A_fit, color, ls, lw, label in zip(A_delta, colors, linestyles, linewidths, labels_delta):
    ax3.plot(t_full, A_fit, color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.95)
ax3.set_xlabel('Time (day)', fontsize=13, fontweight='bold')
ax3.set_ylabel(y_label_A, fontsize=13, fontweight='bold')  # 保留Y轴文字
ax3.set_title('(C) Effect of $d$ on Susceptible Adults ($A(t)$)', fontweight='bold', loc='left', pad=15)
ax3.legend(loc='upper left', ncol=1)
ax3.set_xticks(t_full[::4])
ax3.set_xlim(1, 31); ax3.set_ylim(bottom=0)
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
ax3.grid(True, alpha=0.15, linestyle='-', linewidth=0.8)

ax4 = axes[1, 1]  # d对I(t)
for I_fit, color, ls, lw, label in zip(I_delta, colors, linestyles, linewidths, labels_delta):
    ax4.plot(t_full, I_fit, color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.95)
ax4.set_xlabel('Time (day)', fontsize=13, fontweight='bold')
ax4.set_ylabel(y_label_I, fontsize=13, fontweight='bold')  # 保留Y轴文字
ax4.set_title('(D) Effect of $d$ on Infected Adults ($I(t)$)', fontweight='bold', loc='left', pad=15)
ax4.legend(loc='upper right', ncol=1)
ax4.set_xticks(t_full[::4])
ax4.set_xlim(1, 31); ax4.set_ylim(bottom=0)
ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)
ax4.grid(True, alpha=0.15, linestyle='-', linewidth=0.8)

# 5. 第3行：β的影响（隐藏Y轴文字）
ax5 = axes[2, 0]  # β对A(t)
for A_fit, color, ls, lw, label in zip(A_beta, colors, linestyles, linewidths, labels_beta):
    ax5.plot(t_full, A_fit, color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.95)
ax5.set_xlabel('Time (day)', fontsize=13, fontweight='bold')
# 隐藏Y轴文字
ax5.set_title('(E) Effect of β on Susceptible Adults ($A(t)$)', fontweight='bold', loc='left', pad=15)
ax5.legend(loc='upper left', ncol=1)
ax5.set_xticks(t_full[::4])
ax5.set_xlim(1, 31); ax5.set_ylim(bottom=0)
ax5.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False)
ax5.grid(True, alpha=0.15, linestyle='-', linewidth=0.8)
ax5.set_yticklabels([])  # 取消Y轴刻度文字

ax6 = axes[2, 1]  # β对I(t)
for I_fit, color, ls, lw, label in zip(I_beta, colors, linestyles, linewidths, labels_beta):
    ax6.plot(t_full, I_fit, color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.95)
ax6.set_xlabel('Time (day)', fontsize=13, fontweight='bold')
# 隐藏Y轴文字
ax6.set_title('(F) Effect of β on Infected Adults ($I(t)$)', fontweight='bold', loc='left', pad=15)
ax6.legend(loc='upper right', ncol=1)
ax6.set_xticks(t_full[::4])
ax6.set_xlim(1, 31); ax6.set_ylim(bottom=0)
ax6.spines['top'].set_visible(False); ax6.spines['right'].set_visible(False)
ax6.grid(True, alpha=0.15, linestyle='-', linewidth=0.8)
ax6.set_yticklabels([])  # 取消Y轴刻度文字

# 6. 保存高清图（SCI标准）
plt.savefig('Sigma_Delta_Beta_Effect_on_A_I_3x2_Final.svg', dpi=600, format='svg', bbox_inches='tight')
plt.savefig('Sigma_Delta_Beta_Effect_on_A_I_3x2_Final.png', dpi=600, format='png', bbox_inches='tight')
plt.savefig('Sigma_Delta_Beta_Effect_on_A_I_3x2_Final.eps', dpi=600, format='eps', bbox_inches='tight')

# 显示图像
plt.show()
