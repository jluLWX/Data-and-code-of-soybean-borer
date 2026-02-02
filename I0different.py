import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import time
from matplotlib.gridspec import GridSpec

# %% 1. 固定所有参数（不变）
α1 = 0.25;
α2 = 0.08;
α3 = 0.25;
μ1 = 0.005;
μ2 = 0.015;
μ3 = 0.01;
μ4 = 0.15;
b = 0.6;
K = 0.1113;
β = 0.8;
μ5 = 0.12; σ = 0.8; d = 0.1
t1, t2, t3, t4 = 7, 14, 19, 27
base_γ = 0.5823
base_α4 = 0.3461
γ_coeff = [0.01, 2.0, 7.8, 0.3, 0.001];
α4_coeff = [1.0, 0.2, 0.05, 2.0, 30.0]

# %% 2. 含I仓室的EQPA模型（不变）
def eqpa_with_I_model(t, y, I0):
    E, Q, P, A, I = y
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
    dE_dt = b*A - (α1+μ1)*E;
    dQ_dt = α1*E - (α2+μ2)*Q;
    dP_dt = α2*Q - (α3+μ3)*P
    dA_dt = γ*A*(1-A/K) + α3*P - (α4+μ4)*A - β*A*I;
    dI_dt = β*A*I - (μ5+σ*d)*I
    return [dE_dt, dQ_dt, dP_dt, dA_dt, dI_dt]

# %% 3. 计算所有I0梯度下的5个仓室结果（不变）
I0_list = [1, 30, 50, 70, 90]; t_full = np.arange(1, 32); all_compartments = []
start_time = time.time()
for I0 in I0_list:
    A0 = 0.0004; E0 = A0*10; Q0 = A0*8; P0 = A0*5; y0 = [E0, Q0, P0, A0, I0]
    sol = solve_ivp(
        fun=lambda t,y: eqpa_with_I_model(t,y,I0), t_span=(1,31), y0=y0, t_eval=t_full,
        method='RK45', max_step=0.05, atol=1e-12, rtol=1e-10
    )
    E_fit = interp1d(sol.t, sol.y[0], kind='linear')(t_full)
    Q_fit = interp1d(sol.t, sol.y[1], kind='linear')(t_full)
    P_fit = interp1d(sol.t, sol.y[2], kind='linear')(t_full)
    A_fit = interp1d(sol.t, sol.y[3], kind='linear')(t_full)
    I_fit = interp1d(sol.t, sol.y[4], kind='linear')(t_full)
    all_compartments.append([E_fit, Q_fit, P_fit, A_fit, I_fit])
print(f"所有I0梯度计算完成，运行时间：{time.time() - start_time:.2f}秒")

# %% 4. 组图可视化（核心优化：颜色统一+线型差异化+峰值标注+竖线）
plt.rcParams['font.family'] = ['Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['lines.linewidth'] = 2.2
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10.5
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1.2

# 1. 画布与布局
fig = plt.figure(figsize=(18, 8))
gs = GridSpec(2, 3, figure=fig, width_ratios=[0.7, 0.18, 0.18], height_ratios=[1, 1])
ax_A = fig.add_subplot(gs[:, 0])
ax_E = fig.add_subplot(gs[0, 1])
ax_Q = fig.add_subplot(gs[0, 2])
ax_P = fig.add_subplot(gs[1, 1])
ax_I = fig.add_subplot(gs[1, 2])
plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12, wspace=0.3, hspace=0.5)

# 2. 核心优化：颜色与前文Figure4完全一致 + 差异化线型（适配黑白打印）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 与Figure4配色统一
linestyles = ['-', '--', '-.', ':', '-']  # 差异化线型
linewidths = [2.2, 2.2, 2.2, 2.5, 2.8]  # 部分加粗增强区分
legend_labels = [f'$I_0 = {i}$' for i in I0_list]
y_label_unified = 'Density (individuals/m²)'
compartment_titles = ['(B) Eggs ($E(t)$)', '(C) Larvae ($Q(t)$)', '(D) Pupae ($P(t)$)', '(E) Infected Adults ($I(t)$)']
peak_day = 19  # 峰值时间（第19天）

# 3. 左侧大面板：A(t)（重点优化：峰值竖线+数值标注）
peak_values_A = []  # 存储每条曲线的峰值
for compartments, color, ls, lw, label in zip(all_compartments, colors, linestyles, linewidths, legend_labels):
    A_data = compartments[3]
    ax_A.plot(t_full, A_data, color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.95)
    # 计算并存储峰值（第19天对应索引为18）
    peak_val = A_data[peak_day - 1]
    peak_values_A.append(peak_val)

# 添加峰值竖线（红色虚线，醒目且不干扰）
ax_A.axvline(x=peak_day, color='black', linestyle='-', linewidth=2.0, alpha=0.7, label='Peak Day')
# 标注每条曲线的峰值数值（位置在峰值右侧，避免遮挡）
for i, (peak_val, color) in enumerate(zip(peak_values_A, colors)):
    ax_A.text(peak_day + 0.85, peak_val + 0.002, f'{peak_val:.3f}',
              color=color, fontsize=14, fontweight='bold', va='bottom')

# A(t)面板其他样式
ax_A.set_xlabel('Time (day)', fontsize=14, fontweight='bold')
ax_A.set_ylabel(f'Susceptible Adults ($A(t)$)\n{y_label_unified}', fontsize=13, fontweight='bold')
ax_A.set_title('(A) Susceptible Adult Dynamics', fontsize=15, fontweight='bold', loc='left', pad=15)
ax_A.legend(loc='upper left', framealpha=0.92, ncol=1, frameon=True, edgecolor='black')
ax_A.set_xticks(t_full[::3])
ax_A.set_xlim(1, 31); ax_A.set_ylim(bottom=0)
ax_A.spines['top'].set_visible(False); ax_A.spines['right'].set_visible(False)
ax_A.grid(True, alpha=0.15, linestyle='-', linewidth=0.8)

# 4. 右侧2×2小面板（统一线型和颜色，添加峰值竖线）
def plot_small_panel(ax, compartment_idx, title):
    for compartments, color, ls, lw in zip(all_compartments, colors, linestyles, linewidths):
        ax.plot(t_full, compartments[compartment_idx], color=color, linestyle=ls, linewidth=lw, alpha=0.95)
    # 小面板添加相同峰值竖线（保持全局一致）
 #   ax.axvline(x=peak_day, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.set_xticks([1, 31])
    ax.set_xlabel('Time (day)', fontsize=11, fontweight='bold')
    ax.set_ylabel(y_label_unified, fontsize=11, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10, loc='left')
    ax.set_xlim(1, 31); ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.8)

# 调用绘制右侧面板
plot_small_panel(ax_E, 0, compartment_titles[0])
plot_small_panel(ax_Q, 1, compartment_titles[1])
plot_small_panel(ax_P, 2, compartment_titles[2])
plot_small_panel(ax_I, 4, compartment_titles[3])

# 5. 保存高清图（SCI标准）
plt.savefig('All_Compartments_Group_Figure_Optimized.svg', dpi=600, format='svg', bbox_inches='tight')
plt.savefig('All_Compartments_Group_Figure_Optimized.png', dpi=600, format='png', bbox_inches='tight')
plt.savefig('All_Compartments_Group_Figure_Optimized.eps', dpi=600, format='eps', bbox_inches='tight')

plt.show()

# 输出峰值数值（方便论文引用）
print("\n各I0梯度下易感成虫峰值（第19天）：")
for i0, peak_val in zip(I0_list, peak_values_A):
    print(f"I0 = {i0}: {peak_val:.4f} individuals/m²")
