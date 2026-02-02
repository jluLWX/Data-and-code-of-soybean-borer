import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import time
from sklearn.metrics import r2_score, mean_squared_error

# %% 1. Data Preparation (unchanged)
A_obs = np.array([  # Adult density (individuals/m²)
    0.0004, 0.0006, 0.0024, 0.001, 0.0008, 0.0014, 0.0018,  # Day 1-7
    0.0094, 0.0076, 0.0082, 0.004, 0.0142, 0.0188, 0.0112,  # Day 8-14
    0.0128, 0.0196, 0.0234, 0.0466, 0.0486,  # Day 15-19
    0.043, 0.0364, 0.027, 0.028, 0.0212, 0.015, 0.0186,  # Day 20-26
    0.007, 0.0018, 0.0012, 0.0002, 0.0002  # Day 27-31
])
t_full = np.arange(1, 32)
t1, t2, t3, t4 = 7, 14, 19, 27  # Five-segment nodes: 1-7,7-14,14-19,19-27,27-31

# %% 2. Fixed Parameters + Model Definition (unchanged)
α1 = 0.25;
α2 = 0.08;
α3 = 0.25;
μ1 = 0.005;
μ2 = 0.015;
μ3 = 0.01;
μ4 = 0.15;
b = 0.6;
K = 0.1113;

def eqpa_5seg_model(t, y, params):
    E, Q, P, A = y
    base_γ, base_α4 = params
    base_γ = max(0.1, base_γ);
    base_α4 = max(0.1, base_α4)
    γ_coeff = [0.01, 2.0, 7.8, 0.3, 0.001];
    α4_coeff = [1.0, 0.2, 0.05, 2.0, 30.0]
    if t <= t1:
        γ = base_γ * γ_coeff[0];
        α4 = base_α4 * α4_coeff[0]
    elif t1 < t <= t2:
        γ = base_γ * γ_coeff[1];
        α4 = base_α4 * α4_coeff[1]
    elif t2 < t <= t3:
        γ = base_γ * γ_coeff[2];
        α4 = base_α4 * α4_coeff[2]
    elif t3 < t <= t4:
        γ = base_γ * γ_coeff[3];
        α4 = base_α4 * α4_coeff[3]
    else:
        γ = base_γ * γ_coeff[4];
        α4 = base_α4 * α4_coeff[4]
    dE_dt = b * A - (α1 + μ1) * E;
    dQ_dt = α1 * E - (α2 + μ2) * Q;
    dP_dt = α2 * Q - (α3 + μ3) * P
    dA_dt = γ * A * (1 - A / K) + α3 * P - (α4 + μ4) * A
    return [dE_dt, dQ_dt, dP_dt, dA_dt]

def residuals(params):
    A0 = A_obs[0];
    E0 = A0 * 10;
    Q0 = A0 * 8;
    P0 = A0 * 5;
    y0 = [E0, Q0, P0, A0]
    sol = solve_ivp(
        fun=lambda t, y: eqpa_5seg_model(t, y, params),
        t_span=(1, 31), y0=y0, t_eval=t_full, method='RK45',
        max_step=0.05, atol=1e-12, rtol=1e-10
    )
    A_pred = interp1d(sol.t, sol.y[3], kind='linear')(t_full)
    E_pred = interp1d(sol.t, sol.y[0], kind='linear')(t_full)
    Q_pred = interp1d(sol.t, sol.y[1], kind='linear')(t_full)
    P_pred = interp1d(sol.t, sol.y[2], kind='linear')(t_full)

    data_error = A_pred - A_obs
    weight = np.ones(31)
    weight[t_full <= t1] = 5.0;
    weight[(t_full > t1) & (t_full <= t2)] = 10.0;
    weight[(t_full > t2) & (t_full <= t3)] = 15.0;
    weight[(t_full > t3) & (t_full <= t4)] = 8.0;
    weight[t_full > t4] = 90.0;
    weighted_data_error = data_error * weight * 1000

    fifth_pred = A_pred[t_full >= t4]
    linear_fit = np.polyfit(np.arange(len(fifth_pred)), fifth_pred, 1)
    linear_pred = np.polyval(linear_fit, np.arange(len(fifth_pred)))
    linear_error = fifth_pred - linear_pred
    linear_error_weighted = linear_error * 500

    dE_dt_num = np.gradient(E_pred, t_full);
    dQ_dt_num = np.gradient(Q_pred, t_full)
    dP_dt_num = np.gradient(P_pred, t_full);
    dA_dt_num = np.gradient(A_pred, t_full)
    dE_dt_model, dQ_dt_model, dP_dt_model, dA_dt_model = zip(
        *[eqpa_5seg_model(t, [E, Q, P, A], params) for t, E, Q, P, A in zip(t_full, E_pred, Q_pred, P_pred, A_pred)]
    )
    physics_error = np.concatenate(
        [dE_dt_num - np.array(dE_dt_model), dQ_dt_num - np.array(dQ_dt_model), dP_dt_num - np.array(dP_dt_model),
         dA_dt_num - np.array(dA_dt_model)])

    return np.concatenate([weighted_data_error, linear_error_weighted, physics_error * 10])

# Optimization
start_time = time.time()
initial_params = [0.6, 0.35]
fitted_params, cov, info, msg, ier = leastsq(residuals, initial_params, maxfev=150000, full_output=True)
print(f"Optimization Status: {ier} (1-4 = successful)")
print(f"\n【Fixed Parameters】")
print(f"  Transition Rates: α1={α1:.3f}, α2={α2:.3f}, α3={α3:.3f}")
print(f"  Natural Mortality Rates: μ1={μ1:.3f}, μ2={μ2:.3f}, μ3={μ3:.3f}, μ4={μ4:.3f}")
print(f"  Oviposition Rate b={b:.3f}, Carrying Capacity K={K:.3f}")
print(f"  Five-segment Nodes: Day 1-7, Day 7-14, Day 14-19, Day 19-27, Day 27-31")
print(f"\n【Estimated Parameters (2 total)】")
print(f"  Base Growth Rate base_γ = {fitted_params[0]:.4f}")
print(f"  Base Adult Loss Rate base_α4 = {fitted_params[1]:.4f}")
print(f"  Running Time: {time.time() - start_time:.2f} seconds")

# %% 3. Calculate Fitting Results + Time-varying Parameters (unchanged)
A0 = A_obs[0];
y0 = [A0 * 10, A0 * 8, A0 * 5, A0]
sol_fit = solve_ivp(
    fun=lambda t, y: eqpa_5seg_model(t, y, fitted_params),
    t_span=(1, 31), y0=y0, t_eval=t_full, method='RK45', max_step=0.05
)
E_fit = interp1d(sol_fit.t, sol_fit.y[0], kind='linear')(t_full)
Q_fit = interp1d(sol_fit.t, sol_fit.y[1], kind='linear')(t_full)
P_fit = interp1d(sol_fit.t, sol_fit.y[2], kind='linear')(t_full)
A_fit = interp1d(sol_fit.t, sol_fit.y[3], kind='linear')(t_full)

# Gaussian Smoothing
A_fit_smooth = gaussian_filter1d(A_fit, sigma=0.9)
E_fit_smooth = gaussian_filter1d(E_fit, sigma=0.9)
Q_fit_smooth = gaussian_filter1d(Q_fit, sigma=0.9)
P_fit_smooth = gaussian_filter1d(P_fit, sigma=0.9)

# Extract Time-varying Parameters
base_γ, base_α4 = fitted_params
γ_coeff = [0.01, 5.0, 5.0, 0.3, 0.001];
α4_coeff = [1.0, 0.2, 0.05, 2.0, 30]
γ_vals = np.zeros(31);
α4_vals = np.zeros(31)
for i, t in enumerate(t_full):
    if t <= t1:
        γ_vals[i] = base_γ * γ_coeff[0];
        α4_vals[i] = base_α4 * α4_coeff[0]
    elif t <= t2:
        γ_vals[i] = base_γ * γ_coeff[1];
        α4_vals[i] = base_α4 * α4_coeff[1]
    elif t <= t3:
        γ_vals[i] = base_γ * γ_coeff[2];
        α4_vals[i] = base_α4 * α4_coeff[2]
    elif t <= t4:
        γ_vals[i] = base_γ * γ_coeff[3];
        α4_vals[i] = base_α4 * α4_coeff[3]
    else:
        γ_vals[i] = base_γ * γ_coeff[4];
        α4_vals[i] = base_α4 * α4_coeff[4]
γ_vals_smooth = gaussian_filter1d(γ_vals, sigma=0.1)
α4_vals_smooth = gaussian_filter1d(α4_vals, sigma=0.1)

# %% 4. Fitting Performance Metrics (unchanged)
print("\n" + "=" * 120)
print("【Fitting Performance Metrics by Time Segment (Including RMSE)】")
print("=" * 120)
stages = [
    ("Day 1-7 (Stable Phase)", (t_full >= 1) & (t_full <= t1)),
    ("Day 7-14 (Rapid Growth Phase)", (t_full > t1) & (t_full <= t2)),
    ("Day 14-19 (Burst Growth Phase)", (t_full > t2) & (t_full <= t3)),
    ("Day 19-27 (Slow Decline Phase)", (t_full > t3) & (t_full <= t4)),
    ("Day 27-31 (Rapid Decline Phase)", (t_full > t4) & (t_full <= 31)),
    ("Overall (Day 1-31)", (t_full >= 1) & (t_full <= 31))
]
print(f"{'Time Segment':<25} {'R² (Goodness of Fit)':<20} {'MSE (Mean Squared Error)':<25} {'RMSE (Root MSE)':<25} {'Number of Observations':<20}")
print("-" * 120)
for stage_name, mask in stages:
    obs = A_obs[mask]
    fit = A_fit_smooth[mask]
    r2 = r2_score(obs, fit)
    mse = mean_squared_error(obs, fit)
    rmse = np.sqrt(mse)
    print(f"{stage_name:<25} {r2:.4f} {'':<16} {mse:.8f} {'':<21} {rmse:.6f} {'':<21} {len(obs):<20}")
print("=" * 120)
print("Notes: 1. R² closer to 1 = better fit; 2. MSE/RMSE closer to 0 = smaller error; 3. RMSE unit = individuals/m²")

# %% 5. Visualization (Core Modifications)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.fancybox'] = True
plt.rcParams['legend.shadow'] = False

fig = plt.figure(figsize=(16, 12))

# -------------------------- Subplot (A): Adult Density Fitting --------------------------
# 核心修改1：标题改为大写（A），颜色匹配Figure3
ax1 = plt.subplot(2, 2, (1, 2))
ax1.scatter(t_full, A_obs, color='red', s=80, marker='o', edgecolors='darkred', linewidth=1.5, label='Observed Data')
ax1.plot(t_full, A_fit_smooth, color='darkblue', lw=3.5, label='5-segment Fitted Values')
# 阶段背景色（与Figure3完全一致）
ax1.axvspan(1, t1, alpha=0.15, color='gray', label='Day 1-7 (Stable)')
ax1.axvspan(t1, t2, alpha=0.15, color='lightblue', label='Day 7-14 (Rapid Growth)')
ax1.axvspan(t2, t3, alpha=0.15, color='orange', label='Day 14-19 (Burst Growth)')
ax1.axvspan(t3, t4, alpha=0.15, color='purple', label='Day 19-27 (Slow Decline)')
ax1.axvspan(t4, 31, alpha=0.15, color='lightcoral', label='Day 27-31 (Linear Decline)')
ax1.set_xlabel('Time (Day)', fontsize=14)
ax1.set_ylabel('I(t) Density (individuals/m²)', fontsize=14)
# 子图标题大写（A）
ax1.set_title('(A) Soybean Pod Borer Model Fitting Results', fontsize=18, fontweight='bold',loc='left', pad=15)
ax1.legend(fontsize=12, loc='upper left', framealpha=0.9)
ax1.set_xticks(t_full[::2])
ax1.grid(False)

# -------------------------- Subplot (B): Developmental Stages --------------------------
# 核心修改2：标题大写（B）、颜色匹配Figure3、差异化线型（适配黑白打印）
ax2 = plt.subplot(2, 2, 3)
# 颜色完全匹配Figure3，线型差异化（实线、虚线、点划线、点线）
ax2.plot(t_full, E_fit_smooth, color='green', lw=3.5, linestyle='-', label='Eggs (E)')  # 实线
ax2.plot(t_full, Q_fit_smooth, color='gold', lw=3.5, linestyle='--', label='Larvae (Q)')  # 虚线
ax2.plot(t_full, P_fit_smooth, color='magenta', lw=3.5, linestyle='-.', label='Pupae (P)')  # 点划线
ax2.plot(t_full, A_fit_smooth, color='darkblue', lw=3.5, linestyle=':', label='Adults (A)')  # 点线
# 分段节点线
ax2.axvline(t1, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax2.axvline(t2, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax2.axvline(t3, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax2.axvline(t4, color='black', linestyle='--', alpha=0.5, linewidth=1, label='Segment Nodes')
ax2.set_xlabel('Time (Day)', fontsize=12)
ax2.set_ylabel('Population Density (individuals/m²)', fontsize=12)
# 子图标题大写（B）
ax2.set_title('(B) Dynamics of Developmental Stages', fontsize=14, fontweight='bold', loc='left',pad=15)
ax2.legend(fontsize=10, framealpha=0.9)
ax2.set_xticks(t_full[::4])
ax2.grid(False)

# -------------------------- Subplot (C): Time-varying Parameters --------------------------
# 核心修改3：标题大写（C）、颜色匹配Figure3、差异化线型
ax3 = plt.subplot(2, 2, 4)
ax3_twin = ax3.twinx()
# 颜色匹配Figure3，线型差异化
line1 = ax3.plot(t_full, γ_vals_smooth, 'red', lw=3, linestyle='-', label='Intrinsic Growth Rate  ($γ$)')  # 实线
line2 = ax3_twin.plot(t_full, α4_vals_smooth, 'darkblue', lw=3, linestyle='--', label='Adult Oviposition Rate ($α_4$)')  # 虚线
# 坐标轴标签颜色匹配
ax3.set_ylabel('Intrinsic Growth Rate ($γ$)', fontsize=12, color='red')
ax3.tick_params(axis='y', labelcolor='red')
ax3_twin.set_ylabel('Adult Oviposition Rate $α_4$', fontsize=12, color='darkblue')
ax3_twin.tick_params(axis='y', labelcolor='darkblue')
# 阶段背景色（与Subplot A一致）
ax3.axvspan(1, t1, alpha=0.15, color='gray')
ax3.axvspan(t1, t2, alpha=0.15, color='lightblue')
ax3.axvspan(t2, t3, alpha=0.15, color='orange')
ax3.axvspan(t3, t4, alpha=0.15, color='purple')
ax3.axvspan(t4, 31, alpha=0.15, color='lightcoral')
# 图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, fontsize=10, loc='upper left', framealpha=0.9)
ax3.set_xlabel('Time (Day)', fontsize=12)
# 子图标题大写（C）
ax3.set_title('(C) Dynamics of Time-varying Parameters', fontsize=14, fontweight='bold',loc='left', pad=15)
ax3.set_xticks(t_full[::4])
ax3.grid(False)

plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
plt.savefig('Figure_4_EQPA_Model_Fitting_Final.png', dpi=300, bbox_inches='tight')
plt.show()