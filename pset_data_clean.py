import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ===================== 1. 核心参数与实验背景配置 =====================
test_area = 10000  # 试验田面积：10000 m²（1公顷）
data_days = 31  # 观测天数：31天（Day 1-31）
data_idx = np.arange(data_days)  # 索引0-30（对应Day 1-31）
experiment_info = "1-hectare experimental field at Jilin Agricultural University"

# ===================== 2. 数据导入与预处理 =====================
male_data_31 = np.array(
    [2, 3, 12, 5, 4, 7, 9, 47, 38, 41, 20, 71, 94, 56, 64, 98, 117, 339, 243, 215, 182, 135, 140, 106, 75, 93, 35, 9, 6,
     1, 1])
total_data_31 = male_data_31 * 2  # 原始总数（个）
std_total_data_31 = total_data_31 / test_area  # 标准化总数（individuals/m²）
std_male_data_31 = male_data_31 / test_area  # 标准化雄性数（individuals/m²）

# 数据框构建
data_df = pd.DataFrame({
    'Day_Index': data_idx,
    'Actual_Day': data_idx + 1,
    'Male_Count_Raw': male_data_31,
    'Total_Count_Raw': total_data_31,
    'Male_Count_Standardized': std_male_data_31,
    'Total_Count_Standardized': std_total_data_31,
    'Is_Outlier': False,
    'Total_Count_Standardized_Cleaned': std_total_data_31,
    'Total_Count_Raw_Cleaned': total_data_31
})

# 异常值检测与清洗
Q1 = np.percentile(data_df['Total_Count_Standardized'], 25)
Q3 = np.percentile(data_df['Total_Count_Standardized'], 75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
data_df.loc[data_df['Total_Count_Standardized'] > upper_bound, 'Is_Outlier'] = True
outliers_df = data_df[data_df['Is_Outlier'] == True].copy()


def clean_outliers_standardized(data, outliers_mask):
    data_cleaned = data.copy()
    outliers_idx = np.where(outliers_mask)[0]
    for idx in outliers_idx:
        if idx == 0:
            replace_val = data[1:4].mean()
        elif idx == len(data) - 1:
            replace_val = data[-4:-1].mean()
        else:
            replace_val = data[max(0, idx - 1):min(len(data), idx + 2)].mean()
        data_cleaned[idx] = replace_val
    return data_cleaned


data_df['Total_Count_Standardized_Cleaned'] = clean_outliers_standardized(
    data=data_df['Total_Count_Standardized'].values,
    outliers_mask=data_df['Is_Outlier'].values
)
data_df['Total_Count_Raw_Cleaned'] = data_df['Total_Count_Standardized_Cleaned'] * test_area

# ===================== 3. 图表绘制（所有子图双Y轴+间距调整） =====================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.titlepad'] = 15  # 全局调整标题间距（默认12，增大到15）

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

# -------------------------- 子图1：箱线图（双Y轴） --------------------------
# 左Y轴：标准化密度（主轴）
ax1_left = ax1
box_plot = ax1_left.boxplot(
    data_df['Total_Count_Standardized'],
    patch_artist=True,
    boxprops=dict(facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=1.2),
    flierprops=dict(marker='o', markerfacecolor='red', markersize=8, markeredgecolor='darkred', linewidth=1),
    medianprops=dict(color='darkblue', linewidth=2.5),
    whiskerprops=dict(color='black', linewidth=1.2),
    capprops=dict(color='black', linewidth=1.2)
)
ax1_left.axhline(y=upper_bound, color='orange', linestyle='--', linewidth=1.5,
                 label=f'Upper Threshold: {upper_bound:.6f} ind/m²')
ax1_left.scatter(
    np.ones_like(data_df['Male_Count_Standardized']),
    data_df['Male_Count_Standardized'],
    color='gray',
    s=30,
    alpha=0.7,
    label='Std Male Count (Ref.)'
)
ax1_left.set_title('(A) Box Plot of Standardized Total Count', fontsize=12, loc='left', fontweight='bold',pad=10)
ax1_left.set_ylabel('Standardized Count (individuals/m²)', fontsize=11, color='black')
ax1_left.tick_params(axis='y', labelcolor='black')
ax1_left.legend(fontsize=9, loc='upper left', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
ax1_left.grid(False)

# 右Y轴：原始计数（对应标准化值×面积）
ax1_right = ax1_left.twinx()
ax1_right.set_ylabel('Raw Count (Individuals)', fontsize=11, color='darkred')
ax1_right.tick_params(axis='y', labelcolor='darkred')
# 右轴范围与左轴对应（原始值 = 标准化值 × 10000）
y_min_left, y_max_left = ax1_left.get_ylim()
ax1_right.set_ylim(y_min_left * test_area, y_max_left * test_area)

# -------------------------- 子图2：清洗前时间序列（双Y轴+增大标题间距） --------------------------
# 左Y轴：原始计数
ax2_left = ax2
line1 = ax2_left.plot(
    data_df['Actual_Day'],
    data_df['Total_Count_Raw'],
    color='blue',
    linestyle='-',
    linewidth=2,
    label='Total Count (Raw, ind.)'
)
line2 = ax2_left.plot(
    data_df['Actual_Day'],
    data_df['Male_Count_Raw'],
    color='gray',
    linestyle='-',
    linewidth=1.5,
    label='Male Count (Raw, ind.)'
)
if len(outliers_df) > 0:
    ax2_left.scatter(
        outliers_df['Actual_Day'],
        outliers_df['Total_Count_Raw'],
        color='red',
        s=100,
        zorder=5,
        label=f'Outlier (n={len(outliers_df)})'
    )
    for _, row in outliers_df.iterrows():
        ax2_left.annotate(
            f'Day {int(row["Actual_Day"])}\nRaw: {int(row["Total_Count_Raw"])} ind.\nStd: {row["Total_Count_Standardized"]:.6f} ind/m²',
            xy=(row['Actual_Day'], row['Total_Count_Raw']),
            xytext=(row['Actual_Day'] - 5, row['Total_Count_Raw'] - 80),
            fontsize=8,
            fontweight='bold',
            color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9)
        )
ax2_left.set_title('(B) Time Series of Raw & Standardized Counts (Before Cleaning)', fontsize=12, loc='left', fontweight='bold',
                   pad=10)  # 单独增大间距
ax2_left.set_xlabel('Day (1-31)', fontsize=11)
ax2_left.set_ylabel('Raw Count (Individuals)', fontsize=11, color='black')
ax2_left.tick_params(axis='y', labelcolor='black')
ax2_left.set_ylim(0, max(data_df['Total_Count_Raw'].max(), data_df['Male_Count_Raw'].max()) * 1.2)

# 右Y轴：标准化密度
ax2_right = ax2_left.twinx()
line3 = ax2_right.plot(
    data_df['Actual_Day'],
    data_df['Total_Count_Standardized'],
    color='darkblue',
    linestyle='--',
    linewidth=2,
    label='Total Count (Std, ind/m²)'
)
ax2_right.set_ylabel('Standardized Count (individuals/m²)', fontsize=11, color='darkblue')
ax2_right.tick_params(axis='y', labelcolor='darkblue')
ax2_right.set_ylim(0, max(data_df['Total_Count_Standardized'].max(), data_df['Male_Count_Standardized'].max()) * 1.2)

# 合并图例
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax2_left.legend(lines, labels, fontsize=8.5, loc='upper right', frameon=True, fancybox=True, shadow=False)
ax2_left.grid(False)

# -------------------------- 子图3：清洗前后对比（双Y轴） --------------------------
# 左Y轴：原始计数
ax3_left = ax3
line1 = ax3_left.plot(
    data_df['Actual_Day'],
    data_df['Total_Count_Raw'],
    color='blue',
    linestyle='--',
    linewidth=1.5,
    label='Before Cleaning (Raw, ind.)'
)
line2 = ax3_left.plot(
    data_df['Actual_Day'],
    data_df['Total_Count_Raw_Cleaned'],
    color='darkgreen',
    linestyle='-',
    linewidth=2.5,
    marker='o',
    markersize=4,
    label='After Cleaning (Raw, ind.)'
)
if len(outliers_df) > 0:
    ax3_left.scatter(
        outliers_df['Actual_Day'],
        outliers_df['Total_Count_Raw_Cleaned'],
        color='darkgreen',
        s=120,
        zorder=5,
        label='Cleaned Value (Raw, ind.)'
    )
ax3_left.set_title('(C) Comparison: Raw vs. Standardized Count (Before vs. After Cleaning)', fontsize=12,loc='left',
                   fontweight='bold',pad=10)
ax3_left.set_xlabel('Day (1-31)', fontsize=11)
ax3_left.set_ylabel('Raw Count (Individuals)', fontsize=11, color='black')
ax3_left.tick_params(axis='y', labelcolor='black')
ax3_left.set_ylim(0, max(data_df['Total_Count_Raw'].max(), data_df['Total_Count_Raw_Cleaned'].max()) * 1.2)

# 右Y轴：标准化密度
ax3_right = ax3_left.twinx()
line3 = ax3_right.plot(
    data_df['Actual_Day'],
    data_df['Total_Count_Standardized'],
    color='lightblue',
    linestyle='--',
    linewidth=1.5,
    label='Before Cleaning (Std, ind/m²)'
)
line4 = ax3_right.plot(
    data_df['Actual_Day'],
    data_df['Total_Count_Standardized_Cleaned'],
    color='darkblue',
    linestyle='-',
    linewidth=2.5,
    marker='x',
    markersize=4,
    label='After Cleaning (Std, ind/m²)'
)
ax3_right.set_ylabel('Standardized Count (individuals/m²)', fontsize=11, color='darkblue')
ax3_right.tick_params(axis='y', labelcolor='darkblue')
ax3_right.set_ylim(0, max(data_df['Total_Count_Standardized'].max(),
                          data_df['Total_Count_Standardized_Cleaned'].max()) * 1.2)

# 合并图例
lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax3_left.legend(lines, labels, fontsize=8, loc='upper right', frameon=True, fancybox=True, shadow=False)
ax3_left.grid(False)

# -------------------------- 子图4：最终清洗数据（双Y轴+增大标题间距） --------------------------
# 左Y轴：原始计数
ax4_left = ax4
line1 = ax4_left.plot(
    data_df['Actual_Day'],
    data_df['Total_Count_Raw_Cleaned'],
    color='darkgreen',
    linestyle='-',
    linewidth=3,
    marker='o',
    markersize=5,
    label='Cleaned Total Count (Raw, ind.)'
)
line2 = ax4_left.plot(
    data_df['Actual_Day'],
    data_df['Male_Count_Raw'],
    color='gray',
    linestyle='--',
    linewidth=2,
    marker='s',
    markersize=3,
    alpha=0.7,
    label='Male Count (Raw, ind.)'
)
if len(outliers_df) > 0:
    outlier_days = outliers_df['Actual_Day'].values
    outlier_raw_vals = outliers_df['Total_Count_Raw_Cleaned'].values
    outlier_std_vals = outliers_df['Total_Count_Standardized_Cleaned'].values

    ax4_left.scatter(
        outlier_days,
        outlier_raw_vals,
        color='red',
        marker='*',
        s=300,
        zorder=10,
        edgecolors='white',
        linewidth=2,
        label='Outlier Replacement'
    )

    for day, raw_val, std_val in zip(outlier_days, outlier_raw_vals, outlier_std_vals):
        ax4_left.annotate(
            f'Day {int(day)}\nRaw: {int(raw_val)} ind.\nStd: {std_val:.6f} ind/m²',
            xy=(day, raw_val),
            xytext=(day + 1, raw_val + 50),
            fontsize=8,
            fontweight='bold',
            color='darkred',
            ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9)
        )



ax4_left.set_title('(D) Final Cleaned Data: Raw vs. Standardized Count', fontsize=12, fontweight='bold',loc='left',
                   pad=10)  # 单独增大间距
ax4_left.set_xlabel('Day (1-31)', fontsize=11)
ax4_left.set_ylabel('Raw Count (Individuals)', fontsize=11, color='black')
ax4_left.tick_params(axis='y', labelcolor='black')
ax4_left.set_ylim(0, max(data_df['Total_Count_Raw_Cleaned'].max(), data_df['Male_Count_Raw'].max()) * 1.2)

# 右Y轴：标准化密度
ax4_right = ax4_left.twinx()
line3 = ax4_right.plot(
    data_df['Actual_Day'],
    data_df['Total_Count_Standardized_Cleaned'],
    color='darkblue',
    linestyle='-',
    linewidth=2,
    marker='x',
    markersize=4,
    label='Cleaned Total Count (Std, ind/m²)'
)
ax4_right.set_ylabel('Standardized Count (individuals/m²)', fontsize=11, color='darkblue')
ax4_right.tick_params(axis='y', labelcolor='darkblue')
ax4_right.set_ylim(0, max(data_df['Total_Count_Standardized_Cleaned'].max(),
                          data_df['Male_Count_Standardized'].max()) * 1.2)

# 合并图例
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax4_left.legend(
    lines, labels,
    fontsize=8.5,
    loc='upper right',
    frameon=True,
    fancybox=True,
    shadow=False,
    framealpha=0.9
)
ax4_left.grid(False)

# -------------------------- 全局布局调整 --------------------------
plt.tight_layout(pad=3.0)  # 增大子图间距（默认1.0，调整为3.0）
plt.savefig('Figure_2_Data_Cleaning_Standardization.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================== 4. 输出最终数据 =====================
final_data = data_df[
    ['Actual_Day', 'Total_Count_Raw_Cleaned', 'Total_Count_Standardized_Cleaned', 'Male_Count_Standardized']].round(6)
final_data.columns = ['Day', 'Raw_Total_Count_Cleaned (individuals)', 'Std_Total_Count_Cleaned (individuals/m²)',
                      'Std_Male_Count (individuals/m²)']
print("\n【Final Cleaned Data for Model Fitting】")
print(final_data.to_string(index=False))
final_data.to_csv('Cleaned_Pest_Data.csv', index=False)
print("\nData saved to 'Cleaned_Pest_Data.csv'")
