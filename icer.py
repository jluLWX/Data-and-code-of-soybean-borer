import numpy as np

# -------------------------- 1. 基础参数设定 --------------------------
# （1）各类策略的总费用（TC）
TC = {
    'NS': 0,
    'S1': 15000,  # 投放感染成虫
    'S2': 12000,  # 提升感染传播率β
    'S3': 10000,  # 提升病毒有效率σ
    'S4': 27000,  # S1+S2
    'S5': 37000  # S1+S2+S3
}

# （2）基于4.3节数值模拟结果，提取各类策略的种群峰值差值（ΔA, ΔE, ΔQ）
# 格式：{策略: [ΔA, ΔE, ΔQ]}（单位：individuals/m²）
delta_pop = {
    'NS': [0, 0, 0],
    'S1': [0.04, 0.08, 0.06],  # S1策略下的种群减少量（示例数据，可替换为实际模拟结果）
    'S2': [0.03, 0.06, 0.04],  # S2策略下的种群减少量
    'S3': [0.025, 0.05, 0.035],  # S3策略下的种群减少量
    'S4': [0.075, 0.14, 0.105],  # S4策略下的种群减少量
    'S5': [0.095, 0.18, 0.135],  # S5策略下的种群减少量
}

# （3）种群权重
weights = [0.5, 0.2, 0.3]  # A的权重0.5，E的权重0.2，Q的权重0.3

# -------------------------- 2. 计算总效益（TB） --------------------------
TB = {}
for strategy in delta_pop.keys():
    delta_A, delta_E, delta_Q = delta_pop[strategy]
    tb = weights[0] * delta_A + weights[1] * delta_E + weights[2] * delta_Q
    TB[strategy] = tb * 100000  # 放大系数，使TB数值更易读（基于1公顷=10000m²换算）

# -------------------------- 3. 计算成本效益比（CER） --------------------------
CER = {}
for strategy in TC.keys():
    if TB[strategy] == 0:
        CER[strategy] = '-'
    else:
        cer = TC[strategy] / TB[strategy]
        CER[strategy] = round(cer, 4)

# -------------------------- 4. 计算增量成本效益比（ICER） --------------------------
# 策略优先级：NS → S3 → S2 → S1 → S4 → S5
strategies_order = ['NS', 'S3', 'S2', 'S1', 'S4', 'S5']
ICER = []
for i in range(1, len(strategies_order)):
    strategy1 = strategies_order[i]
    strategy2 = strategies_order[i - 1]
    delta_TC = TC[strategy1] - TC[strategy2]
    delta_TB = TB[strategy1] - TB[strategy2]
    if delta_TB == 0:
        icer = '-'
    else:
        icer = round(delta_TC / delta_TB, 4)
    ICER.append({
        '对比组合': f'{strategy1} vs {strategy2}',
        '增量成本（ΔTC）': delta_TC,
        '增量效益（ΔTB）': round(delta_TB, 2),
        '增量成本效益比（ICER）': icer
    })

# -------------------------- 5. 输出结果 --------------------------
print("=" * 50)
print("基础成本效益比（CER）结果：")
print("-" * 50)
print(f"{'策略':<10}{'总费用（TC）':<12}{'总效益（TB）':<15}{'成本效益比（CER）':<15}")
print("-" * 50)
for strategy in strategies_order:
    print(f"{strategy:<10}{TC[strategy]:<12}{TB[strategy]:<15.2f}{CER[strategy]:<15}")

print("\n" + "=" * 50)
print("增量成本效益比（ICER）结果：")
print("-" * 50)
print(f"{'对比组合':<20}{'增量成本（ΔTC）':<15}{'增量效益（ΔTB）':<15}{'增量成本效益比（ICER）':<15}")
print("-" * 50)
for item in ICER:
    print(
        f"{item['对比组合']:<20}{item['增量成本（ΔTC）']:<15}{item['增量效益（ΔTB）']:<15.2f}{item['增量成本效益比（ICER）']:<15}")

# -------------------------- 6. 结果保存为Excel（可选） --------------------------
try:
    import pandas as pd

    # 基础CER表
    cer_data = {
        '防控策略': strategies_order,
        '总费用（TC）': [TC[s] for s in strategies_order],
        '总效益（TB）': [round(TB[s], 2) for s in strategies_order],
        '成本效益比（CER）': [CER[s] for s in strategies_order]
    }
    cer_df = pd.DataFrame(cer_data)

    # ICER表
    icer_df = pd.DataFrame(ICER)

    # 保存到Excel
    with pd.ExcelWriter('成本效益分析结果.xlsx', engine='openpyxl') as writer:
        cer_df.to_excel(writer, sheet_name='基础成本效益比', index=False)
        icer_df.to_excel(writer, sheet_name='增量成本效益比', index=False)
    print("\n结果已保存到'成本效益分析结果.xlsx'")
except ImportError:
    print("\n未安装pandas，跳过Excel保存")
