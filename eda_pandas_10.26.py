import pandas as pd
import numpy as np

# ===== 第 1 步：读文件 =====
df = pd.read_csv(r'C:\Users\Arlene\Desktop\新疆项目\bms_xj01bms20250217_202609151657.csv')

# ===== 第 2 步：看长相（可留可删，不影响）=====
print("形状:", df.shape)
print(df.info())

# ===== 第 3 步：找脏数据（用真实列名）=====
missing = df.isnull().sum()
print("\n缺失值最多的前 10 列:\n", missing.sort_values(ascending=False).head(10))

core = ['ts', 'voltage', 'current', 'temper', 'soc', 'soh',
        'remain_capacity', 'cumulative_charge', 'cumulative_discharge']
df_core = df[core].copy()

print("\n电压 > 1000 的行数(疑似65535溢出):", (df['voltage'] > 1000).sum())
print("温度 > 100 的行数(疑似异常):", (df['temper'] > 100).sum())
print("温度 < -50 的行数(疑似异常):", (df['temper'] < -50).sum())

print("\nSOC 唯一值数量:", df['soc'].nunique())
print(df['soc'].value_counts().head())
print("\nSOC 均值/最大/最小:", df['soc'].mean(), df['soc'].max(), df['soc'].min())

df['ts'] = pd.to_datetime(df['ts'])
print("\n数据时间范围:", df['ts'].min(), "→", df['ts'].max())


core = ['ts', 'voltage', 'current', 'temper', 'soc', 'soh',
        'remain_capacity', 'cycle_time_module',
        'cumulative_charge', 'cumulative_discharge', 'voltage_std', 'soe']
df_core = df[core].copy()
df_core['ts'] = pd.to_datetime(df_core['ts'])
print(df_core.describe())   # 这次不会再有 NaN 了




# 看 2025-11 前后 soc_flag 的取值分布，但数据为空
seg2 = df[(df['ts'] >= '2025-11-01') & (df['ts'] <= '2026-01-15')]
print(seg2['soc_flag'].value_counts())


# 方案 A：看跳变时段 soe（有 25480 条数据）是否也同步跳
# soe 是能量状态，如果 BMS 重置，它应该和 soc 一起跳
seg2 = df[(df['ts'] >= '2025-11-20') & (df['ts'] <= '2025-12-10')]
print(seg2[['ts','soc','soe','cycle_time_module']].iloc[::20])  # 每20行抽1行看趋势

# 方案 B：跳变前后 cycle_time_module（循环次数）有没有变化
print("\ncycle_time_module 全列唯一值数:", df['cycle_time_module'].nunique())
print(df['cycle_time_module'].value_counts().head())




# 画 SOC 和电流随时间变化，直观看到「卡在82.6 + 电流大部分为0」
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(df_core['ts'], df_core['soc'], lw=0.6)
axes[0].set_ylabel('SOC (%)')
axes[1].plot(df_core['ts'], df_core['current'], lw=0.6)
axes[1].set_ylabel('Current (A)')
plt.tight_layout()
plt.savefig('soc_current.png', dpi=150, bbox_inches='tight')
plt.show()