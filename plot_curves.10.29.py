import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False     # 修复负号显示


df = pd.read_csv(r'C:\Users\Arlene\Desktop\新疆项目\bms_xj01bms20250217_202609151657.csv')
df['ts'] = pd.to_datetime(df['ts'])

# 子图：三行分别画 SOC / 电压 / 温度
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
axes[0].plot(df['ts'], df['soc'], lw=0.5, color='crimson')
axes[0].set_ylabel('SOC (%)')
axes[1].plot(df['ts'], df['voltage'], lw=0.5, color='seagreen')
axes[1].set_ylabel('电压 (V)')
axes[2].plot(df['ts'], df['temper'], lw=0.5, color='steelblue')
axes[2].set_ylabel('温度 (℃)')
plt.tight_layout()
plt.savefig('curves.png', dpi=150, bbox_inches='tight')
plt.show()
