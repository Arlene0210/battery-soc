import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False     # 修复负号显示


df = pd.read_csv(r'C:\Users\Arlene\Desktop\新疆项目\bms_xj01bms20250217_202609151657.csv')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(df['soc'], bins=50, color='crimson')
axes[0].set_title('SOC 分布')
axes[1].hist(df['voltage'], bins=50, color='seagreen')
axes[1].set_title('电压分布')
axes[2].hist(df['temper'], bins=50, color='steelblue')
axes[2].set_title('温度分布')
plt.tight_layout()
plt.savefig('hist.png', dpi=150, bbox_inches='tight')
plt.show()
