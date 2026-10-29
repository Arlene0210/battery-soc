import pandas as pd

df = pd.read_csv(r'C:\Users\Arlene\Desktop\新疆项目\bms_xj01bms20250217_202609151657.csv')
df['ts'] = pd.to_datetime(df['ts'])

# ===== 分组统计 1：SOC 按月份看分布 =====
df['月份'] = df['ts'].dt.to_period('M')
print("每月 SOC 均值:")
print(df.groupby('月份')['soc'].mean().round(2))

# ===== 分组统计 2：电流按「充/放/待机」三种状态占比 =====
def 状态(x):
    if x > 5:   return '充电'
    elif x < -5: return '放电'
    else:        return '待机'
df['状态'] = df['current'].apply(状态)
print("\n电流状态分布:")
print(df['状态'].value_counts())

# ===== 数据透视：每种状态下的 SOC / 温度平均值 =====
pivot = df.pivot_table(index='状态',
                       values=['soc', 'temper'],
                       aggfunc='mean')
print("\n透视表(各状态平均SOC/温度):\n", pivot.round(2))
