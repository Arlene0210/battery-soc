import numpy as np
data = np.genfromtxt('battery_demo.csv', delimiter=',', skip_header=1)
names = ['time', 'voltage', 'current', 'temp', 'soc']

for i, name in enumerate(names):
    c = data[:, i]
    print(f"{name:8s} 均值{c.mean():8.2f} 标准差{c.std():8.2f} "
          f"最小{c.min():8.2f} 最大{c.max():8.2f} 中位数{np.median(c):8.2f}")

# 特征相关性：电压/电流/温度/SOC 两两之间相关程度（-1~1）
corr = np.corrcoef(data[:, 1:].T)
print("\n相关系数矩阵(voltage,current,temp,soc):\n", np.round(corr, 2))
