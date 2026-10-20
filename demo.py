# 第一步：先造一份"假"电池数据练手（因为你可能还没拿到课题真实数据）
import pandas as pd
import numpy as np

n = 100
data = {
    'time': range(n),
    'voltage': np.linspace(3.2, 4.2, n),        # 电压 3.2~4.2V
    'current': np.random.uniform(-10, 10, n),   # 电流
    'temp': np.random.uniform(15, 45, n),       # 温度
    'soc': np.linspace(100, 20, n)              # SOC 从100%降到20%
}
df = pd.DataFrame(data)
df.to_csv('battery_demo.csv', index=False)
print('已生成 battery_demo.csv')

# 第二步：把数据读进来，打印看看
df2 = pd.read_csv('battery_demo.csv')
print(df2.head())      # 前 5 行
print('形状(行,列):', df2.shape)
print('列名:', list(df2.columns))
