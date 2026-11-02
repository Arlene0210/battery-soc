import pandas as pd

df = pd.read_csv(r'C:\Users\Arlene\Desktop\新疆项目\bms_xj01bms20250217_202609151657.csv')

X = df[['voltage', 'current', 'temper']]
y = df['soc']

# 第1步：生成打乱的索引
import numpy as np
idx = np.arange(len(X))
np.random.shuffle(idx)          # 打乱，防止顺序偏差

# 第2步：按 8:2 切分
split = int(0.8 * len(X))

X_train = X.iloc[idx[:split]]   # 前80% → 训练
X_test  = X.iloc[idx[split:]]   # 后20% → 测试
y_train = y.iloc[idx[:split]]
y_test  = y.iloc[idx[split:]]

print("训练集:", X_train.shape, y_train.shape)   # (20472,3) (20472,)
print("测试集:", X_test.shape,  y_test.shape)    # (5118,3)  (5118,)
