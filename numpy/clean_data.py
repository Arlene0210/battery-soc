import numpy as np

# 读假数据
data = np.genfromtxt('battery_demo.csv', delimiter=',', skip_header=1)

# ===== 第 1 步：故意弄脏（真实数据里这些都会出现）=====
dirty = data.copy()
dirty[5, 1] = np.nan      # 第5行电压设成缺失
dirty[10, 3] = np.nan     # 第10行温度设成缺失
dirty[20, 1] = 99.0       # 电压设成异常值 99V（真电池不可能）
dirty[30, 3] = -999.0     # 温度设成异常值 -999

print("弄脏后：缺失值总数 =", np.isnan(dirty).sum())

# ===== 第 2 步：清洗 =====
# ① 缺失值：用该列均值填充
for col in range(dirty.shape[1]):
    col_data = dirty[:, col]
    nan_mask = np.isnan(col_data)
    if nan_mask.any():
        col_data[nan_mask] = np.nanmean(col_data)   # 用忽略NaN的均值填

# ② 异常值：温度超过 [-50, 80] 的当成错误，截断到边界
temp = dirty[:, 3]
temp[temp < -50] = -50
temp[temp > 80] = 80
dirty[:, 3] = temp

# ③ 电压超过 [2.5, 4.5] 的当成错误，直接剔除那一行
volt = dirty[:, 1]
valid = (volt >= 2.5) & (volt <= 4.5)
cleaned = dirty[valid]

print("清洗后：缺失值总数 =", np.isnan(cleaned).sum(), "| 剩余行数 =", len(cleaned))
