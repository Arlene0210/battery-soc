import pandas as pd

df = pd.read_csv(r'C:\Users\Arlene\Desktop\新疆项目\bms_xj01bms20250217_202609151657.csv')

# 第1步：定标签 y —— 要预测什么？今天先演示「SOC」
y = df['soc']

# 第2步：定特征 X —— 用什么去预测？挑有意义的列
# 结合你 EDA 的结论，核心特征就这几个：
X = df[['voltage', 'current', 'temper']]

print("特征 X 形状:", X.shape)   # (25590, 3) → 25590个样本，3个特征
print("标签 y 形状:", y.shape)   # (25590,)    → 25590个答案

# 第3步：看一眼 X 和 y 长什么样（这就是模型「学习」的输入输出）
print("\nX 前5行:\n", X.head())
print("\ny 前5个:\n", y.head())
