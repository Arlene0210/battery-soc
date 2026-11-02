import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv(r'C:\Users\Arlene\Desktop\新疆项目\bms_xj01bms20250217_202609151657.csv')

X = df[['voltage', 'current', 'temper']]
y = df['soc']

# 划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 训练：线性回归
print("\n" + "=" * 50)
print("第 4 步：训练模型（线性回归）")
model = LinearRegression()
model.fit(X_train, y_train)
print("训练完成！")
print("学到的权重 w:", np.round(model.coef_, 3))
print("学到的截距 b:", round(model.intercept_, 3))
print("预测公式: soc ≈", np.round(model.coef_[0],3), "×电压 +",
      np.round(model.coef_[1],3), "×电流 +",
      np.round(model.coef_[2],3), "×温度 +", round(model.intercept_,2))

print("\n" + "=" * 50)
print("第 5 步：在测试集上预测")
y_pred = model.predict(X_test)
print("预测出", len(y_pred), "个值")
print("前 20 个对比（真值 vs 预测）:")
for i in range(20):
    print(f"  真实={y_test.iloc[i]:6.2f}  预测={y_pred[i]:6.2f}  误差={abs(y_test.iloc[i]-y_pred[i]):6.2f}")

print("\n" + "=" * 50)
print("第 6 步：评估模型")
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("MAE (平均绝对误差):", round(mae, 3), "→ 平均每个预测错这么多")
print("MSE (均方误差):", round(mse, 3))
print("RMSE (均方根误差):", round(rmse, 3), "→ 对大误差更敏感")
print("SOC 总范围: 0~100，MAE 越小越准")