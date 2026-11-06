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




# 关键：同时在「训练集」和「测试集」上评估
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae  = mean_absolute_error(y_test, y_test_pred)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse  = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("=" * 50)
print(f"{'':8s} {'MAE':>10s} {'RMSE':>10s}")
print(f"{'训练集':<8s} {train_mae:10.3f} {train_rmse:10.3f}")
print(f"{'测试集':<8s} {test_mae:10.3f} {test_rmse:10.3f}")


# Ridge正则化对比

print("\n" + "=" * 50)
print("防过拟合实验 1：Ridge 正则化")
from sklearn.linear_model import Ridge

for alpha in [0.1, 1.0, 10.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    r_train = mean_absolute_error(y_train, ridge.predict(X_train))
    r_test  = mean_absolute_error(y_test,  ridge.predict(X_test))
    print(f"alpha={alpha:<6} 训练MAE={r_train:8.3f}  测试MAE={r_test:8.3f}")


# 五折交叉验证

print("\n" + "=" * 50)
print("防过拟合实验 2：5 折交叉验证")
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
print("5 折各自 MAE:", np.round(-scores, 3))
print("平均 MAE:", round(-scores.mean(), 3), "±", round(scores.std(), 3))



# ============ 任务3：baseline 调优 ============
print("\n" + "=" * 50)
print("实验1：换特征组合")
from sklearn.ensemble import RandomForestRegressor

feature_sets = {
    'A: 仅电压':           ['voltage'],
    'B: 电压+电流+温度':     ['voltage','current','temper'],
    'C: 加容量/能量':        ['voltage','current','temper','remain_capacity','soe'],
}

for name, cols in feature_sets.items():
    X = df[cols].dropna()
    y = df['soc'].loc[X.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    m = LinearRegression().fit(X_train, y_train)
    mae = mean_absolute_error(y_test, m.predict(X_test))
    print(f"{name:20s} → MAE = {mae:.3f}")

print("\n" + "=" * 50)
print("实验2：换随机森林（非线性）")
X = df[['voltage','current','temper']]
y = df['soc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("随机森林 MAE =", round(mean_absolute_error(y_test, rf.predict(X_test)), 3))


# 特征重要性
importances = rf.feature_importances_
for name, imp in zip(['voltage','current','temper'], importances):
    print(f"{name:10s} 重要性 = {imp:.3f}")

for n in [10, 50, 100, 200]:
    rf = RandomForestRegressor(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, rf.predict(X_test))
    print(f"树数量={n:4d} → MAE = {mae:.3f}")
