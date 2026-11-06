import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv(r'C:\Users\Arlene\Desktop\新疆项目\bms_xj01bms20250217_202609151657.csv')
X = df[['voltage','current','temper']]
y = df['soc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义要搜的「网格」
param_grid = {
    'n_estimators': [50, 100, 200],   # 树的数量
    'max_depth':    [5, 10, None],     # 树的深度
}

# 网格搜索 + 3 折交叉验证
gs = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',   # 用 MAE 打分（取负，越大越好）
    n_jobs=-1,                            # 并行加速
)
gs.fit(X_train, y_train)

print("最优参数:", gs.best_params_)
print("最优交叉验证 MAE:", round(-gs.best_score_, 3))

# 用最优参数在测试集上最终评估
best = gs.best_estimator_
test_mae = mean_absolute_error(y_test, best.predict(X_test))
print("最优模型在测试集 MAE:", round(test_mae, 3))
