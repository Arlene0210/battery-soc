import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

df = pd.read_csv(r'C:\Users\Arlene\Desktop\新疆项目\bms_xj01bms20250217_202609151657.csv')
X = df[['voltage','current','temper']]
y = df['soc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    '随机森林(bagging)':   RandomForestRegressor(n_estimators=100, random_state=42),
    'GBDT(boosting)':      GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost(boosting)':   XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
}

print(f"{'模型':22s} {'MAE':>10s}")
print("-" * 34)
for name, model in models.items():
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"{name:22s} {mae:10.3f}")
