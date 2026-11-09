import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

models = ['线性回归', 'GBDT', 'XGBoost', '随机森林']
mae = [11.93, 2.58, 1.26, 1.08]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, mae, color=['#d9534f', '#f0ad4e', '#5bc0de', '#5cb85c'])
plt.ylabel('MAE（越低越好）')
plt.title('V1 各模型 SOC 预测误差对比')
for bar, v in zip(bars, mae):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{v}', ha='center', va='bottom', fontsize=11)
plt.savefig('model_compare.png', dpi=150, bbox_inches='tight')
plt.show()
