# -*- coding: utf-8 -*-
"""
EDA：探索真实 BMS 电池数据
作用：用 pandas 把新疆项目的 BMS 数据「体检」一遍——
      看结构、算统计、找缺失、抓脏数据、画图。
运行：在 VSCode 里点右上角 ▶ 运行本文件。
"""
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 第 1 步：读入数据 ----------
# 路径里有中文和空格没关系，前面加 r 表示"原始字符串"
path = r"C:\Users\Arlene\Desktop\新疆项目\bms_xj01bms20250217_202609151657.csv"
df = pd.read_csv(path)
print("【1】数据读进来了")
print("形状(行, 列):", df.shape)          # 应该是 (25590, 79)
print("列名(前20个):")
print(list(df.columns[:20]))

# ---------- 第 2 步：看前几行 + 每列类型 ----------
print("\n【2】前 3 行数据")
print(df.head(3))
print("\n各列数据类型(只看前10列):")
print(df.dtypes.head(10))

# ---------- 第 3 步：时间列处理 ----------
print("\n【3】时间范围")
df["ts"] = pd.to_datetime(df["ts"])       # 把字符串时间变成真正的时间类型
print("开始:", df["ts"].min())
print("结束:", df["ts"].max())
# 相邻两条的间隔（秒），取中位数
gap = df["ts"].diff().dt.total_seconds().median()
print("采样间隔约:", gap / 60, "分钟")

# ---------- 第 4 步：关键列的统计描述 ----------
print("\n【4】关键列统计(describe 给出 均值/方差/分位数/最大最小)")
key_cols = ["voltage", "current", "temper", "soc", "soh"]
print(df[key_cols].describe())

# ---------- 第 5 步：缺失值 ----------
print("\n【5】每列缺失值个数(只显示有缺失的列)")
missing = df.isna().sum()
missing = missing[missing > 0]
print(missing if len(missing) else "没有缺失值")

# ---------- 第 6 步：抓脏数据 ----------
print("\n【6】脏数据/异常点排查")
# 6.1 停机时段：总电压为 0 的条数
stop_rows = (df["voltage"] == 0).sum()
print("电压=0(疑似停机/断电)的条数:", stop_rows, "占总", round(stop_rows / len(df) * 100, 2), "%")
# 6.2 电流异常大(取绝对值 > 1000A 看看有多少)
big_current = (df["current"].abs() > 1000).sum()
print("电流绝对值>1000A 的条数:", big_current)
# 6.3 温度异常(>60℃ 基本不合理，或 =0 疑似传感器离线)
high_temp = (df["temper"] > 60).sum()
zero_temp = (df["temper"] == 0).sum()
print("温度>60℃ 的条数:", high_temp, "| 温度=0 的条数:", zero_temp)
# 6.4 SOC 取值是否都在 0~100 合法区间
print("SOC 范围:", df["soc"].min(), "~", df["soc"].max())

# ---------- 第 7 步：画图 ----------
print("\n【7】开始画图，保存 3 张 PNG 到本目录")
# 7.1 电压随时间（取前 2000 条，避免太密）
plt.figure(figsize=(12, 4))
plt.plot(df["ts"][:2000], df["voltage"][:2000], linewidth=0.6)
plt.title("Battery pack voltage over time (first 2000 samples)")
plt.xlabel("time"); plt.ylabel("voltage (V)")
plt.tight_layout(); plt.savefig("eda_voltage.png", dpi=110); plt.close()

# 7.2 SOC 随时间
plt.figure(figsize=(12, 4))
plt.plot(df["ts"][:2000], df["soc"][:2000], linewidth=0.6, color="green")
plt.title("SOC over time (first 2000 samples)")
plt.xlabel("time"); plt.ylabel("SOC (%)")
plt.tight_layout(); plt.savefig("eda_soc.png", dpi=110); plt.close()

# 7.3 电压分布直方图（只统计"开机"状态，电压>0）
plt.figure(figsize=(8, 4))
df_on = df[df["voltage"] > 0]   # 过滤掉停机
plt.hist(df_on["voltage"], bins=60, color="steelblue", edgecolor="white")
plt.title("Voltage distribution (power-on only)")
plt.xlabel("voltage (V)"); plt.ylabel("count")
plt.tight_layout(); plt.savefig("eda_voltage_hist.png", dpi=110); plt.close()

print("完成！3 张图已保存：eda_voltage.png / eda_soc.png / eda_voltage_hist.png")
