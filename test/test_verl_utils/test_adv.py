import numpy as np

# ① 在这里填你的数据，用逗号分隔
# 例如：data = [1.2, 3.4, 5.6, 7.8]
#data = [1, 2, 3, 4, 5]   # 改成你的数据
#data = [1/2, 1/2,1,1/3,1/3,1/3, 1/4,1/4,1/4,1/4,0,0,0,0,0,0,0,0,0,0]   # 改成你的数据
data = [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]   
# 转成数组
arr = np.array(data, dtype=float)

# ② 计算均值和方差（这里用总体方差：除以 N）
mean = np.mean(arr)
var = np.var(arr)  # 如果你想用样本方差就改成 np.var(arr, ddof=1)

# ③ 计算 (x - 均值) / 方差
result = (arr - mean) / (var+1e-8)

print("原始数据：", arr)
print("均值：", mean)
print("方差（除以 N）：", var)
print("每个数字的 (x - 均值) / 方差：")
for x, r in zip(arr, result):
    print(f"{x} -> {r}")