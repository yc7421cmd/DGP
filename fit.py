import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import pandas as pd


# 取的特征是信号的熵,这个时候二阶拟合比一阶好
def calculate_entropy(spectrum):
    # 归一化频谱
    normalized_spectrum = spectrum / np.sum(spectrum)
    
    # 计算熵
    entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum))
    
    return entropy

# 时域和频域主要看load的是什么
# 时域的二阶貌似也比一阶好,频域和时域几乎差不多
def energy(lis):
    signal = np.array(lis)
    energy = np.sum(np.square(np.abs(signal)))
    return energy

# 效果挺好但是反了？应该松紧度越大频谱宽度越大
def freq_length(signal):

    # 计算信号的频谱
    fft_result = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal))

    # 找到主要峰值的索引
    main_peak_index = np.argmax(np.abs(fft_result))
    top_freq = freq[main_peak_index]
    # 找到功率的一半的频率点
    half_power_frequency = np.abs(freq[np.where(np.abs(fft_result) >= 0.5 * np.max(np.abs(fft_result)))[0]])

    # 计算半功率带宽
    bandwidth = 2 * np.abs(half_power_frequency[0])
    return bandwidth

# 能量的加权集中度，效果也挺好的
def center_energy(signal):
    spectrum = np.fft.fft(signal)
    # 有负是正常的，正一半负一半
    freq = np.fft.fftfreq(len(signal))
    # 计算频谱的幅度谱（能量）
    magnitude_spectrum = np.abs(spectrum)
    # 计算能量加权的平均频率
    weighted_average_frequency = np.sum(freq * np.abs(magnitude_spectrum)) / np.sum(magnitude_spectrum)
    # 计算频谱的总能量
    total_energy = np.sum(magnitude_spectrum)
    # 计算能量集中度
    energy_concentration = weighted_average_frequency / total_energy

    return energy_concentration


m = 0
X = []
y = []

while(m <= 3600):
    m += 400
    data=np.load("data/sound_{}.npy".format(m))
    X_tmp = [0] * len(data[0])
    for i in range(len(data)):
        for k in range(len(data[0])):
            X_tmp[k] += data[i][k][0]
    for i in range(len(X_tmp)):
        X_tmp[i] /= len(data)
    X.append(X_tmp)
    y.append(m)
feature = []
for i in range(len(X)):
    feature.append(center_energy(X[i]))
    
# 表格化
data = {'feature': feature, 'Y': y}
df = pd.DataFrame(data)
# 显示DataFrame
print(df)


# 显示拟合曲线
# 使用polyfit进行拟合，1代表一阶多项式
coefficients = np.polyfit(y, feature, 2)

# 生成拟合曲线的数据
fit_y = np.polyval(coefficients, y)

# 绘制原始数据和拟合曲线
plt.plot(y, feature, label='original')
plt.plot(y, fit_y, label='fit', color='red')
plt.xlabel('Tightness')
plt.ylabel('Feature')
plt.legend()
plt.show()

