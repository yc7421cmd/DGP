import scipy.io as scio
from scipy.fftpack import fftn
import numpy as np
import matplotlib.pyplot as plt
import time
import re
from pathlib import Path

# 从 .mat 文件加载数据
train_set = []
train_target = []
N = 512
fs = 51200 # 采样率，根据实际情况调整
lis_mainfre = []
lis_emergy = []
t0 = time.time()
folder_path = Path('train_data.txt')
j = 0
while(j <= 3600):
    tmp = []
    j += 400
    path = '{}raw.mat'.format(j)
    label = int(re.findall(r'\d+', path)[0])
    # print(label)
    data = scio.loadmat(path)["data"]
    audio_signal = data.astype(float)
    time_signal = abs(audio_signal)
    # 设置固定的时长
    fixed_duration = 1  # 以秒为单位
    locations = []

    i = 0
    while i < len(audio_signal):
        if np.abs(audio_signal[i]) >= 26:
            locations.append(i)
            i += 1000
        else:
            audio_signal[i] = 0
            i += 1
    # 分离出的每段音频时长相等
    separated_signals = []
    # print(len(locations))
    for i in range(len(locations)):
        separated_signals.append(audio_signal[locations[i] - int(N /4):locations[i] + int(N* 3 /4)])
    # seperated_signals 就是分割好的音频信号
    np.save('data/sound_{}.npy'.format(j), separated_signals)
    amp = np.zeros(shape=(N, 1))
    amp_list = []
    # f实际上就是横轴，就是采样频率作为横轴，而fft点数不是采样频率，是N
    f = np.arange(N) * fs / N
    plt.figure()
    for i, segment in enumerate(separated_signals):
        fft_result = fftn(segment)
        # print(fft_result)
        # 上面那个确实是X（k）
        # 计算频谱的振幅
        amplitude = np.abs(fft_result)
        total_emerge = np.sum(amplitude**2)
        
        tmp.append(np.max(amplitude))
        train_set.append(tmp)
        train_target.append(label)
        amp = amp + amplitude
        amp_list.append(amplitude)
    np.save('data/fft_{}.npy'.format(j), amp_list)

        
#     # 添加 x 轴标签、y 轴标签和标题
#     amp = amp / len(locations)
#     total_emerge = np.sum(amp**2)
    
#     main_frequence = f[np.argmax(amp)]
    
#     tmp.append(np.max(amp))
#     train_set.append(tmp)
#     train_target.append(label)
    
    
#     lis_mainfre.append(main_frequence)
#     lis_emergy.append(total_emerge)
#     plt.plot(f, amp)
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('P1(f)')
#     plt.title('Spectrum of Separated Signals')
#     plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
#     plt.savefig('{}-{}.png'.format(path, N))
#     # plt.show()
# np.savetxt('train_set',np.array(train_set))
# np.savetxt('train_target',np.array(train_target))
# print('main fre:',lis_mainfre)
# print('total energy:',lis_emergy)
# print(time.time() - t0)
