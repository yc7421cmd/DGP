import librosa
import numpy as np

m = 0
while m <= 3600:
    m += 400
    data = np.load("sound_{}.npy".format(m))
    mfcc_lis = []
    for i in range(len(data)):
        # 预加重
        pre_emphasis_data = librosa.effects.preemphasis(data[i], coef=0.97, zi=np.zeros((len(data[i]), 1), dtype=data[i].dtype))


        pre_emphasis_data = pre_emphasis_data.flatten()
        frames = librosa.util.frame(pre_emphasis_data, frame_length=400, hop_length=160)

        # 计算MFCC
        mfcc = librosa.feature.mfcc(y=pre_emphasis_data, sr=16000, n_mfcc=13, hop_length=160)
        # 添加到列表
        mfcc_lis.append(mfcc.tolist())
    np.save('mfcc_{}.npy'.format(m), mfcc_lis)
