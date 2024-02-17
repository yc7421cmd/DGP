from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score ,classification_report
import numpy as np

m = 0
X_train = []
y_train = []
X_test = []
y_test = []
while(m <= 3600):
    m += 400
    # 直接把分段之后的声音信号输入其中
    data=np.load("data/sound_{}.npy".format(m))
    for i in range(len(data) - 4):
        tmp = []
        for k in range(len(data[0])):
            tmp.append(data[i][k][0])
        X_train.append(tmp)
        y_train.append(m)
    for j in range(len(data) - 4,len(data)):
        tmp = []
        for k in range(len(data[0])):
            tmp.append(data[j][k][0])
        X_test.append(tmp)
        y_test.append(m)

# # 创建 SVM 分类器
clf = svm.SVC(kernel='linear')  # 选择线性核，也可以尝试其他核函数
# 训练模型
clf.fit(X_train, y_train)
# 在测试集上进行预测
y_pred = clf.predict(X_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print('Classification Report:\n', report)
# 时间是0.95，fft是0.875 而且核需要是linear，mfcc 是0.65