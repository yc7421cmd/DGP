from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import numpy as np

m = 0
X_train = []
y_train = []
X_test = []
y_test = []
while(m <= 3600):
    m += 400

    data=np.load("data/fft_{}.npy".format(m))
    
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

# 创建 KNN 分类器
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# 在训练集上训练模型
knn_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = knn_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 打印分类报告
report = classification_report(y_test, y_pred)
print('Classification Report:\n', report)

#fft是0.8，直接声音是0.78,mfcc是0.57
