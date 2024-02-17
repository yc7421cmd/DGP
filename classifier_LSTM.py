import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

m = 0
X_train = []
y_train = []
X_test = []
y_test = []
while(m <= 3600):
    m += 400
    
    data=np.load("data/sound_{}.npy".format(m))
    
    for i in range(len(data) - 4):
        tmp = []
        for k in range(len(data[0])):
            tmp.append(data[i][k][0])
        X_train.append(tmp)
        y_train.append(m//400 - 1)
    for j in range(len(data) - 4,len(data)):
        tmp = []
        for k in range(len(data[0])):
            tmp.append(data[j][k][0])
        X_test.append(tmp)
        y_test.append(m//400 - 1)
X_train = np.array(X_train)
X_test = np.array(X_test)
# 将标签转换为 one-hot 编码
train_labels = to_categorical(y_train, num_classes=10)
test_labels = to_categorical(y_test, num_classes=10)


#效果变好了一点，但没差多少
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], 1)))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(10, activation='softmax'))
#减轻过拟合
model.add(Dropout(0.5))


model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, train_labels, epochs = 1024, validation_data=(X_test, test_labels))

# 评估模型
test_loss, test_acc = model.evaluate(X_test, test_labels)

print(f'Test accuracy: {test_acc:.4f}')

# 36轮，双向LSTM，learning_rate是0.001
#直接声音是0.475，fft是0.5，mfcc是0.55
# 128轮，0.001，mfcc是0.65


