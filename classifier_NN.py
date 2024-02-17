import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc(x)
        return x

m = 0
X_train = []
y_train = []
X_test = []
y_test = []
while(m <= 3600):
    m += 400
    # 直接把分段之后的声音信号输入其中
    data=np.load("data/sound_{}.npy".format(m))
    data1=np.load("data/sound_{}.npy".format(m))
    for i in range(len(data) - 4):
        tmp = []
        for k in range(len(data[0])):
            tmp.append(data[i][k][0])
        for l in range(len(data1[0])):
            tmp.append(data1[i][l][0])
        X_train.append(tmp)
        y_train.append(m//400 - 1)
    for j in range(len(data) - 4,len(data)):
        tmp = []
        for k in range(len(data[0])):
            tmp.append(data[j][k][0])
        for l in range(len(data1[0])):
            tmp.append(data1[j][l][0])
        X_test.append(tmp)
        y_test.append(m//400 - 1)


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

y_test = y_test.view(-1)
y_train = y_train.view(-1)

input_size = X_train.shape[1]
hidden_size = 3 * input_size
num_classes = 10
model = SimpleClassifier(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 3000
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    model.eval()
    total_correct = 0
    total_samples = 0

    for i in range(len(y_test)):
        inputs = X_test[i]
        labels = y_test[i]
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 0)
        total_correct += torch.sum(predicted == labels).item()
        total_samples += 1  # Since labels is a scalar now, not a tensor

accuracy = total_correct / total_samples
print("Accuracy: {}%".format(100*accuracy))
# 2000轮，0.01，4096层 是95%