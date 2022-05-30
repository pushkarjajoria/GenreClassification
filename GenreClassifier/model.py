"""
@Time    : 30.11.21 11:26
@Author  : Pushkar Jajoria
@File    : model.py
@Package : MLwithAudio
"""
import torch


class GenreClassifier(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.dense1 = torch.nn.Linear(in_features=input_shape, out_features=512)
        self.dense2 = torch.nn.Linear(in_features=512, out_features=256)
        self.dense3 = torch.nn.Linear(in_features=256, out_features=64)
        self.dense4 = torch.nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = torch.from_numpy(x)
        x = x.resize(x.shape[0], x.shape[1]*x.shape[2])
        x = x.float()
        dense1 = torch.relu(self.dense1(x))
        dense2 = torch.relu(self.dense2(dense1))
        dense3 = torch.relu(self.dense3(dense2))
        out = torch.softmax(self.dense4(dense3), dim=1)
        return out

    def accuracy(self, x, labels):
        with torch.no_grad():
            y = self.forward(x)
            l = len(labels)
            pred = torch.argmax(y, dim=1)
            accuracy_tensor = torch.abs(pred - torch.from_numpy(labels))
            accuracy_tensor[accuracy_tensor != 0] = 1
            return ((l - torch.sum(accuracy_tensor)) / l) * 100





