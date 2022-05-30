"""
@Time    : 01.12.21 14:53
@Author  : Pushkar Jajoria
@File    : rnn_model.py
@Package : MLwithAudio
"""
import torch
from torch import nn


class RnnGenreClassifier(nn.Module):
    def __init__(self, input_shape, num_layers=1, hidden_size=64):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_shape, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = torch.from_numpy(x)
        x = x.float()
        lstm1 = torch.relu(self.lstm1(x)[0])
        lstm2 = torch.relu(self.lstm2(lstm1)[0][:, -1, :])
        out = torch.softmax(self.dropout(self.fc1(lstm2)), dim=1)
        return out

    def accuracy(self, x, labels):
        with torch.no_grad():
            self.eval()
            y = self.forward(x)
            l = len(labels)
            pred = torch.argmax(y, dim=1)
            accuracy_tensor = torch.abs(pred - torch.from_numpy(labels))
            accuracy_tensor[accuracy_tensor != 0] = 1
            return ((l - torch.sum(accuracy_tensor)) / l) * 100
