"""
@Time    : 30.11.21 11:27
@Author  : Pushkar Jajoria
@File    : genre_classification.py
@Package : MLwithAudio
"""

import json
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from GenreClassifier import preprocessor
from GenreClassifier.model import GenreClassifier
from GenreClassifier.rnn_model import RnnGenreClassifier


def load_data(file_path):
    with open(file_path, "r") as handle:
        data = json.load(handle)
    # convert list in numpy array
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    return inputs, targets


def fit():
    # Load Data
    inputs, targets = load_data(preprocessor.JSON_PATH)

    # Divide into train test
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3)
    # Create model
    # model = GenreClassifier(inputs.shape[1]*inputs.shape[2])
    model = RnnGenreClassifier(input_shape=inputs.shape[2], num_layers=2, hidden_size=64)

    optim = Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 50
    batch_size = 32
    for epoch in range(num_epochs):
        c = list(zip(inputs_train, targets_train))
        random.shuffle(c)
        inputs_train, targets_train = zip(*c)
        inputs_train, targets_train = np.array(inputs_train), np.array(targets_train)
        loss_arr = []
        for batch_start_idx in tqdm(range(0, inputs_train.shape[0], batch_size)):
            optim.zero_grad()
            y = model(inputs_train[batch_start_idx:min(batch_start_idx+batch_size, inputs_train.shape[0])])
            loss = criterion(y, torch.from_numpy(targets_train[batch_start_idx:min(batch_start_idx+batch_size, inputs_train.shape[0])]))
            loss.backward()
            loss_arr.append(loss.item())
            optim.step()
        print(f"Epoch{epoch} mean loss {np.mean(loss_arr)}")
        print(f"Epoch{epoch} Train Accuracy {model.accuracy(inputs_train, targets_train)}")
        print(f"Epoch{epoch} Test Accuracy {model.accuracy(inputs_test, targets_test)}")


if __name__ == "__main__":
    fit()
