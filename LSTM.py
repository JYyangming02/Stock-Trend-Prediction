import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=100, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(100 * 2, 3)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return torch.softmax(x, dim=1)
    
# 30 is the sequence length, 5 is the number of features
# features_train = features_train.reshape(X_train.shape[0], 30, 5)
# features_val = features_val.reshape(X_test.shape[0], 30, 5)