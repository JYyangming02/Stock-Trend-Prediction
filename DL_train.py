import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ANN import ANN

def create_lagged_features(df, days=30):
    features = []
    targets = []

    for i in tqdm(range(days-1, len(df))):
        feature_row = []
        target = df.iloc[i]['1_trend']

        # 過去days天的數據
        for j in range(days-1, -1, -1):
            feature_row.extend([
                df.iloc[i - j]['close'],
                df.iloc[i - j]['open'],
                df.iloc[i - j]['high'],
                df.iloc[i - j]['low'],
                df.iloc[i - j]['volume']
            ])
        features.append(feature_row)
        targets.append(target)

    return np.array(features), np.array(targets)

def create_test_features(df, days=30):
    
    features = []
    for i in tqdm(range(days-1, len(df)+1, days)):
        feature_row = []

        for j in range(days-1, -1, -1):
            feature_row.extend([
                df.iloc[i - j]['close'],
                df.iloc[i - j]['open'],
                df.iloc[i - j]['high'],
                df.iloc[i - j]['low'],
                df.iloc[i - j]['volume']
            ])

        features.append(feature_row)
    
    return np.array(features)

def train_ann(features_train, targets_train, features_val, targets_val):
    input_size = 150 # 5 features * 30 days
    hidden_size = 64
    output_size = 3 # 3 classes
    model = ANN(input_size, hidden_size, output_size)
    class_weights = torch.tensor([1.14, 0.79, 1.15], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 2000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features_train)
        loss = criterion(output, targets_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1} / {epochs}, loss: {loss.item()}')

        model.eval()
        with torch.no_grad():
            output = model(features_val)
            val_loss = criterion(output, targets_val)
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1} / {epochs}, val_loss: {val_loss.item()}')
    return model

if __name__ == '__main__':
    train_df = pd.read_csv('./data/train.csv')
    train_df['1_trend'] = train_df['1_trend'].map({-1: 0, 0: 1, 1: 2})
    train_df['5_trend'] = train_df['5_trend'].map({-1: 0, 0: 1, 1: 2})
    train_df['10_trend'] = train_df['10_trend'].map({-1: 0, 0: 1, 1: 2})
    features, targets = create_lagged_features(train_df)

    # Split the data
    features_train, features_val, targets_train, targets_val = train_test_split(features, targets, test_size=0.3, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_val = scaler.transform(features_val)

    # Convert the data to PyTorch tensors
    features_train = torch.tensor(features_train, dtype=torch.float32)
    features_val = torch.tensor(features_val, dtype=torch.float32)
    targets_train = torch.tensor(targets_train, dtype=torch.long)
    targets_val = torch.tensor(targets_val, dtype=torch.long)

    model = train_ann(features_train, targets_train, features_val, targets_val)

    model.eval()
    with torch.no_grad():
        output = model(features_val)
        _, predicted = torch.max(output, 1)
        print('ANN accuracy:', accuracy_score(targets_val, predicted))
        print(classification_report(targets_val, predicted))

    # Test the model
    test_df = pd.read_csv('./data/test.csv')
    test_features = create_test_features(test_df)

    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_features = scaler.transform(test_features)
    test_features = torch.tensor(test_features, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        output = model(test_features)
        predicted = torch.argmax(output, 1).numpy()

    class_map = {0: -1, 1: 0, 2: 1}
    mapped_predictions = np.array([class_map[p] for p in predicted])

    output_df = pd.DataFrame({
        'id': [i for i in range(len(mapped_predictions))],
        'trend': mapped_predictions
    })

    output_df.to_csv('./data/ann_predictions1.csv', index=False)

