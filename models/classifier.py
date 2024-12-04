import torch.nn as nn
import torch.nn.functional as F

class LabelClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LabelClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn2(self.fc2(self.dropout(x))))
        # return self.fc3(x)
        return self.fc3(self.dropout(x))

class DomainClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn2(self.fc2(self.dropout(x))))
        # return self.fc3(x)
        return self.fc3(self.dropout(x))
