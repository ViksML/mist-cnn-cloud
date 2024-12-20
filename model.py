import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_Final(nn.Module):
    def __init__(self):
        super(Model_Final, self).__init__()

        # Use float32 for better compatibility with Metal
        self.dtype = torch.float32
        self.name = 'model_final'
        
        # First block - extract basic features
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        # Second block - increase feature complexity
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        # Third block - maintain feature depth and prepare for GAP
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 10, 1),  # Changed to output 10 channels (number of classes)
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(-1, 10)  # Reshape to (batch_size, num_classes)
        return F.log_softmax(x, dim=1)



class Model_ImprovingAccuracy(nn.Module):
    def __init__(self):
        super(Model_ImprovingAccuracy, self).__init__()
        
        # Use float32 for better compatibility with Metal
        self.dtype = torch.float32
        self.name = 'model_improvingccuracy'
        
       # First block - extract basic features
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        # Second block - increase feature complexity
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        # Third block - maintain feature depth and prepare for GAP
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 10, 1),  # Changed to output 10 channels (number of classes)
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        self.fc = nn.Sequential(nn.Linear(10 * 3 * 3, 10))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class Model_ReducingParams(nn.Module):
    def __init__(self):
        super(Model_ReducingParams, self).__init__()
        
        # Use float32 for better compatibility with Metal
        self.dtype = torch.float32
        self.name = 'model_reducing_params'
        
        # First block - extract basic features
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),  
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Second block - increase feature complexity
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Third block - maintain feature depth
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 3 * 3, 10)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class Model_Base(nn.Module):
    def __init__(self):
        super(Model_Base, self).__init__()
        
        # Use float32 for better compatibility with Metal
        self.dtype = torch.float32
        self.name = 'model_base'

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 25, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(625, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 25 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.log_softmax(self.fc2(x), dim=1)
        return x

        