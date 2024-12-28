import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_Final(nn.Module):
    # ----------------------------------------------------------------
    #         Layer (type)               Output Shape         Param #
    # ================================================================
    #             Conv2d-1            [-1, 8, 28, 28]              80
    #             ReLU-2            [-1, 8, 28, 28]               0
    #     BatchNorm2d-3            [-1, 8, 28, 28]              16
    #             Conv2d-4            [-1, 8, 28, 28]             584
    #             ReLU-5            [-1, 8, 28, 28]               0
    #     BatchNorm2d-6            [-1, 8, 28, 28]              16
    #         MaxPool2d-7            [-1, 8, 14, 14]               0
    #         Dropout-8            [-1, 8, 14, 14]               0
    #             Conv2d-9           [-1, 16, 14, 14]           1,168
    #             ReLU-10           [-1, 16, 14, 14]               0
    #     BatchNorm2d-11           [-1, 16, 14, 14]              32
    #         Conv2d-12           [-1, 16, 14, 14]           2,320
    #             ReLU-13           [-1, 16, 14, 14]               0
    #     BatchNorm2d-14           [-1, 16, 14, 14]              32
    #         MaxPool2d-15             [-1, 16, 7, 7]               0
    #         Dropout-16             [-1, 16, 7, 7]               0
    #         Conv2d-17             [-1, 16, 7, 7]           2,320
    #             ReLU-18             [-1, 16, 7, 7]               0
    #     BatchNorm2d-19             [-1, 16, 7, 7]              32
    #         Conv2d-20             [-1, 10, 7, 7]             170
    #             ReLU-21             [-1, 10, 7, 7]               0
    #     BatchNorm2d-22             [-1, 10, 7, 7]              20
    #         MaxPool2d-23             [-1, 10, 3, 3]               0
    #         Dropout-24             [-1, 10, 3, 3]               0
    # AdaptiveAvgPool2d-25             [-1, 10, 1, 1]               0
    # ================================================================
    # Total params: 6,790
    # Trainable params: 6,790
    # Non-trainable params: 0
    # ----------------------------------------------------------------
    # Input size (MB): 0.00
    # Forward/backward pass size (MB): 0.50
    # Params size (MB): 0.03
    # Estimated Total Size (MB): 0.53

    # ----------------------------------------------------------------
    # Receptive Field (RF) calculation:

    # RF = 1 + sum((kernel_size - 1) * stride_product)
    # stride_product = product of all previous strides

    # Layer details:
    # Conv1: RF_in=1, k=3, s=1, p=1 → RF_out=3
    # Conv2: RF_in=3, k=3, s=1, p=1 → RF_out=5
    # MaxPool1: RF_in=5, k=2, s=2 → RF_out=6

    # Conv3: RF_in=6, k=3, s=1, p=1 → RF_out=10
    # Conv4: RF_in=10, k=3, s=1, p=1 → RF_out=14
    # MaxPool2: RF_in=14, k=2, s=2 → RF_out=16

    # Conv5: RF_in=16, k=3, s=1, p=1 → RF_out=20
    # Conv6: RF_in=20, k=1, s=1 → RF_out=20
    # MaxPool3: RF_in=20, k=2, s=2 → RF_out=22

    # Final RF = 22x22
    #  ----------------------------------------------------------------

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

        