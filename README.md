# MNIST Digit Classification with PyTorch

[![Model Architecture Tests](https://github.com/ViksML/mist-back-propagation/actions/workflows/model_tests.yml/badge.svg)](https://github.com/ViksML/mist-back-propagation/actions/workflows/model_tests.yml)

This project implements a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset using PyTorch.

## Model Architecture
The model uses a three-block CNN architecture:

First Block:
- Conv2d (8 channels) → ReLU → BatchNorm2d
- Conv2d (8 channels) → ReLU → BatchNorm2d
- MaxPool2d → Dropout

Second Block:
- Conv2d (16 channels) → ReLU → BatchNorm2d
- Conv2d (16 channels) → ReLU → BatchNorm2d
- MaxPool2d → Dropout

Third Block:
- Conv2d (16 channels) → ReLU → BatchNorm2d
- Conv2d (10 channels) → ReLU → BatchNorm2d
- MaxPool2d → Dropout
- Global Average Pooling (GAP)

Model Summary:
- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)
- Total parameters: 6,790 (< 8,000)
- All blocks include BatchNorm, ReLU, MaxPooling, and Dropout

## Training Configuration
- Batch size: 16
- Epochs: 15 (max)
- Optimizer: Adam (lr=0.01)
- Loss Function: Cross Entropy Loss

## Training Results
Epoch: 1
Training: Average loss: 0.0344, Accuracy: 51421/60000 (85.70%)
Testing Model: Average loss: 0.0060, Accuracy: 9754/10000 (97.54%)

Epoch: 2
Training: Average loss: 0.0124, Accuracy: 56801/60000 (94.67%)
Testing Model: Average loss: 0.0047, Accuracy: 9763/10000 (97.63%)

Epoch: 3
Training: Average loss: 0.0092, Accuracy: 57415/60000 (95.69%)
Testing Model: Average loss: 0.0047, Accuracy: 9798/10000 (97.98%)

Epoch: 4
Training: Average loss: 0.0077, Accuracy: 57852/60000 (96.42%)
Testing Model: Average loss: 0.0037, Accuracy: 9816/10000 (98.16%)

Epoch: 5
Training: Average loss: 0.0063, Accuracy: 58200/60000 (97.00%)
Testing Model: Average loss: 0.0024, Accuracy: 9868/10000 (98.68%)

Epoch: 6
Training: Average loss: 0.0055, Accuracy: 58451/60000 (97.42%)
Testing Model: Average loss: 0.0025, Accuracy: 9876/10000 (98.76%)

Epoch: 7
Training: Average loss: 0.0049, Accuracy: 58586/60000 (97.64%)
Testing Model: Average loss: 0.0020, Accuracy: 9905/10000 (99.05%)

Epoch: 8
Training: Average loss: 0.0042, Accuracy: 58767/60000 (97.94%)
Testing Model: Average loss: 0.0020, Accuracy: 9894/10000 (98.94%)

Epoch: 9
Training: Average loss: 0.0037, Accuracy: 58904/60000 (98.17%)
Testing Model: Average loss: 0.0016, Accuracy: 9917/10000 (99.17%)

Epoch: 10
Training: Average loss: 0.0034, Accuracy: 59033/60000 (98.39%)
Testing Model: Average loss: 0.0016, Accuracy: 9912/10000 (99.12%)

Epoch: 11
Training: Average loss: 0.0029, Accuracy: 59152/60000 (98.59%)
Testing Model: Average loss: 0.0014, Accuracy: 9929/10000 (99.29%)

Epoch: 12 (Best)
Training: Average loss: 0.0027, Accuracy: 59214/60000 (98.69%)
Testing Model: Average loss: 0.0012, Accuracy: 9947/10000 (99.47%)

Epoch: 13
Training: Average loss: 0.0024, Accuracy: 59292/60000 (98.82%)
Testing Model: Average loss: 0.0012, Accuracy: 9942/10000 (99.42%)

Epoch: 14
Training: Average loss: 0.0023, Accuracy: 59341/60000 (98.90%)
Testing Model: Average loss: 0.0012, Accuracy: 9944/10000 (99.44%)

Epoch: 15
Training: Average loss: 0.0021, Accuracy: 59407/60000 (99.01%)
Testing Model: Average loss: 0.0012, Accuracy: 9942/10000 (99.42%)

## Model Performance
Best Model Statistics:
- Best Test Accuracy: 99.47%
- Achieved at Epoch: 12
- Training Loss at Best Epoch: 0.0027
- Final Test Accuracy: 99.42%
- Training Time per Epoch: ~30 seconds
- Loss Trend: 0.0344 → 0.0021 (Epoch 1 → 15)
- Accuracy Trend: 85.70% → 99.01% (Epoch 1 → 15)

## Model Testing
The project includes automated tests to verify model architecture requirements:

### Test Cases (test.py)
1. Parameter Count Test ✅
   - Verifies total parameters < 8,000
   - Current count: 6,790 parameters

2. Batch Normalization Test ✅
   - Checks for BatchNorm2d layers
   - Found: 6 BatchNorm layers

3. Dropout Test ✅
   - Verifies use of Dropout layers
   - Found: 3 Dropout layers

4. Global Average Pooling Test ✅
   - Checks for AdaptiveAvgPool2d layer
   - Found: 1 GAP layer

All tests passed successfully! ✅

## Project Structure
├── model.py          # CNN model architecture
├── train.py         # Training and evaluation code
├── test.py          # Model architecture tests
├── requirements.txt # Project dependencies
├── .github/
│   └── workflows/
│       └── model_tests.yml  # GitHub Actions workflow
└── README.md        # Project documentation

## Installation & Usage
1. Install dependencies:
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

2. Run training:
python train.py

3. Run tests:
python test.py

## License
MIT License
