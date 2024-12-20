import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Model_Final

# Check for Metal device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class ModelTests:
    @staticmethod
    def test_parameter_count():
        """Test 1: Verify total parameters are less than 20k"""
        model = Model_Final().to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_params < 8000, f"Model has {total_params} parameters, should be < 8000"
        print(f"\nTest 1 Passed ✅ - Total parameters: {total_params}")
        return total_params

    @staticmethod
    def test_batch_normalization():
        """Test 2: Verify the use of Batch Normalization"""
        model = Model_Final().to(device)
        has_batchnorm = False
        bn_count = 0
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                has_batchnorm = True
                bn_count += 1
        assert has_batchnorm, "Model should use Batch Normalization"
        print(f"Test 2 Passed ✅ - Found {bn_count} Batch Normalization layers")
        return bn_count

    @staticmethod
    def test_dropout():
        """Test 3: Verify the use of Dropout"""
        model = Model_Final().to(device)
        has_dropout = False
        dropout_count = 0
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                has_dropout = True
                dropout_count += 1
        assert has_dropout, "Model should use Dropout"
        print(f"Test 3 Passed ✅ - Found {dropout_count} Dropout layers")
        return dropout_count

    @staticmethod
    def test_gap_layer():
        """Test 4: Verify the use of Global Average Pooling layer"""
        model = Model_Final().to(device)
        gap_count = 0
        for module in model.modules():
            if isinstance(module, nn.AdaptiveAvgPool2d):
                gap_count += 1
        assert gap_count > 0, "Model should use Global Average Pooling layer"
        print(f"Test 4 Passed ✅ - Found {gap_count} Global Average Pooling layers")
        return gap_count

def run_tests():
    """Run all tests and generate report"""
    print("Running Model Architecture Tests...")
    print("="*50)
    
    try:
        results = {
            "parameters": ModelTests.test_parameter_count(),
            "batch_norm": ModelTests.test_batch_normalization(),
            "dropout": ModelTests.test_dropout(),
            "gap_layers": ModelTests.test_gap_layer()
        }
        
        print("\nTest Summary:")
        print("="*50)
        print(f"Total Parameters: {results['parameters']} (< 8000)")
        print(f"Batch Norm Layers: {results['batch_norm']}")
        print(f"Dropout Layers: {results['dropout']}")
        print(f"Global Average Pooling Layers: {results['gap_layers']}")
        print("\nAll tests passed successfully! ✅")
        
    except AssertionError as e:
        print(f"\nTest Failed! ❌")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    run_tests() 