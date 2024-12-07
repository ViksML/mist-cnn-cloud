import torch
import torch.nn as nn
from model import Net
import pytest

class ModelTests:
    @staticmethod
    def test_parameter_count():
        """Test 1: Verify total parameters are less than 20k"""
        model = Net()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_params < 20000, f"Model has {total_params} parameters, should be < 20000"
        print(f"\nTest 1 Passed ✅ - Total parameters: {total_params}")
        return total_params

    @staticmethod
    def test_batch_normalization():
        """Test 2: Verify the use of Batch Normalization"""
        model = Net()
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
        model = Net()
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
    def test_fully_connected():
        """Test 4: Verify the use of Fully Connected layer or GAP"""
        model = Net()
        has_fc = False
        fc_count = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                has_fc = True
                fc_count += 1
        assert has_fc, "Model should use either Fully Connected layer or GAP"
        print(f"Test 4 Passed ✅ - Found {fc_count} Fully Connected layers")
        return fc_count

def run_tests():
    """Run all tests and generate report"""
    print("Running Model Architecture Tests...")
    print("="*50)
    
    try:
        results = {
            "parameters": ModelTests.test_parameter_count(),
            "batch_norm": ModelTests.test_batch_normalization(),
            "dropout": ModelTests.test_dropout(),
            "fc_layers": ModelTests.test_fully_connected()
        }
        
        print("\nTest Summary:")
        print("="*50)
        print(f"Total Parameters: {results['parameters']} (< 20k)")
        print(f"Batch Norm Layers: {results['batch_norm']}")
        print(f"Dropout Layers: {results['dropout']}")
        print(f"Fully Connected Layers: {results['fc_layers']}")
        print("\nAll tests passed successfully! ✅")
        
    except AssertionError as e:
        print(f"\nTest Failed! ❌")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    run_tests() 