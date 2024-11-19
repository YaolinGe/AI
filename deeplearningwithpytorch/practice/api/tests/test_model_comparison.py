import unittest
import torch
import torch.nn as nn
import random
import numpy as np

from model.l2 import LSTMAutoEncoder as OldLSTMAutoEncoder
from model.LSTMAutoEncoder import LSTMAutoEncoder as NewLSTMAutoEncoder

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def copy_weights(source_model, target_model):
    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()
    
    print(f"Source model keys: {source_state_dict.keys()}") 
    print(f"Target model keys: {target_state_dict.keys()}")

    # Ensure the state dicts have the same keys
    assert len(source_state_dict.keys()) == len(target_state_dict.keys()), "Model structures don't match"
    
    for key in source_state_dict:
        target_state_dict[key] = source_state_dict[key].clone()
    
    target_model.load_state_dict(target_state_dict)

class TestLSTMAutoEncoder(unittest.TestCase):
    def setUp(self):
        seed = 42
        set_seed(seed)

        self.old_model = OldLSTMAutoEncoder()
        self.new_model = NewLSTMAutoEncoder(
            input_size=7,
            hidden_sizes_encoder=[128, 16],
            hidden_sizes_decoder=[16, 128],
            output_size=7
        )

        # Copy weights from old model to new model
        copy_weights(self.old_model, self.new_model)

        self.input_data = torch.randn(1, 30, 7)

    def test_model_outputs(self):
        self.old_model.eval()
        self.new_model.eval()

        with torch.no_grad():
            old_output = self.old_model(self.input_data)
            new_output = self.new_model(self.input_data)

        print(f"Old output shape: {old_output.shape}")
        print(f"New output shape: {new_output.shape}")
        print(f"Old output mean: {old_output.mean():.6f}")
        print(f"New output mean: {new_output.mean():.6f}")
        print(f"Old output std: {old_output.std():.6f}")
        print(f"New output std: {new_output.std():.6f}")
        
        is_close = torch.allclose(old_output, new_output, atol=1e-6)
        if not is_close:
            diff = (old_output - new_output).abs()
            print(f"Max difference: {diff.max().item():.6f}")
            print(f"Mean difference: {diff.mean().item():.6f}")

        self.assertTrue(is_close, "The outputs of the old and new models are not equal.")

    def test_model_parameters(self):
        old_params = sum(p.numel() for p in self.old_model.parameters())
        new_params = sum(p.numel() for p in self.new_model.parameters())

        print(f"Old model parameters: {old_params}")
        print(f"New model parameters: {new_params}")

        self.assertEqual(old_params, new_params,
                         f"Number of parameters differ. Old: {old_params}, New: {new_params}")

    def test_layer_outputs(self):
        self.old_model.eval()
        self.new_model.eval()

        def hook_fn(module, input, output):
            print(f"Layer: {module.__class__.__name__}")
            print(f"Input shape: {input[0].shape}")
            print(f"Output shape: {output[0].shape}")
            print(f"Output mean: {output[0].mean().item():.6f}")
            print(f"Output std: {output[0].std().item():.6f}")
            print("---")

        for name, module in self.old_model.named_modules():
            if isinstance(module, nn.LSTM):
                module.register_forward_hook(hook_fn)

        for name, module in self.new_model.named_modules():
            if isinstance(module, nn.LSTM):
                module.register_forward_hook(hook_fn)

        print("Old Model:")
        self.old_model(self.input_data)
        print("\nNew Model:")
        self.new_model(self.input_data)

if __name__ == '__main__':
    unittest.main()