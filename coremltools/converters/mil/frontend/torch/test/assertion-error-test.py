import unittest
import torch
import torch.nn as nn
from coremltools.converters.mil.frontend.torch.test.testing_utils import (
    ModuleWrapper,
)

class TestModuleWrapper(unittest.TestCase):
    def setUp(self):
        self.model = ModuleWrapper(
            function=nn.functional.scaled_dot_product_attention,
            kwargs={
                "attn_mask": None,
                "is_causal": True,
            },
        )

    def test_forward(self):
        query = torch.randn(10, 32)
        key = torch.randn(10, 32)
        value = torch.randn(10, 32)

        outputs = self.model(query, key, value)
        output = outputs[0]  

        print(f"Actual output shape: {output.shape}")

        self.assertEqual(output.shape, (10, 32))

if __name__ == "__main__":
    unittest.main()