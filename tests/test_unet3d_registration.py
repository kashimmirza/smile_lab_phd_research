import importlib.util
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "models" / "unet3d_registration.py"
SPEC = importlib.util.spec_from_file_location("unet3d_registration", MODULE_PATH)
unet3d_registration = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(unet3d_registration)


class TestUNet3DRegistration(unittest.TestCase):
    def test_model_initialization_and_forward_shape(self):
        model = unet3d_registration.UNet3DRegistration(in_channels=1, base_filters=8, out_channels=3)

        self.assertTrue(hasattr(model, "initialize_weights"))

        model.initialize_weights()

        sample = torch.randn(2, 1, 16, 16, 16)
        with torch.no_grad():
            output = model(sample)

        self.assertEqual(output.shape, (2, 3, 16, 16, 16))


if __name__ == "__main__":
    unittest.main()
