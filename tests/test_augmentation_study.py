import unittest

from data.augmentation_study import build_augmentation_sweep, summarize_augmentation_sweep


class TestAugmentationStudy(unittest.TestCase):
    def test_build_augmentation_sweep(self):
        sweep = build_augmentation_sweep()
        names = [item["name"] for item in sweep]

        self.assertIn("rotation", names)
        self.assertIn("translation", names)
        self.assertIn("crop", names)
        self.assertIn("noise", names)

    def test_summarize_augmentation_sweep(self):
        summary = summarize_augmentation_sweep()
        self.assertIn("rotation", summary)
        self.assertIn("crop", summary)


if __name__ == "__main__":
    unittest.main()
