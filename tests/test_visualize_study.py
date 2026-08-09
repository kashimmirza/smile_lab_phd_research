import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.visualize_study import visualize_image_samples


class TestVisualizeStudy(unittest.TestCase):
    def test_visualize_image_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "images"
            paper_dir = root / "papers"
            image_dir.mkdir(parents=True, exist_ok=True)
            paper_dir.mkdir(parents=True, exist_ok=True)

            np.save(image_dir / "sample_a.npy", np.random.rand(16, 16))
            (paper_dir / "sample_a.txt").write_text("Retinal microvascular imaging study", encoding="utf-8")

            output_path = visualize_image_samples(image_dir, paper_dir, limit=1, output_path=str(root / "out.png"))
            self.assertTrue(Path(output_path).exists())


if __name__ == "__main__":
    unittest.main()
