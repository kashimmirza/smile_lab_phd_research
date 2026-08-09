import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.experimental_pipeline import build_experiment_manifest, run_ablation_study


class TestExperimentalStudy(unittest.TestCase):
    def test_build_manifest_and_ablation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "images"
            paper_dir = root / "papers"
            image_dir.mkdir(parents=True, exist_ok=True)
            paper_dir.mkdir(parents=True, exist_ok=True)

            np.save(image_dir / "sample_a.npy", np.random.rand(8, 8))
            np.save(image_dir / "sample_b.npy", np.random.rand(8, 8))

            (paper_dir / "sample_a.txt").write_text("cardiac motion registration retinal imaging", encoding="utf-8")
            (paper_dir / "sample_b.txt").write_text("language modeling benchmark for text generation", encoding="utf-8")

            manifest = build_experiment_manifest(image_dir=image_dir, paper_dir=paper_dir, keywords=["cardiac", "retinal"])
            self.assertEqual(len(manifest), 2)

            results = run_ablation_study(manifest, keywords=["cardiac", "retinal"])
            self.assertIn("image_only", results)
            self.assertIn("text_only", results)
            self.assertIn("image_text", results)


if __name__ == "__main__":
    unittest.main()
