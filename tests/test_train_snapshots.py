"""Check that fixed-step snapshots survive independent of the best checkpoint."""

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


class SnapshotTests(unittest.TestCase):
    def run_training(self, base, *extra):
        corpus = base / "corpus.txt"
        corpus.write_text("今天天气真不错啊\n" * 3000, encoding="utf-8")
        output = base / "run"
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "neural/train.py"),
                "--corpus", str(corpus),
                "--out", str(output),
                "--preset", "keyboard",
                "--vocab", "32",
                "--n-layer", "1",
                "--n-head", "2",
                "--n-embd", "16",
                "--context", "8",
                "--steps", "4",
                "--batch", "2",
                "--warmup", "2",
                "--eval-every", "3",
                *extra,
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return output

    def assert_best_matches_log(self, output):
        best = torch.load(output / "checkpoint.pt", weights_only=True)
        log = (output / "progress.log").read_text(encoding="utf-8")
        losses = [float(value) for value in re.findall(r"\bval (\d+\.\d+)", log)]
        self.assertEqual(round(best["val"], 4), min(losses))
        return best

    def test_one_run_keeps_both_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.run_training(Path(directory), "--snapshot-every", "2")
            first = torch.load(output / "checkpoint-step-000002.pt", weights_only=True)
            last = torch.load(output / "checkpoint-step-000004.pt", weights_only=True)
            self.assertEqual((first["step"], last["step"]), (2, 4))
            self.assertFalse(torch.equal(first["model"]["tok.weight"], last["model"]["tok.weight"]))
            best = self.assert_best_matches_log(output)
            self.assertIn(best["step"], (2, 3, 4))

    def test_default_keeps_only_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.run_training(Path(directory))
            self.assertEqual(list(output.glob("checkpoint-step-*.pt")), [])
            best = self.assert_best_matches_log(output)
            self.assertIn(best["step"], (3, 4))


if __name__ == "__main__":
    unittest.main()
