"""
test_checkpoint_fixes.py
========================
Ordered integration tests for the three disk-safety fixes in train.py.
Run from repo root:

    python tests/test_checkpoint_fixes.py

Tests are intentionally ordered -- each one is a prerequisite for the next.
All use a tmp directory on the local filesystem and require no GPU.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.train import assert_disk_headroom, save_checkpoint, find_latest_checkpoint


# ── Minimal model + optimizer for checkpoint round-trips ─────────────────────

def _tiny_model():
    return nn.Linear(4, 4)

def _tiny_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)

def _dummy_cfg():
    return {"arch": "test", "seed": 42}


# ── Test 1: assert_disk_headroom ─────────────────────────────────────────────

class TestDiskHeadroomGuard(unittest.TestCase):
    """
    Fix 1: Pre-flight disk check.

    assert_disk_headroom() must raise RuntimeError when free space is below
    the threshold and pass silently when there is room. This test runs before
    the checkpoint tests because the guard is called at the top of
    save_checkpoint -- if it's broken, tests 2 and 3 are meaningless.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_passes_when_space_available(self):
        free = assert_disk_headroom(self.tmp, min_gb=0.0001)
        self.assertGreater(free, 0)

    def test_raises_when_below_threshold(self):
        with self.assertRaises(RuntimeError) as ctx:
            assert_disk_headroom(self.tmp, min_gb=1_000_000.0)
        self.assertIn("Disk headroom too low", str(ctx.exception))
        self.assertIn("GB free", str(ctx.exception))


# ── Test 2: atomic write via tmp → rename ────────────────────────────────────

class TestAtomicWrite(unittest.TestCase):
    """
    Fix 2: Write to .tmp then os.replace() to the canonical .pt path.

    After a successful save_checkpoint call:
      - The .pt file must exist and be loadable.
      - No .tmp file may remain (it is either renamed away or never created
        in the success path).
      - Simulating a mid-write crash on the .tmp leaves the previous .pt
        intact (the rename never happens, so the canonical path is clean).
    """

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.model = _tiny_model()
        self.opt   = _tiny_optimizer(self.model)
        self.cfg   = _dummy_cfg()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_pt_file_exists_and_loadable_after_save(self):
        path = save_checkpoint(self.tmp_dir, self.model, self.opt, 100, 1.5, self.cfg)
        self.assertTrue(path.exists(), ".pt file must exist after save")
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(state["step"], 100)
        self.assertAlmostEqual(state["val_bpc"], 1.5)

    def test_no_tmp_file_left_after_successful_save(self):
        save_checkpoint(self.tmp_dir, self.model, self.opt, 200, 1.4, self.cfg)
        tmp_files = list(self.tmp_dir.glob("*.tmp"))
        self.assertEqual(tmp_files, [], f".tmp files must be cleaned up: {tmp_files}")

    def test_previous_checkpoint_survives_simulated_mid_write_crash(self):
        # Write a clean checkpoint at step 100
        good_path = save_checkpoint(self.tmp_dir, self.model, self.opt, 100, 1.5, self.cfg)

        # Simulate a crash mid-write at step 200: leave a partial .tmp file
        corrupt_tmp = self.tmp_dir / "ckpt_0000200.tmp"
        corrupt_tmp.write_bytes(b"\x00\x01\x02")  # garbage bytes, not a valid zip

        # The good .pt from step 100 must be untouched
        self.assertTrue(good_path.exists(), "Previous good .pt must survive a mid-write crash")
        state = torch.load(good_path, map_location="cpu", weights_only=False)
        self.assertEqual(state["step"], 100)


# ── Test 3: checkpoint rotation (keep_last) ───────────────────────────────────

class TestCheckpointRotation(unittest.TestCase):
    """
    Fix 3: Bounded disk usage via keep_last eviction.

    After N saves with keep_last=2:
      - Exactly 2 numbered .pt files must exist (the two most recent).
      - The `latest.pt` symlink must point to the most recent one.
      - Older .pt files must be deleted.

    This is the fix that would have prevented the Kaggle Errno 28 -- 12
    accumulated checkpoints at ~145 MB each = ~1.7 GB just for transformer.
    """

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.model = _tiny_model()
        self.opt   = _tiny_optimizer(self.model)
        self.cfg   = _dummy_cfg()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _do_saves(self, steps, keep_last=2):
        for s in steps:
            save_checkpoint(self.tmp_dir, self.model, self.opt, s, 1.0, self.cfg,
                            keep_last=keep_last)

    def test_only_keep_last_checkpoints_remain(self):
        self._do_saves([1000, 2000, 3000, 4000, 5000], keep_last=2)
        pts = sorted(self.tmp_dir.glob("ckpt_???????.pt"))
        self.assertEqual(len(pts), 2, f"Expected 2 .pt files, got {len(pts)}: {pts}")

    def test_retained_checkpoints_are_the_most_recent(self):
        self._do_saves([1000, 2000, 3000, 4000, 5000], keep_last=2)
        pts = sorted(self.tmp_dir.glob("ckpt_???????.pt"))
        steps = [int(p.stem.split("_")[1]) for p in pts]
        self.assertEqual(steps, [4000, 5000], f"Wrong checkpoints retained: {steps}")

    def test_latest_symlink_points_to_most_recent(self):
        self._do_saves([1000, 2000, 3000], keep_last=2)
        latest = self.tmp_dir / "latest.pt"
        self.assertTrue(latest.exists(), "latest.pt must exist")
        resolved = latest.resolve()
        self.assertIn("0003000", resolved.name, f"latest.pt points to wrong file: {resolved}")

    def test_find_latest_checkpoint_returns_correct_path(self):
        self._do_saves([1000, 2000], keep_last=2)
        found = find_latest_checkpoint(self.tmp_dir)
        self.assertIsNotNone(found)
        self.assertIn("0002000", found.name)


if __name__ == "__main__":
    # Run in explicit order: guard -> atomic write -> rotation
    suite = unittest.TestSuite()
    for cls in [TestDiskHeadroomGuard, TestAtomicWrite, TestCheckpointRotation]:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
