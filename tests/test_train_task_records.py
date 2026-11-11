"""src/train_task.py writes its JSON record at every eval (complete=False) and once
more at the end (complete=True), so a run killed by a budget deadline keeps its
curve so far (plan2.md 0.2 lost four partial runs to this)."""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.train_task as tt


def test_record_checkpointed_every_eval(tmp_path, monkeypatch):
    cfg = yaml.safe_load(open("configs/recall/transformer_small.yaml"))
    cfg.update(max_steps=4, eval_every=2, eval_batches=1, batch_size=4, warmup_steps=1)
    cfg["model"].update(n_layers=1, d_model=16, n_heads=2, d_ff=32, max_seq_len=64)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "run.json"

    writes = []
    real_write = tt.write_record

    def spy(record, path):
        writes.append(json.loads(json.dumps(record)))
        real_write(record, path)

    monkeypatch.setattr(tt, "write_record", spy)
    monkeypatch.setattr(sys, "argv", [
        "train_task.py", "--config", str(cfg_path), "--task", "selective_copy",
        "--task_args", "n_data=4,seq_len=32", "--device", "cpu", "--quiet",
        "--out", str(out)])
    tt.main()

    assert [w["complete"] for w in writes] == [False, False, True]
    assert [w["steps_run"] for w in writes] == [2, 4, 4]
    assert writes[0]["final_val_acc"] is None
    final = json.load(open(out))
    assert final["complete"] is True and final["final_val_acc"] is not None
    assert not (tmp_path / "run.json.tmp").exists()
