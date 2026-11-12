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


def _run(tmp_path, monkeypatch, config, model_overrides, extra, steps=6):
    cfg = yaml.safe_load(open(config))
    cfg.update(max_steps=steps, eval_every=1, eval_batches=1, batch_size=4, warmup_steps=1)
    cfg["model"].update(model_overrides)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "run.json"
    monkeypatch.setattr(sys, "argv", [
        "train_task.py", "--config", str(cfg_path), "--task", "selective_copy",
        "--task_args", "n_data=4,seq_len=32", "--device", "cpu", "--quiet",
        "--out", str(out)] + extra)
    tt.main()
    return json.load(open(out))


TINY_TF = dict(n_layers=1, d_model=16, n_heads=2, d_ff=32, max_seq_len=64)


def test_early_stop_needs_consecutive_evals(tmp_path, monkeypatch):
    # every eval is > -1, so the run stops at the 3rd eval
    r = _run(tmp_path, monkeypatch, "configs/recall/transformer_small.yaml", TINY_TF,
             ["--early_stop_acc", "-1", "--early_stop_evals", "3"])
    assert r["stopped_early"] and r["steps_run"] == 3
    assert r["early_stop_acc"] == -1 and r["early_stop_evals"] == 3


def test_early_stop_is_strict(tmp_path, monkeypatch):
    # accuracy can never be strictly above 1.0
    r = _run(tmp_path, monkeypatch, "configs/recall/transformer_small.yaml", TINY_TF,
             ["--early_stop_acc", "1.0", "--early_stop_evals", "1"], steps=3)
    assert not r["stopped_early"] and r["steps_run"] == 3


def test_scan_and_checkpoint_flags_reach_custom_mamba(tmp_path, monkeypatch):
    tiny = dict(n_layers=1, d_model=16, d_state=4)
    r = _run(tmp_path, monkeypatch, "configs/recall/mamba_small.yaml", tiny,
             ["--scan", "parallel", "--grad_checkpoint", "false"], steps=2)
    assert r["scan"] == "parallel" and r["grad_checkpoint"] is False


def test_dump_answer_hist_counts_every_answer(tmp_path, monkeypatch):
    r = _run(tmp_path, monkeypatch, "configs/recall/transformer_small.yaml", TINY_TF,
             ["--dump_answer_hist"], steps=2)
    h = r["answer_hist"]
    n_answers = 4 * 4            # eval_batches=1, batch_size=4, n_data=4
    assert sum(h["pred"].values()) == sum(h["target"].values()) == n_answers
    assert all(2 <= int(k) < 64 for k in h["target"])


def test_dump_slot_acc_one_entry_per_slot_and_consistent(tmp_path, monkeypatch):
    r = _run(tmp_path, monkeypatch, "configs/recall/transformer_small.yaml", TINY_TF,
             ["--dump_slot_acc"], steps=2)
    assert len(r["slot_acc"]) == 4                     # n_data=4
    assert all(0.0 <= a <= 1.0 for a in r["slot_acc"])
    # equal slot sizes, so the mean of the slots is the overall accuracy
    assert abs(sum(r["slot_acc"]) / 4 - r["final_val_acc"]) < 1e-9


def test_schedule_length_independent_of_stop_step(tmp_path, monkeypatch):
    """--schedule_steps sets the lr schedule; --steps only decides when to stop."""
    seen = []
    real = tt.get_lr

    def spy(step, warmup, total, lr):
        seen.append(total)
        return real(step, warmup, total, lr)

    monkeypatch.setattr(tt, "get_lr", spy)
    r = _run(tmp_path, monkeypatch, "configs/recall/transformer_small.yaml", TINY_TF,
             ["--schedule_steps", "20000"], steps=3)
    assert r["schedule_steps"] == 20000 and r["max_steps"] == 3 and r["steps_run"] == 3
    assert set(seen) == {20000}


def test_schedule_defaults_to_stop_step(tmp_path, monkeypatch):
    r = _run(tmp_path, monkeypatch, "configs/recall/transformer_small.yaml", TINY_TF, [],
             steps=3)
    assert r["schedule_steps"] == 3
