"""
D3 CAP-extension rule (results/LOG.md, 2026-11-19), shared by the Phase 3 and Phase 4
reports. A run stopped at CAP is extended to the full 20,000 steps only if it is
ambiguous (final val_acc in [0.2, 0.9]) or unconverged (val_acc rose by more than 0.05
over its last 1000 steps, i.e. the last 4 evals at eval_every 250).
"""

AMBIG_LO, AMBIG_HI = 0.2, 0.9
RISE, WINDOW_STEPS = 0.05, 1000


def needs_extension(record: dict) -> list:
    """Reasons to extend (empty list = final at CAP). Early-stopped runs never extend."""
    if record.get("stopped_early"):
        return []
    h = record["history"]
    final = h[-1]["val_acc"]
    reasons = []
    if AMBIG_LO <= final <= AMBIG_HI:
        reasons.append(f"ambiguous: final {final:.3f} in [{AMBIG_LO}, {AMBIG_HI}]")
    back = [x for x in h if x["step"] <= h[-1]["step"] - WINDOW_STEPS]
    if back and final - back[-1]["val_acc"] > RISE:
        reasons.append(f"unconverged: +{final - back[-1]['val_acc']:.3f} over the last "
                       f"{WINDOW_STEPS} steps")
    return reasons
