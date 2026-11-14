"""
select_blend.py - choose the per-current blend weights of versions A-D on the VALIDATION set.

For every current the FINAL NMOS ON prediction is sum_v w_v * prediction_v (w_v >= 0, sum = 1).
All weight combinations on a 0.05 grid are tried and the one with the highest validation mean
accuracy (100 - mean(min(err, 100))) is written to blend_weights.json. The test sets are not used.

Needs the trained models and their prediction caches (results/model_*/pred_cache_seed*.npz), which
ml_model/train_parallel.py writes. Run:  python select_blend.py
"""
import itertools, json, os
import numpy as np, joblib
import pipeline as PL
import physics_a, physics_b, physics_c, physics_d
from non_common import load_split, load_test_aug, CURRENTS

VERSIONS = {"A": physics_a, "B": physics_b, "C": physics_c, "D": physics_d}
STEP = 0.05
HERE = os.path.dirname(os.path.abspath(__file__))


def version_val_predictions(va):
    preds = {}
    for tag, mod in VERSIONS.items():
        vdir = os.path.join(PL.OUT, f"model_{tag.lower()}")
        PL.PB = mod
        mod.NOM.update(joblib.load(os.path.join(vdir, "nominal.joblib")))
        B = PL.build_table(va, joblib.load(os.path.join(vdir, "physics_params.joblib")))
        outs = [np.load(os.path.join(vdir, f"pred_cache_seed{s}.npz"))["val"] for s in PL.SEEDS]
        preds[tag] = PL.to_currents(B, np.mean(outs, axis=0))
    return preds


def mean_acc(t, q):
    e = np.abs(q - t) / np.abs(t) * 100
    return 100 - np.minimum(e, 100).mean()


def main():
    _, _, va, _ = load_split()
    preds = version_val_predictions(va)
    n = int(round(1 / STEP))
    grid = [np.array(c) / n for c in itertools.product(range(n + 1), repeat=len(VERSIONS)) if sum(c) == n]
    tags = list(VERSIONS)
    old = json.load(open(os.path.join(HERE, "blend_weights.json")))
    weights = {}
    for c in CURRENTS:
        t = va[c].values
        P = np.column_stack([preds[v][c].values for v in tags])
        accs = [mean_acc(t, P @ w) for w in grid]
        best = grid[int(np.argmax(accs))]
        weights[c] = {v: round(float(x), 2) for v, x in zip(tags, best)}
        old_acc = mean_acc(t, P @ np.array([old[c][v] for v in tags]))
        print(f"{c}: weights {weights[c]}  val mean accuracy {max(accs):.3f}%  (previous weights: {old_acc:.3f}%)")
    json.dump(weights, open(os.path.join(HERE, "blend_weights.json"), "w"), indent=1)
    print("written blend_weights.json")


if __name__ == "__main__":
    main()
