"""Compare all models on identical, group-aware cross-validation folds.

Why cross-validation: 117 clean images is too few for a single 15% test split
(~18 images -> one mistake moves accuracy by 5 points). 5-fold x 3 repeats gives
every image a held-out prediction three times, so the numbers are more stable.
Why *group* folds: photos from the same series stay on the same side of a split.

Outputs
  reports/metrics.json          accuracy / precision / recall / F1 / calibration per model
  reports/confusion_<model>.png confusion matrices
  reports/errors_<model>.csv    every held-out misclassification (for failure analysis)
  models/thresholds.json     calibrated temperature + uncertainty thresholds

Usage:  python -m scripts.evaluate
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from .common import EXCLUSIONS, config, features, load_manifest, read_csv, write_csv
from .train import PROBES, choose_C, fit_head

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEEDS = [0, 1, 2]
N_SPLITS = 5
TARGET_PRECISION = 0.95   # "confident" predictions should be right at least this often
MIN_THRESHOLD = 0.50


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(logits, y, T=1.0):
    p = softmax(logits / T)
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))


def fit_temperature(logits, y) -> float:
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
    return float(grid[np.argmin([nll(logits, y, T) for T in grid])])


def ece(probs, y, bins=10) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def choose_threshold(probs, y) -> tuple[float, float, float]:
    """Smallest confidence threshold whose accepted predictions reach TARGET_PRECISION.
    Floored at MIN_THRESHOLD: a near-perfect model on 117 images would otherwise get a
    threshold so low it never warns, which would not hold on harder real-world images."""
    conf, correct = probs.max(1), probs.argmax(1) == y
    for t in np.arange(MIN_THRESHOLD, 0.96, 0.01):
        m = conf >= t
        if m.sum() >= 5 and correct[m].mean() >= TARGET_PRECISION:
            return round(float(t), 2), float(correct[m].mean()), float(m.mean())
    return 0.9, float(correct[conf >= 0.9].mean() if (conf >= 0.9).any() else 0), float((conf >= 0.9).mean())


def plot_confusion(cm, labels, title, path):
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cmn[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    from app.models import get_ood_head, zeroshot_head

    rows, y, groups = load_manifest()
    labels = [config.CLASSES[c] for c in config.CLASS_IDS]
    feats = {bb: features(bb, rows, "train") for bb in set(PROBES.values())}
    zs = zeroshot_head()
    zs_logits = zs(__import__("torch").from_numpy(feats["siglip2"][0])).numpy()

    models = ["siglip-zeroshot", *PROBES]
    oof = {m: [] for m in models}   # one (n, k) logit matrix per seed
    chosen_C = {m: [] for m in PROBES}
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        seed_logits = {m: np.zeros((len(y), len(labels))) for m in models}
        seed_logits["siglip-zeroshot"] = zs_logits           # no training -> same every fold
        for tr, te in cv.split(y, y, groups):
            for key, bb in PROBES.items():
                X, Xf = feats[bb]
                C = choose_C(X[tr], Xf[tr], y[tr], groups[tr], seed)   # nested: test fold never seen
                clf = fit_head(X[tr], Xf[tr], y[tr], C)
                seed_logits[key][te] = clf.decision_function(X[te])
                chosen_C[key].append(C)
        for m in models:
            oof[m].append(seed_logits[m])

    config.REPORTS_DIR.mkdir(exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    report = {"dataset": {"images": int(len(y)), "groups": int(len(set(groups))),
                          "per_class": {labels[k]: int((y == k).sum()) for k in range(len(labels))},
                          "excluded": len(read_csv(EXCLUSIONS))},
              "protocol": f"StratifiedGroupKFold k={N_SPLITS} x {len(SEEDS)} repeats; nested C selection; "
                          "temperature + thresholds fitted on out-of-fold predictions",
              "models": {}}
    thresholds = {}
    for m in models:
        L = np.concatenate(oof[m])
        Y = np.tile(y, len(SEEDS))
        T = fit_temperature(L, Y)
        P_raw, P = softmax(L), softmax(L / T)
        pred = P.argmax(1)
        accs = [(softmax(l).argmax(1) == y).mean() for l in oof[m]]
        top3 = np.mean([y[i] in np.argsort(-L[r])[:3] for r, i in enumerate(np.tile(np.arange(len(y)), len(SEEDS)))])
        p, r, f, _ = precision_recall_fscore_support(Y, pred, labels=range(len(labels)), zero_division=0)
        tau, prec_at_tau, coverage = choose_threshold(P, Y)
        thresholds[m] = {"temperature": round(T, 3), "min_confidence": tau, "min_margin": 0.10}
        cm = confusion_matrix(Y, pred, labels=range(len(labels)))
        plot_confusion(cm, labels, f"{m} (held-out, {len(SEEDS)}x{N_SPLITS}-fold)",
                       config.REPORTS_DIR / f"confusion_{m}.png")

        errs = []
        L0 = softmax(oof[m][0] / T)
        for i in np.where(L0.argmax(1) != y)[0]:
            errs.append({"file_name": rows[i]["file_name"], "true": labels[y[i]],
                         "predicted": labels[L0[i].argmax()], "confidence": round(float(L0[i].max()), 3)})
        if errs:
            write_csv(config.REPORTS_DIR / f"errors_{m}.csv", errs)

        report["models"][m] = {
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "accuracy_std": round(float(np.std(accs)), 4),
            "top3_accuracy": round(float(top3), 4),
            "macro_precision": round(float(p.mean()), 4),
            "macro_recall": round(float(r.mean()), 4),
            "macro_f1": round(float(f.mean()), 4),
            "per_class": {labels[k]: {"precision": round(float(p[k]), 3), "recall": round(float(r[k]), 3),
                                      "f1": round(float(f[k]), 3)} for k in range(len(labels))},
            "confusion_matrix": cm.tolist(),
            "calibration": {"temperature": round(T, 3), "nll_before": round(float(nll(L, Y)), 4),
                            "nll_after": round(float(nll(L, Y, T)), 4), "ece_before": round(ece(P_raw, Y), 4),
                            "ece_after": round(ece(P, Y), 4)},
            "uncertainty": {"min_confidence": tau, "precision_when_confident": round(prec_at_tau, 3),
                            "coverage": round(coverage, 3)},
            "chosen_C": sorted(set(chosen_C.get(m, []))) or None,
        }

    # Out-of-distribution gate: should reject the junk images removed in the audit,
    # and should NOT reject clean in-distribution images.
    import torch

    # "unrelated" junk (maps, people, landscapes) must be rejected; "defence-context" junk
    # (interiors, ceremonies, drawings) is defence content, so passing the gate is acceptable.
    ood = get_ood_head()
    excl = read_csv(EXCLUSIONS)
    Xe, _ = features("siglip2", [{"file_name": r["file_name"]} for r in excl], "excluded")
    flag = lambda X: np.array([ood.labels[i].startswith("other:")
                               for i in ood(torch.from_numpy(X)).argmax(1).numpy()])
    fe = flag(Xe)
    unrelated = np.array([r["content"] == "unrelated" for r in excl])
    report["ood_gate"] = {
        "unrelated_rejected": f"{int(fe[unrelated].sum())}/{int(unrelated.sum())}",
        "defence_context_rejected": f"{int(fe[~unrelated].sum())}/{int((~unrelated).sum())}",
        "false_reject_rate_clean": round(float(flag(feats["siglip2"][0]).mean()), 3),
    }

    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    (config.ARTIFACTS_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    print(f"\n{'model':18s} {'acc':>12s} {'top3':>6s} {'macroF1':>8s} {'ECE->':>12s} {'tau':>5s} {'cover':>6s}")
    for m, d in report["models"].items():
        c = d["calibration"]
        print(f"{m:18s} {d['accuracy_mean']:.3f}±{d['accuracy_std']:.3f} {d['top3_accuracy']:6.3f} "
              f"{d['macro_f1']:8.3f} {c['ece_before']:.3f}->{c['ece_after']:.3f} "
              f"{d['uncertainty']['min_confidence']:5.2f} {d['uncertainty']['coverage']:6.2f}")
    print("OOD gate:", report["ood_gate"])


if __name__ == "__main__":
    main()
