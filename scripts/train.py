"""Fit the final linear heads on all clean images and save them to models/.

The head is a multinomial logistic regression on frozen, L2-normalised embeddings.
Training data is augmented with horizontally-flipped copies. The regularisation
strength C is chosen by group-aware cross-validation, so photos from the same
series never appear on both sides of a split.

Usage:  python -m scripts.train
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from .common import config, features, load_manifest

C_GRID = [1, 3, 10, 30, 100, 300, 1000, 3000]   # embeddings are unit-norm, so useful C is large
PROBES = {"siglip-probe": "siglip2", "effnet-probe": "efficientnet_b0"}


def fit_head(X, X_flip, y, C) -> LogisticRegression:
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(np.concatenate([X, X_flip]), np.concatenate([y, y]))
    return clf


def choose_C(X, X_flip, y, groups, seed=0) -> float:
    """Inner group-aware CV over C (log-loss), using only the data it is given."""
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    best, best_loss = C_GRID[0], np.inf
    for C in C_GRID:
        losses = []
        for tr, va in cv.split(X, y, groups):
            clf = fit_head(X[tr], X_flip[tr], y[tr], C)
            p = clf.predict_proba(X[va])
            losses.append(-np.mean(np.log(p[np.arange(len(va)), y[va]] + 1e-12)))
        if np.mean(losses) < best_loss:
            best, best_loss = C, np.mean(losses)
    return best


def main():
    from app.models import LinearHead

    rows, y, groups = load_manifest()
    labels = [config.CLASSES[c] for c in config.CLASS_IDS]
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    for key, bb in PROBES.items():
        X, X_flip = features(bb, rows, "train")
        C = choose_C(X, X_flip, y, groups)
        clf = fit_head(X, X_flip, y, C)
        LinearHead.from_sklearn(clf, labels).save(config.ARTIFACTS_DIR / f"{key}.npz")
        print(f"{key}: trained on {len(y)} images (+flips), C={C}, train acc={clf.score(X, y):.3f}")


if __name__ == "__main__":
    main()
