"""Build dataset/manifest.csv from the ASTRA starter set.

1. Drop images listed in dataset/exclusions.csv (label noise found in a manual audit).
2. Validate every remaining file with the same loader the API uses.
3. Assign a *group* to each image so near-duplicates never straddle train/test:
   - photos from the same series (e.g. four shots of one tank at 'BAHNA 2018')
     share a filename stem -> same group;
   - perceptually near-identical images (pHash distance <= 8) are merged too.

Usage:  python -m scripts.prepare_data
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import imagehash

from .common import EXCLUSIONS, MANIFEST, config, open_path, read_csv, write_csv

NOISE = {"jpg", "jpeg", "png", "crop", "cropped", "JPG", "jp", "jpe"}


def series_key(file_name: str) -> str:
    stem = re.sub(r"^\d{3}-", "", Path(file_name).stem)
    tokens = [t for t in stem.split("-") if t and not t.isdigit() and t not in NOISE]
    return "-".join(tokens[:3]).lower()


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main():
    labels = read_csv(config.STARTER_DIR / "labels.csv")
    excluded = {r["file_name"] for r in read_csv(EXCLUSIONS)}
    rows, bad = [], []
    for r in labels:
        if r["file_name"] in excluded:
            continue
        try:
            img = open_path(config.STARTER_DIR / r["file_name"])
        except Exception as e:  # corrupt files are reported, not silently used
            bad.append((r["file_name"], str(e)))
            continue
        rows.append({**r, "phash": imagehash.phash(img)})

    uf = UnionFind(len(rows))
    by_series: dict[str, int] = {}
    for i, r in enumerate(rows):
        key = (r["category"], series_key(r["file_name"]))
        if key in by_series:
            uf.union(i, by_series[key])
        else:
            by_series[key] = i
    dupes = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i]["phash"] - rows[j]["phash"] <= 8:
                uf.union(i, j)
                dupes += 1

    out = [{"file_name": r["file_name"], "category": r["category"], "group": f"g{uf.find(i):03d}"}
           for i, r in enumerate(rows)]
    write_csv(MANIFEST, out)

    print(f"kept {len(out)} / {len(labels)} images ({len(excluded)} excluded, {len(bad)} unreadable)")
    for f, e in bad:
        print(f"  unreadable: {f}: {e}")
    print(f"near-duplicate pairs (pHash<=8): {dupes}")
    print(f"groups: {len({r['group'] for r in out})}")
    for c, n in sorted(Counter(r["category"] for r in out).items()):
        print(f"  {c:18s} {n}")


if __name__ == "__main__":
    main()
