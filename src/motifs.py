import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import representation as rep
from q4_failures import raw_confidence
from normalization import fit_basis, normalize

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

R = 20
WIN = 16         
STRIDE = 8
MOTION_KEEP = 0.5 


def windows(Z, vid):
    out = []
    for s in range(0, len(Z) - WIN + 1, STRIDE):
        out.append((Z[s:s + WIN], vid))
    return out


def nearest_cross_video(Q, qv, B, bv, chunk=512):
    best = np.full(len(Q), np.inf, dtype=np.float32)
    Bsq = (B ** 2).sum(1)
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk]
        d2 = (q ** 2).sum(1)[:, None] + Bsq[None, :] - 2 * q @ B.T
        d2[qv[i:i + chunk, None] == bv[None, :]] = np.inf
        best[i:i + chunk] = np.sqrt(np.maximum(d2.min(1), 0))
    return best


def run(name, trajs, videos):
    W = [w for Z, v in zip(trajs, videos) for w in windows(Z, v)]
    X = np.stack([w for w, _ in W]).astype(np.float32)       
    V = np.array([v for _, v in W])

    motion = np.linalg.norm(np.diff(X, axis=1), axis=2).sum(1)
    keep = motion >= np.quantile(motion, 1 - MOTION_KEEP)
    X, V = X[keep], V[keep]

    fwd = X.reshape(len(X), -1)
    rev = X[:, ::-1].reshape(len(X), -1)
    d_fwd = nearest_cross_video(fwd, V, fwd, V)
    d_rev = nearest_cross_video(fwd, V, rev, V)

    rng = np.random.RandomState(0)
    uv = np.unique(V)
    win_of = {v: np.flatnonzero(V == v) for v in uv}
    boot = []
    for _ in range(2000):
        idx = np.concatenate([win_of[v] for v in rng.choice(uv, len(uv))])
        boot.append((d_fwd[idx] < d_rev[idx]).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])

    res = {
        "fraction_forward_closer_95ci_by_video": [float(lo), float(hi)],
        "windows": int(len(X)),
        "window_frames": WIN,
        "stride": STRIDE,
        "fraction_forward_closer": float((d_fwd < d_rev).mean()),
        "median_forward_distance": float(np.median(d_fwd)),
        "median_reversed_distance": float(np.median(d_rev)),
        "median_ratio_reversed_over_forward": float(np.median(d_rev / d_fwd)),
    }
    print("  %-11s %5d windows | forward closer in %.1f%% "
          "(95%% CI %.1f-%.1f, by video) | median rev/fwd %.3f"
          % (name, res["windows"], 100 * res["fraction_forward_closer"],
             100 * lo, 100 * hi, res["median_ratio_reversed_over_forward"]))
    return res, d_fwd, d_rev


def main():
    basis = np.load(RESULTS / "basis.npz", allow_pickle=True)
    train_videos = set(basis["train_videos"].tolist())

    print("loading sequences...")
    seqs = rep.load_all(verbose=False)
    for s in seqs:
        s["N"], _ = normalize(s["X"], raw_confidence(rep.POSE_DIR / s["file"]))
    train = [s for s in seqs if s["video_id"] in train_videos]
    videos = [s["video_id"] for s in seqs]

    out, dists = {}, {}
    for name, key, center in [("raw", "X", False), ("normalized", "N", True)]:
        U, _, mu = fit_basis(
            np.concatenate([s[key] for s in train], axis=0).T, center)
        trajs = [(s[key] - mu.T) @ U[:, :R] for s in seqs]
        out[name], d_fwd, d_rev = run(name, trajs, videos)
        dists[name] = (d_fwd, d_rev)

    (RESULTS / "motifs.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, name in zip(ax, dists):
        d_fwd, d_rev = dists[name]
        ratio = np.log2(d_rev / d_fwd)
        a.hist(ratio, bins=60, color="tab:blue")
        a.axvline(0, color="k", ls="--", lw=1)
        a.set_xlabel("log2(reversed match / forward match)   (> 0: forward closer)")
        a.set_ylabel("windows")
        a.set_title("%s: forward closer in %.1f%%"
                    % (name, 100 * out[name]["fraction_forward_closer"]))
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "motifs.png", dpi=150)
    print("\nwrote results/motifs.json, figures/motifs.png")


if __name__ == "__main__":
    main()
