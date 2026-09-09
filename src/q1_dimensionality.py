"""
1. Is the structure low-dimensional? How quickly do the singular values decay? 
    How much variance can be explained with r << D? Does reconstruction remain reasonable at small r?
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import representation as rep

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

R_GRID = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80, 100, 152]


def reconstruction_errors(M, U, r_grid):
    """Relative Frobenius error and mean per-keypoint pixel error vs r.

    Reconstruction uses the rank-r projector: M_hat = Ur Ur^T M.
    """
    total = np.linalg.norm(M)
    out = []
    for r in r_grid:
        Ur = U[:, :r]
        M_hat = Ur @ (Ur.T @ M)
        resid = M - M_hat
        rel = np.linalg.norm(resid) / total
        # residual per keypoint, in pixels: reshape (152, K) -> (76, 2, K)
        per_pt = np.sqrt((resid.reshape(rep.N_POINTS, 2, -1) ** 2).sum(axis=1))
        out.append({
            "r": int(r),
            "relative_frobenius_error": float(rel),
            "mean_keypoint_error_px": float(per_pt.mean()),
            "median_keypoint_error_px": float(np.median(per_pt)),
        })
    return out


def main():
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    print("loading sequences...")
    seqs = rep.load_all()
    train, test = rep.split_by_video(seqs)
    rep.describe(seqs, "all")
    rep.describe(train, "train")
    rep.describe(test, "held-out")

    M = rep.stack(train)                      # (152, K)
    print("\nM shape: %s  (D=%d, K=%d)" % (M.shape, M.shape[0], M.shape[1]))

    # M = U S V^T
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    print("computed SVD; %d singular values" % len(S))

    energy = S ** 2
    cum = np.cumsum(energy) / energy.sum()

    recon = reconstruction_errors(M, U, R_GRID)

    thresholds = {}
    for q in [0.90, 0.95, 0.99, 0.999]:
        thresholds["r_for_%.3f" % q] = int(np.searchsorted(cum, q) + 1)

    results = {
        "D": rep.D,
        "n_frames_train": int(M.shape[1]),
        "n_sequences_train": len(train),
        "n_videos_train": len({s["video_id"] for s in train}),
        "centred": False,
        "singular_values": [float(x) for x in S],
        "cumulative_variance_explained": [float(x) for x in cum],
        "variance_thresholds": thresholds,
        "reconstruction": recon,
    }
    (RESULTS / "q1.json").write_text(json.dumps(results, indent=2))

    print("\nvariance explained:")
    for r in [1, 2, 3, 5, 10, 20, 50]:
        print("  r=%3d  %6.2f%%" % (r, 100 * cum[r - 1]))
    print("\ncomponents needed:")
    for k, v in thresholds.items():
        print("  %-14s %d" % (k, v))
    print("\nreconstruction:")
    for row in recon:
        print("  r=%3d  rel.err %6.4f   mean kp error %7.2f px"
              % (row["r"], row["relative_frobenius_error"],
                 row["mean_keypoint_error_px"]))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(range(1, len(S) + 1), S, marker=".")
    ax[0].set_xlabel("component index")
    ax[0].set_ylabel("singular value (log)")
    ax[0].set_title("Singular value decay (uncentered SVD of M)")
    ax[0].grid(alpha=.3)

    ax[1].plot(range(1, len(cum) + 1), 100 * cum, marker=".")
    ax[1].set_xlabel("r")
    ax[1].set_ylabel("cumulative variance explained (%)")
    ax[1].set_title("Variance explained vs r")
    ax[1].grid(alpha=.3)
    err1 = recon[0]["mean_keypoint_error_px"]
    ax[1].annotate("r=1: %.2f%% variance,\nbut %.1f px error"
                   % (100 * cum[0], err1),
                   xy=(1, 100 * cum[0]), xytext=(45, 95.6),
                   fontsize=8, ha="left", va="center",
                   arrowprops=dict(arrowstyle="->", lw=.8, color="0.35"))
    fig.tight_layout()
    fig.savefig(FIGURES / "q1_spectrum.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    rs = [row["r"] for row in recon]
    ax.plot(rs, [row["mean_keypoint_error_px"] for row in recon], marker="o")
    ax.set_xlabel("r")
    ax.set_ylabel("mean keypoint error (pixels)")
    ax.set_title("Reconstruction error vs r")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "q1_reconstruction.png", dpi=150)

    np.savez_compressed(
        RESULTS / "basis.npz",
        U=U, S=S,
        train_videos=np.array(sorted({s["video_id"] for s in train})),
        test_videos=np.array(sorted({s["video_id"] for s in test})),
    )
    print("\nwrote results/q1.json, results/basis.npz, figures/q1_*.png")


if __name__ == "__main__":
    main()
