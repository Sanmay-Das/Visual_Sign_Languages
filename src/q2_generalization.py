"""
2. Learn Ur on one set of sequences and apply it to unseen sequences. Test across
different/unseen videos/pose-sequences. Does one learned space describe motion
produced by other signers?
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


def curves(M, U, r_grid):
    """Variance explained and reconstruction error vs r, for a given matrix."""
    total_sq = float((M ** 2).sum())
    total = np.sqrt(total_sq)
    out = []
    for r in r_grid:
        Ur = U[:, :r]
        Z = Ur.T @ M                    
        M_hat = Ur @ Z
        resid = M - M_hat
        captured = float((Z ** 2).sum()) / total_sq
        per_pt = np.sqrt((resid.reshape(rep.N_POINTS, 2, -1) ** 2).sum(axis=1))
        out.append({
            "r": int(r),
            "variance_explained": captured,
            "relative_frobenius_error": float(np.linalg.norm(resid) / total),
            "mean_keypoint_error_px": float(per_pt.mean()),
            "median_keypoint_error_px": float(np.median(per_pt)),
        })
    return out


def per_video_errors(seqs, U, r):
    """Reconstruction error for each held-out video separately."""
    Ur = U[:, :r]
    by_video = {}
    for s in seqs:
        by_video.setdefault(s["video_id"], []).append(s["X"])
    rows = []
    for vid, arrs in sorted(by_video.items()):
        M = np.concatenate(arrs, axis=0).T
        resid = M - Ur @ (Ur.T @ M)
        per_pt = np.sqrt((resid.reshape(rep.N_POINTS, 2, -1) ** 2).sum(axis=1))
        rows.append({
            "video_id": vid,
            "frames": int(M.shape[1]),
            "relative_frobenius_error": float(
                np.linalg.norm(resid) / np.linalg.norm(M)),
            "mean_keypoint_error_px": float(per_pt.mean()),
        })
    return rows


def main():
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    basis = np.load(RESULTS / "basis.npz", allow_pickle=True)
    U = basis["U"]
    train_videos = set(basis["train_videos"].tolist())
    test_videos = set(basis["test_videos"].tolist())
    print("loaded basis fitted on %d training videos" % len(train_videos))
    print("held-out videos: %d" % len(test_videos))

    print("\nloading sequences...")
    seqs = rep.load_all()
    train = [s for s in seqs if s["video_id"] in train_videos]
    test = [s for s in seqs if s["video_id"] in test_videos]
    rep.describe(train, "train")
    rep.describe(test, "held-out")

    # sanity: the split must be disjoint, or Q2 means nothing
    assert not (train_videos & test_videos), "train/held-out video overlap"

    M_train = rep.stack(train)
    M_test = rep.stack(test)

    train_curve = curves(M_train, U, R_GRID)
    test_curve = curves(M_test, U, R_GRID)

    print("\n  r |  train var%   held-out var%  |  train px   held-out px")
    print("  " + "-" * 60)
    for a, b in zip(train_curve, test_curve):
        print("  %3d |   %6.2f%%       %6.2f%%      |  %7.2f    %7.2f" % (
            a["r"], 100 * a["variance_explained"], 100 * b["variance_explained"],
            a["mean_keypoint_error_px"], b["mean_keypoint_error_px"]))

    # per-video breakdown at a few ranks, to see if failure is uniform
    per_video = {str(r): per_video_errors(test, U, r) for r in [10, 20, 40]}
    for r in [10, 20, 40]:
        errs = [v["mean_keypoint_error_px"] for v in per_video[str(r)]]
        print("\nheld-out per-video error at r=%d: min %.2f  median %.2f  "
              "max %.2f px" % (r, min(errs), float(np.median(errs)), max(errs)))
        worst = sorted(per_video[str(r)],
                       key=lambda v: -v["mean_keypoint_error_px"])[:3]
        for w in worst:
            print("    worst: %-14s %7.2f px  (%d frames)" % (
                w["video_id"], w["mean_keypoint_error_px"], w["frames"]))

    results = {
        "n_train_videos": len(train_videos),
        "n_test_videos": len(test_videos),
        "n_frames_train": int(M_train.shape[1]),
        "n_frames_test": int(M_test.shape[1]),
        "basis_refitted": False,
        "train": train_curve,
        "held_out": test_curve,
        "held_out_per_video": per_video,
    }
    (RESULTS / "q2.json").write_text(json.dumps(results, indent=2))

    rs = [row["r"] for row in train_curve]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(rs, [100 * x["variance_explained"] for x in train_curve],
               marker="o", label="train (80 videos)")
    ax[0].plot(rs, [100 * x["variance_explained"] for x in test_curve],
               marker="s", label="unseen (20 videos)")
    ax[0].set_xlabel("r")
    ax[0].set_ylabel("variance explained (%)")
    ax[0].set_title("Shared basis applied to unseen videos")
    ax[0].legend()
    ax[0].grid(alpha=.3)

    ax[1].plot(rs, [x["mean_keypoint_error_px"] for x in train_curve],
               marker="o", label="train")
    ax[1].plot(rs, [x["mean_keypoint_error_px"] for x in test_curve],
               marker="s", label="unseen")
    ax[1].set_xlabel("r")
    ax[1].set_ylabel("mean keypoint error (pixels)")
    ax[1].set_title("Reconstruction error, train vs unseen")
    ax[1].legend()
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "q2_generalisation.png", dpi=150)

    # spread across held-out videos at r=20
    fig, ax = plt.subplots(figsize=(9, 4))
    rows = sorted(per_video["20"], key=lambda v: v["mean_keypoint_error_px"])
    ax.bar(range(len(rows)), [v["mean_keypoint_error_px"] for v in rows])
    ax.axhline(np.mean([v["mean_keypoint_error_px"] for v in rows]),
               color="k", ls="--", lw=1, label="mean")
    ax.set_xlabel("unseen video (sorted by error)")
    ax.set_ylabel("mean keypoint error (px)")
    ax.set_title("Per-video reconstruction error at r=20 (unseen videos)")
    ax.set_xticks(range(len(rows)))
    ax.legend()
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "q2_per_video.png", dpi=150)

    print("\nwrote results/q2.json, figures/q2_*.png")


if __name__ == "__main__":
    main()
