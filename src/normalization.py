"""
Follow-up: does the low-dimensional structure survive once translation and
scale are removed, and how does a velocity representation compare?

Three representations, each put through the same analysis:

    raw         the original uncentered SVD on stored coordinates (baseline)
    normalized  per frame: subtract the shoulder midpoint, divide by the
                sequence's median shoulder width; SVD is mean-centered
    velocity    frame-to-frame differences of the normalized pose;
                SVD is mean-centered

Missing keypoints (stored as zeros) are filled from the nearest frame in which
that keypoint was detected, as in PoseStitch-SLT. Without this, a dropped hand
would become a large spurious offset after centering.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import representation as rep
from q3_structure import (MAX_LEN, content_words, downsample, jaccard,
                          pairwise_dtw)
from q4_failures import raw_confidence

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

R = 20
L_SHOULDER, R_SHOULDER = 5, 6   # positions within the 76 selected points


def fill_missing(P, present):
    """Fill undetected keypoints from the nearest frame where they exist.

    P is (T, 76, 2), present is (T, 76). A keypoint never detected in the
    sequence is left at NaN and resolved after centering.
    """
    P = P.copy()
    T = P.shape[0]
    for k in range(P.shape[1]):
        ok = np.flatnonzero(present[:, k])
        if len(ok) == 0:
            P[:, k] = np.nan
            continue
        if len(ok) == T:
            continue
        # index of the nearest valid frame for every frame
        pos = np.searchsorted(ok, np.arange(T))
        left = ok[np.clip(pos - 1, 0, len(ok) - 1)]
        right = ok[np.clip(pos, 0, len(ok) - 1)]
        t = np.arange(T)
        nearest = np.where(np.abs(t - left) <= np.abs(right - t), left, right)
        P[:, k] = P[nearest, k]
    return P


def normalize(X, conf):
    """(T, 152) raw pose -> (T, 152) centered and scale-normalized pose."""
    P = X.reshape(-1, rep.N_POINTS, 2)
    P = fill_missing(P, conf > 0)
    mid = (P[:, L_SHOULDER] + P[:, R_SHOULDER]) / 2
    width = np.linalg.norm(P[:, L_SHOULDER] - P[:, R_SHOULDER], axis=1)
    scale = np.nanmedian(width)
    P = (P - mid[:, None, :]) / scale
    P = np.nan_to_num(P, nan=0.0)   # never-detected points sit at the origin
    return P.reshape(-1, rep.D), scale


def fit_basis(M, center):
    """SVD of M (D, K). Returns U, S and the mean that was subtracted."""
    mu = M.mean(axis=1, keepdims=True) if center else np.zeros((M.shape[0], 1))
    U, S, _ = np.linalg.svd(M - mu, full_matrices=False)
    return U, S, mu


def captured(M, U, mu, r):
    """Fraction of (centered) energy that the rank-r basis captures."""
    C = M - mu
    Z = U[:, :r].T @ C
    return float((Z ** 2).sum() / (C ** 2).sum())


def keypoint_error(M, U, mu, r, scales):
    """Mean keypoint reconstruction error, in shoulder widths."""
    C = M - mu
    resid = C - U[:, :r] @ (U[:, :r].T @ C)
    per_pt = np.sqrt((resid.reshape(rep.N_POINTS, 2, -1) ** 2).sum(axis=1))
    return float((per_pt / scales[None, :]).mean())


def zscore(Z):
    """Per-sequence, per-component standardisation.

    Without it, DTW on velocity makes the least-moving clips (trajectories near
    zero) the nearest neighbour of almost every other clip.
    """
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-8)


def local_scaling(Dm, k=10):
    """Hubness reduction (Zelnik-Manor & Perona 2004; Schnitzer et al. 2012).

    Each distance is divided by the geometric mean of both clips' distance to
    their k-th nearest neighbour, so a clip that is close to everything no
    longer counts as close to anything in particular. Results are stable for
    k = 5, 10 and 20.
    """
    s = np.sort(Dm, axis=1)[:, k - 1]
    return Dm / np.sqrt(s[:, None] * s[None, :])


def nn_analysis(Dm, videos, wordsets):
    """Same-video nearest-neighbour rate and word overlap, with chance levels."""
    n = len(videos)
    nn = np.argsort(Dm, axis=1)[:, 0]
    iu = np.triu_indices(n, 1)
    same = np.array([videos[a] == videos[b] for a, b in zip(*iu)])
    jac = np.array([jaccard(wordsets[a], wordsets[b]) for a, b in zip(*iu)])
    chance_text = float(jac[~same].mean())

    rate = float(np.mean([videos[nn[i]] == videos[i] for i in range(n)]))
    cross = [i for i in range(n) if videos[nn[i]] != videos[i]]
    cross_text = float(np.mean([jaccard(wordsets[i], wordsets[nn[i]])
                                for i in cross]))
    d = Dm[iu]
    return {
        "same_video_nn_rate": rate,
        "chance_same_video": float(same.mean()),
        "lift_over_chance": rate / float(same.mean()),
        "cross_video_text_jaccard": cross_text,
        "chance_text_jaccard": chance_text,
        "cross_video_text_lift": cross_text / chance_text,
        "distinct_nearest_neighbours": int(len(set(nn.tolist()))),
        "median_dtw_same_over_different": float(
            np.median(d[same]) / np.median(d[~same])),
    }


def main():
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    basis = np.load(RESULTS / "basis.npz", allow_pickle=True)
    train_videos = set(basis["train_videos"].tolist())

    print("loading sequences...")
    seqs = rep.load_all(verbose=False)
    for s in seqs:
        conf = raw_confidence(rep.POSE_DIR / s["file"])
        s["N"], s["scale"] = normalize(s["X"], conf)
        s["V"] = np.diff(s["N"], axis=0)

    train = [s for s in seqs if s["video_id"] in train_videos]
    test = [s for s in seqs if s["video_id"] not in train_videos]
    videos = [s["video_id"] for s in seqs]

    text = pd.read_csv(rep.DATA / "iSign_v1.1.csv")
    uid2text = dict(zip(text.uid.astype(str), text.text.astype(str)))
    wordsets = [content_words(uid2text.get(Path(s["file"]).stem, ""))
                for s in seqs]

    def frames(group, key):
        return np.concatenate([s[key] for s in group], axis=0).T

    def scales(group, key):
        # raw residuals are pixels; normalized residuals are already in
        # shoulder widths, so they are not rescaled
        return np.concatenate([np.full(len(s[key]),
                                       s["scale"] if key == "X" else 1.0)
                               for s in group])

    reps = {"raw": ("X", False), "normalized": ("N", True),
            "velocity": ("V", True)}
    ranks = [1, 2, 3, 5, 10, 20, 40]
    out = {}
    curves = {}
    old = np.load(RESULTS / "q3_distances.npz", allow_pickle=True)

    for name, (key, center) in reps.items():
        print("\n=== %s ===" % name)
        M_tr, M_te = frames(train, key), frames(test, key)
        U, S, mu = fit_basis(M_tr, center)
        cum = np.cumsum(S ** 2) / (S ** 2).sum()
        curves[name] = cum

        res = {
            "mean_centered_svd": center,
            "variance_explained_train": {r: float(cum[r - 1]) for r in ranks},
            "variance_explained_unseen": {
                r: captured(M_te, U, mu, r) for r in ranks},
            "r_for_90_95_99": [int(np.searchsorted(cum, q) + 1)
                               for q in (0.90, 0.95, 0.99)],
        }
        if name != "velocity":
            res["keypoint_error_shoulder_widths"] = {
                split: {r: keypoint_error(M, U, mu, r, scales(g, key))
                        for r in (5, 10, 20)}
                for split, M, g in [("train", M_tr, train),
                                    ("unseen", M_te, test)]}

        if name == "raw":
            Dm = old["D_0"]          # identical to the original Q3 run
        else:
            cache = RESULTS / ("dtw_%s.npy" % name)
            if cache.is_file():
                Dm = np.load(cache)
            else:
                trajs = [downsample((s[key] - mu.T) @ U[:, :R], MAX_LEN)
                         for s in seqs]
                if name == "velocity":
                    trajs = [zscore(t) for t in trajs]
                Dm = pairwise_dtw(trajs)
                np.save(cache, Dm)
        res["nearest_neighbour"] = nn_analysis(Dm, videos, wordsets)
        res["nearest_neighbour_hubness_corrected"] = nn_analysis(
            local_scaling(Dm), videos, wordsets)
        out[name] = res

        ve = res["variance_explained_train"]
        nn = res["nearest_neighbour"]
        print("  variance explained  r=1 %.2f%%  r=5 %.2f%%  r=20 %.2f%%"
              % (100 * ve[1], 100 * ve[5], 100 * ve[20]))
        print("  components for 90/95/99%%: %s" % res["r_for_90_95_99"])
        print("  unseen variance at r=20: %.2f%%"
              % (100 * res["variance_explained_unseen"][20]))
        if "keypoint_error_shoulder_widths" in res:
            e = res["keypoint_error_shoulder_widths"]
            print("  keypoint error r=20 (shoulder widths): train %.3f  "
                  "unseen %.3f" % (e["train"][20], e["unseen"][20]))
        print("  same-video NN rate %.2f%% (chance %.2f%%, %.1fx)"
              % (100 * nn["same_video_nn_rate"],
                 100 * nn["chance_same_video"], nn["lift_over_chance"]))
        print("  cross-video word overlap %.2fx chance"
              % nn["cross_video_text_lift"])
        print("  distinct nearest neighbours: %d / %d"
              % (nn["distinct_nearest_neighbours"], len(seqs)))
        hc = res["nearest_neighbour_hubness_corrected"]
        print("  hubness-corrected: same-video %.1fx, cross-video words "
              "%.2fx, distinct NN %d" % (hc["lift_over_chance"],
                                         hc["cross_video_text_lift"],
                                         hc["distinct_nearest_neighbours"]))

    (RESULTS / "normalization.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, cum in curves.items():
        ax[0].plot(range(1, 41), 100 * cum[:40], marker=".",
                   label="raw (uncentered)" if name == "raw" else name)
    ax[0].set_xlabel("r")
    ax[0].set_ylabel("variance explained (%)")
    ax[0].set_title("Variance explained vs r")
    ax[0].grid(alpha=.3)
    ax[0].legend()

    names = list(out)
    x = np.arange(len(names))
    hc = "nearest_neighbour_hubness_corrected"
    ax[1].bar(x - .2, [out[n][hc]["lift_over_chance"]
                       for n in names], .4, label="same video")
    ax[1].bar(x + .2, [out[n][hc]["cross_video_text_lift"]
                       for n in names], .4, label="shared words (cross-video)")
    ax[1].axhline(1, color="k", ls="--", lw=1)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names)
    ax[1].set_ylabel("times chance")
    ax[1].set_title("What the nearest trajectory shares\n"
                    "(hubness-corrected, local scaling k = 10)",
                    fontsize=10)
    ax[1].grid(alpha=.3, axis="y")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "normalization.png", dpi=150)
    print("\nwrote results/normalization.json, figures/normalization.png")


if __name__ == "__main__":
    main()
