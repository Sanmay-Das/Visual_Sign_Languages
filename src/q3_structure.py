"""
3. Do similar motions produce similar trajectories? Do different motions separate?
What actually determines the observed structure: motion, signer identity, pose configuration,
or something else?
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

import representation as rep

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

R = 20              
MAX_LEN = 120       
TOP_K = 5           

STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "and", "or", "but", "if",
    "it", "its", "this", "that", "these", "those", "as", "by", "from",
    "he", "she", "they", "we", "you", "i", "his", "her", "their", "our",
    "my", "your", "him", "them", "us", "me", "do", "does", "did", "so",
    "not", "no", "will", "would", "can", "could", "have", "has", "had",
}


# trajectories
def downsample(X, max_len):
    """Keep every k-th frame so no trajectory exceeds max_len."""
    k = int(np.ceil(X.shape[0] / max_len))
    return X[::k] if k > 1 else X


def dtw_distance(A, B):
    C = cdist(A, B, metric="euclidean")
    n, m = C.shape
    inf = np.inf

    d2 = np.full(n + 1, inf)   # diagonal k-2
    d1 = np.full(n + 1, inf)   # diagonal k-1
    d2[0] = 0.0                # D[0, 0]

    for k in range(2, n + m + 1):
        cur = np.full(n + 1, inf)
        lo = max(1, k - m)
        hi = min(n, k - 1)
        if lo <= hi:
            i = np.arange(lo, hi + 1)
            j = k - i
            cur[i] = C[i - 1, j - 1] + np.minimum(
                np.minimum(d1[i - 1], d1[i]), d2[i - 1])
        d2, d1 = d1, cur

    return float(d1[n] / (n + m))


def pairwise_dtw(trajs):
    """DTW distance matrix"""
    n = len(trajs)
    Dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(trajs[i], trajs[j])
            Dm[i, j] = Dm[j, i] = d
        if (i + 1) % 25 == 0:
            print("    row %d/%d" % (i + 1, n))
    np.fill_diagonal(Dm, np.inf)
    return Dm


def content_words(text):
    words = re.findall(r"[a-z']+", str(text).lower())
    return {w for w in words if w not in STOP and len(w) > 2}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# test
def analyse(Dm, videos, wordsets, label):
    n = len(videos)
    order = np.argsort(Dm, axis=1)

    same_video_at_1 = 0
    same_video_in_k = 0
    nn_text_sim = []
    for i in range(n):
        nn = order[i, 0]
        if videos[nn] == videos[i]:
            same_video_at_1 += 1
        if any(videos[j] == videos[i] for j in order[i, :TOP_K]):
            same_video_in_k += 1
        nn_text_sim.append(jaccard(wordsets[i], wordsets[nn]))

    iu = np.triu_indices(n, 1)
    same_video_pairs = np.array(
        [videos[a] == videos[b] for a, b in zip(*iu)], dtype=bool)
    chance_video = float(same_video_pairs.mean())
    all_text = np.array([jaccard(wordsets[a], wordsets[b])
                         for a, b in zip(*iu)])
    chance_text = float(all_text.mean())

    # median DTW distance
    d_pairs = Dm[iu]
    within = float(np.median(d_pairs[same_video_pairs]))
    between = float(np.median(d_pairs[~same_video_pairs]))

    out = {
        "label": label,
        "same_video_at_rank1": same_video_at_1 / n,
        "same_video_within_top%d" % TOP_K: same_video_in_k / n,
        "chance_same_video": chance_video,
        "lift_over_chance": (same_video_at_1 / n) / chance_video
        if chance_video else float("nan"),
        "mean_text_jaccard_at_rank1": float(np.mean(nn_text_sim)),
        "chance_text_jaccard": chance_text,
        "median_dtw_same_video": within,
        "median_dtw_different_video": between,
        "ratio_between_over_within": between / within if within else float("nan"),
    }

    print("\n--- %s ---" % label)
    print("  nearest neighbour shares video : %5.1f%%   (chance %4.2f%%, "
          "lift %.0fx)" % (100 * out["same_video_at_rank1"],
                           100 * chance_video, out["lift_over_chance"]))
    print("  same video within top-%d        : %5.1f%%" % (
        TOP_K, 100 * out["same_video_within_top%d" % TOP_K]))
    print("  text overlap of NN             : %.4f  (chance %.4f)" % (
        out["mean_text_jaccard_at_rank1"], chance_text))
    print("  median DTW  same video         : %.3f" % within)
    print("  median DTW  different video    : %.3f  (ratio %.2f)" % (
        between, out["ratio_between_over_within"]))
    return out


def main():
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    basis = np.load(RESULTS / "basis.npz", allow_pickle=True)
    U = basis["U"]

    print("loading sequences...")
    seqs = rep.load_all()

    import pandas as pd
    text = pd.read_csv(rep.DATA / "iSign_v1.1.csv")
    uid2text = dict(zip(text.uid.astype(str), text.text.astype(str)))

    videos = [s["video_id"] for s in seqs]
    uids = [Path(s["file"]).stem for s in seqs]
    missing = [u for u in uids if u not in uid2text]
    print("sequences without a text row: %d" % len(missing))
    wordsets = [content_words(uid2text.get(u, "")) for u in uids]

    Ur = U[:, :R]
    Z = [downsample((s["X"] @ Ur), MAX_LEN) for s in seqs]
    lens = [z.shape[0] for z in Z]
    print("trajectories: r=%d, lengths %d-%d frames after downsampling"
          % (R, min(lens), max(lens)))

    variants = {
        "all components (z1..z%d)" % R: [z for z in Z],
        "without u1 (z2..z%d)" % R: [z[:, 1:] for z in Z],
    }

    results = {}
    matrices = {}
    for label, trajs in variants.items():
        print("\ncomputing pairwise DTW: %s" % label)
        Dm = pairwise_dtw(trajs)
        matrices[label] = Dm
        results[label] = analyse(Dm, videos, wordsets, label)

    payload = {
        "r": R,
        "max_len": MAX_LEN,
        "top_k": TOP_K,
        "n_sequences": len(seqs),
        "n_videos": len(set(videos)),
        "metric": "length-normalised DTW, euclidean local cost",
        "variants": results,
    }
    (RESULTS / "q3.json").write_text(json.dumps(payload, indent=2))

    labels = list(results.keys())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    x = np.arange(len(labels))
    ax[0].bar(x - .2, [100 * results[l]["same_video_at_rank1"] for l in labels],
              .4, label="nearest neighbour shares video")
    ax[0].bar(x + .2, [100 * results[l]["chance_same_video"] for l in labels],
              .4, label="chance")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(["all\ncomponents", "without\nu1"])
    ax[0].set_ylabel("% of sequences")
    ax[0].set_title("Does the nearest trajectory come from the same video?")
    ax[0].legend()
    ax[0].grid(alpha=.3, axis="y")

    for l, Dm in matrices.items():
        iu = np.triu_indices(len(videos), 1)
        same = np.array([videos[a] == videos[b] for a, b in zip(*iu)])
        d = Dm[iu]
        ax[1].hist(d[same] / np.median(d[~same]), bins=40, alpha=.5,
                   density=True, label="all components" if l.startswith("all") else "without u1")
    ax[1].set_xlabel("DTW distance / median between-video distance")
    ax[1].set_ylabel("density")
    ax[1].set_title("Same-video pair distances, relative to typical")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "q3_structure.png", dpi=150)

    np.savez_compressed(
        RESULTS / "q3_distances.npz",
        **{("D_%d" % i): m for i, m in enumerate(matrices.values())},
        videos=np.array(videos),
    )
    print("\nwrote results/q3.json, results/q3_distances.npz, "
          "figures/q3_structure.png")


if __name__ == "__main__":
    main()
