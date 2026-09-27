import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import representation as rep
from q3_structure import pairwise_dtw
from q4_failures import raw_confidence
from normalization import fit_basis, normalize, zscore

ROOT = Path(__file__).resolve().parents[1]
CISLR = ROOT / "data" / "cislr"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"


def source(uid):
    return uid[:11]


def load_cislr():
    rows = json.loads((CISLR / "manifest.json").read_text())
    clips = []
    for r in rows:
        f = CISLR / "poses" / (r["uid"] + ".npz")
        if not f.is_file():
            continue
        z = np.load(f)
        xy = z["xy"][:, rep.KEYPOINTS, :]
        conf = z["conf"][:, rep.KEYPOINTS]
        body = conf[:, :11] > 0
        if len(xy) < 5 or not body[:, 5:7].all(axis=1).any():
            continue   
        X = xy.reshape(len(xy), rep.D)
        N, _ = normalize(X, conf)
        clips.append({"uid": r["uid"], "gloss": r["gloss"], "X": X, "N": N})
    return clips


def near_duplicates(clips, thresh=0.02):
    R = [c["N"][np.linspace(0, len(c["N"]) - 1, 30).astype(int)] for c in clips]
    n = len(clips)
    dup = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if np.abs(R[i] - R[j]).mean() < thresh:
                dup[i, j] = dup[j, i] = True
    return dup


def retrieval(Dm, words, dup):
    n = len(words)
    words = np.array(words)
    Dm = Dm.copy()
    Dm[dup] = np.inf      
    same = words[:, None] == words[None, :]
    np.fill_diagonal(same, False)

    order = np.argsort(Dm, axis=1)
    p1 = float(np.mean(words[order[:, 0]] == words))
    hit3 = float(np.mean([(words[order[i, :3]] == words[i]).any()
                          for i in range(n)]))

    iu = np.triu_indices(n, 1)
    keep = ~dup[iu]
    d, s = Dm[iu][keep], same[iu][keep]
    pos, neg = d[s], d[~s]
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    auc = float(1 - (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                / (len(pos) * len(neg)))
    chance = float(same.sum(1).mean() / (n - 1))
    return {"p_at_1": p1, "hit_at_3": hit3,
            "chance_p_at_1": chance, "lift_p_at_1": p1 / chance,
            "auc_same_vs_different": auc,
            "distinct_nearest_neighbours": int(len(set(order[:, 0].tolist())))}


def main():
    FIGURES.mkdir(exist_ok=True)
    clips = load_cislr()
    words = [c["gloss"] for c in clips]
    sources = [source(c["uid"]) for c in clips]
    print("CISLR clips usable: %d, words: %d" % (len(clips), len(set(words))))

    n_same_source = sum(sources[i] == sources[j]
                        for i in range(len(clips))
                        for j in range(i + 1, len(clips)))
    dup = near_duplicates(clips)
    print("pairs sharing a YouTube source: %d | near-duplicate pairs "
          "excluded: %d" % (n_same_source, dup.sum() // 2))

    basis = np.load(RESULTS / "basis.npz", allow_pickle=True)
    train_videos = set(basis["train_videos"].tolist())
    seqs = [s for s in rep.load_all(verbose=False)
            if s["video_id"] in train_videos]
    for s in seqs:
        s["N"], _ = normalize(s["X"], raw_confidence(rep.POSE_DIR / s["file"]))
        s["V"] = np.diff(s["N"], axis=0)
    fit = {}
    for key, center in [("X", False), ("N", True), ("V", True)]:
        U, _, mu = fit_basis(
            np.concatenate([s[key] for s in seqs], axis=0).T, center)
        fit[key] = (U, mu)

    for c in clips:
        c["V"] = np.diff(c["N"], axis=0)

    def proj(key, r, c):
        U, mu = fit[key]
        return (c[key] - mu.T) @ U[:, :r]

    variants = {
        "raw, r=20":            lambda c: proj("X", 20, c),
        "normalized, full 152": lambda c: c["N"],
        "normalized, r=10":     lambda c: proj("N", 10, c),
        "normalized, r=20":     lambda c: proj("N", 20, c),
        "velocity, r=20":       lambda c: zscore(proj("V", 20, c)),
    }

    out = {"n_clips": len(clips), "n_words": len(set(words)),
           "pairs_sharing_youtube_source": int(n_same_source),
           "near_duplicate_pairs_excluded": int(dup.sum() // 2),
           "variants": {}}
    for name, f in variants.items():
        Dm = pairwise_dtw([f(c) for c in clips])
        res = retrieval(Dm, words, dup)
        if name == "normalized, r=20":
            D_example = Dm
        out["variants"][name] = res
        print("  %-22s P@1 %5.1f%% (%.1fx chance)  hit@3 %5.1f%%  AUC %.3f  "
              "distinct NN %d"
              % (name, 100 * res["p_at_1"], res["lift_p_at_1"],
                 100 * res["hit_at_3"], res["auc_same_vs_different"],
                 res["distinct_nearest_neighbours"]))

    (RESULTS / "cislr_retrieval.json").write_text(json.dumps(out, indent=2))

    names = list(out["variants"])
    x = np.arange(len(names))
    v = out["variants"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].bar(x, [100 * v[n]["p_at_1"] for n in names], .6,
              label="nearest clip is same word")
    ax[0].axhline(100 * v[names[0]]["chance_p_at_1"], color="k", ls="--",
                  lw=1, label="chance")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax[0].set_ylabel("% of clips")
    ax[0].set_title("CISLR: is the closest clip the same sign?")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3, axis="y")

    ax[1].bar(x, [v[n]["auc_same_vs_different"] for n in names])
    ax[1].axhline(.5, color="k", ls="--", lw=1)
    ax[1].set_ylim(.4, 1)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax[1].set_ylabel("AUC")
    ax[1].set_title("Same-word pairs closer than different-word pairs")
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "cislr_retrieval.png", dpi=150)

    D = D_example.copy()
    D[dup] = np.inf
    nn = np.argmin(D, axis=1)
    hits = [i for i in range(len(clips)) if words[nn[i]] == words[i]]
    hits.sort(key=lambda i: D[i, nn[i]])
    a = hits[len(hits) // 2]
    b = nn[a]
    others = [j for j in np.argsort(D[a]) if words[j] != words[a]]
    c = others[len(others) // 2]          
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for k, axk in enumerate(ax):
        for i, style, lab in [(a, "-", "%s (clip 1)" % words[a]),
                              (b, "--", "%s (clip 2)" % words[b]),
                              (c, ":", "%s" % words[c])]:
            Z = proj("N", 20, clips[i])
            axk.plot(np.linspace(0, 1, len(Z)), Z[:, k], style, lw=1.8,
                     label=lab)
        axk.set_xlabel("time (normalized)")
        axk.set_title("component %d" % (k + 1))
        axk.grid(alpha=.3)
    ax[0].set_ylabel("z (normalized space learned on iSign)")
    ax[0].legend(fontsize=8)
    fig.suptitle("A typical correct match: two clips of '%s' vs a different "
                 "word ('%s')" % (words[a], words[c]))
    fig.tight_layout()
    fig.savefig(FIGURES / "cislr_trajectories.png", dpi=150)
    print("\nwrote results/cislr_retrieval.json, figures/cislr_*.png")


if __name__ == "__main__":
    main()
