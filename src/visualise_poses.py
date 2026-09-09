import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import representation as rep

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

# positions inside the 76 selected points
BODY = list(range(0, 11))
FACE = list(range(11, 34))
LEFT = list(range(34, 55))
RIGHT = list(range(55, 76))

BODY_EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (0, 1), (0, 2), (1, 3), (2, 4)]

# standard MediaPipe hand topology
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4),
              (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16),
              (13, 17), (17, 18), (18, 19), (19, 20),
              (0, 17)]


def draw(ax, P, title=None):
    def seg(idx, edges, colour, lw=1.8):
        for a, b in edges:
            pa, pb = P[idx[a]], P[idx[b]]
            if np.all(pa != 0) and np.all(pb != 0):
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=colour, lw=lw)

    seg(BODY, BODY_EDGES, "tab:blue", 2.2)
    seg(LEFT, HAND_EDGES, "tab:red")
    seg(RIGHT, HAND_EDGES, "tab:green")

    for idx, colour, size in [(BODY, "tab:blue", 14), (FACE, "grey", 4),
                              (LEFT, "tab:red", 8), (RIGHT, "tab:green", 8)]:
        pts = P[idx]
        pts = pts[np.all(pts != 0, axis=1)]
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=size, c=colour, zorder=3)

    ax.set_aspect("equal")
    ax.invert_yaxis()          
    if title:
        ax.set_title(title, fontsize=9)


def frames_of(video_id, seqs):
    group = [s for s in seqs if s["video_id"] == video_id]
    X = np.concatenate([g["X"] for g in group], axis=0)
    return X.reshape(X.shape[0], rep.N_POINTS, 2)


def main():
    FIGURES.mkdir(exist_ok=True)
    print("loading sequences...")
    seqs = rep.load_all(verbose=False)

    s = seqs[0]
    P = s["X"].reshape(-1, rep.N_POINTS, 2)
    picks = np.linspace(0, len(P) - 1, 6).astype(int)
    fig, axes = plt.subplots(1, 6, figsize=(16, 3.4))
    for ax, t in zip(axes, picks):
        draw(ax, P[t], "frame %d" % t)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("%s" % s["file"], fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "poses_skeleton.png", dpi=150)
    print("wrote figures/poses_skeleton.png")

    normal, failure = "b2_Sx4lZhQ0-", "yb7efY9x78U-"
    A, B = frames_of(normal, seqs), frames_of(failure, seqs)

    allpts = np.concatenate([A.reshape(-1, 2), B.reshape(-1, 2)])
    allpts = allpts[np.all(allpts != 0, axis=1)]
    xlim = (allpts[:, 0].min() - 20, allpts[:, 0].max() + 20)
    ylim = (allpts[:, 1].min() - 20, allpts[:, 1].max() + 20)

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for row, (frames, name, err) in enumerate([
            (A, normal, "3.89 px  (typical)"),
            (B, failure, "9.17 px  (failure case)")]):
        picks = np.linspace(0, len(frames) - 1, 4).astype(int)
        for ax, t in zip(axes[row], picks):
            draw(ax, frames[t])
            ax.set_xlim(*xlim)
            ax.set_ylim(ylim[1], ylim[0])
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row][0].set_ylabel("%s\n%s" % (name, err), fontsize=9)
    fig.suptitle("Same axes: a typical unseen video (top) vs a failure "
                 "video (bottom).", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "poses_scale.png", dpi=150)
    print("wrote figures/poses_scale.png")

    # both failure videos next to a typical one, all on the same axes
    rows = [("b2_Sx4lZhQ0-", "typical unseen", 3.89),
            ("yb7efY9x78U-", "failure case", 9.17),
            ("zWRmXJnur9o-", "failure case", 10.09)]
    frames = {vid: frames_of(vid, seqs) for vid, _, _ in rows}

    allpts = np.concatenate([f.reshape(-1, 2) for f in frames.values()])
    allpts = allpts[np.all(allpts != 0, axis=1)]
    xlim = (allpts[:, 0].min() - 20, allpts[:, 0].max() + 20)
    ylim = (allpts[:, 1].min() - 20, allpts[:, 1].max() + 20)

    fig, axes = plt.subplots(len(rows), 5, figsize=(17, 10))
    for row, (vid, label, err) in enumerate(rows):
        f = frames[vid]
        for ax, t in zip(axes[row], np.linspace(0, len(f) - 1, 5).astype(int)):
            draw(ax, f[t])
            ax.set_xlim(*xlim)
            ax.set_ylim(ylim[1], ylim[0])
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row][0].set_ylabel("%s\n%s  %.2f px" % (vid, label, err),
                                fontsize=9)
    fig.suptitle("The two failure videos vs a typical one, identical axes",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "poses_failures.png", dpi=140)
    print("wrote figures/poses_failures.png")


if __name__ == "__main__":
    main()
