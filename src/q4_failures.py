"""
Failure cases
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pose_format import Pose

import representation as rep

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

R = 20
OUTLIERS = ["zWRmXJnur9o-", "yb7efY9x78U-"]

# index ranges inside the 76 selected points
BODY = slice(0, 11)
FACE = slice(11, 34)
LEFT = slice(34, 55)
RIGHT = slice(55, 76)

L_SHOULDER, R_SHOULDER = 5, 6

def raw_confidence(path):
    pose = Pose.read(Path(path).read_bytes())
    conf = np.asarray(pose.body.confidence)[:, 0, :]   # (T, 576)
    return conf[:, rep.KEYPOINTS]


def describe_video(seqs, conf_by_file):
    X = np.concatenate([s["X"] for s in seqs], axis=0)      # (T, 152)
    P = X.reshape(X.shape[0], rep.N_POINTS, 2)              # (T, 76, 2)
    C = np.concatenate([conf_by_file[s["file"]] for s in seqs], axis=0)

    present = C > 0
    shoulders = P[:, [L_SHOULDER, R_SHOULDER], :]
    shoulder_width = np.linalg.norm(
        shoulders[:, 0, :] - shoulders[:, 1, :], axis=1)
    body_ok = present[:, BODY].all(axis=1)

    hands = P[:, LEFT.start:RIGHT.stop, :]
    motion = np.linalg.norm(np.diff(hands, axis=0), axis=2).mean()

    return {
        "frames": int(P.shape[0]),
        "left_hand_missing": float(1 - present[:, LEFT].mean()),
        "right_hand_missing": float(1 - present[:, RIGHT].mean()),
        "face_missing": float(1 - present[:, FACE].mean()),
        "body_missing": float(1 - present[:, BODY].mean()),
        "mean_x": float(P[..., 0].mean()),
        "mean_y": float(P[..., 1].mean()),
        "shoulder_width_px": float(np.median(shoulder_width[body_ok]))
        if body_ok.any() else float("nan"),
        "mean_hand_motion_px": float(motion),
    }


def main():
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    basis = np.load(RESULTS / "basis.npz", allow_pickle=True)
    U = basis["U"]
    Ur = U[:, :R]
    test_videos = set(basis["test_videos"].tolist())
    train_videos = set(basis["train_videos"].tolist())

    print("loading sequences...")
    seqs = rep.load_all()
    conf_by_file = {
        s["file"]: raw_confidence(rep.POSE_DIR / s["file"]) for s in seqs
    }

    by_video = {}
    for s in seqs:
        by_video.setdefault(s["video_id"], []).append(s)

    # reconstruction error per unseen video
    rows = []
    for vid in sorted(test_videos):
        group = by_video[vid]
        M = np.concatenate([g["X"] for g in group], axis=0).T
        resid = M - Ur @ (Ur.T @ M)
        per_pt = np.sqrt((resid.reshape(rep.N_POINTS, 2, -1) ** 2).sum(axis=1))
        stats = describe_video(group, conf_by_file)
        stats["video_id"] = vid
        stats["error_px"] = float(per_pt.mean())
        stats["is_outlier"] = vid in OUTLIERS
        rows.append(stats)

    train_stats = []
    for vid in sorted(train_videos):
        train_stats.append(describe_video(by_video[vid], conf_by_file))

    def ref(key):
        vals = [t[key] for t in train_stats]
        return float(np.median(vals)), float(np.percentile(vals, 5)), \
            float(np.percentile(vals, 95))

    keys = ["left_hand_missing", "right_hand_missing", "face_missing",
            "mean_x", "mean_y", "shoulder_width_px", "mean_hand_motion_px"]

    print("\n%-22s %8s | %s" % ("held-out video", "err px",
                                "  ".join("%14s" % k[:14] for k in keys)))
    print("-" * 130)
    for r in sorted(rows, key=lambda r: -r["error_px"]):
        mark = " <<<" if r["is_outlier"] else ""
        print("%-22s %8.2f | %s%s" % (
            r["video_id"], r["error_px"],
            "  ".join("%14.3f" % r[k] for k in keys), mark))

    print("\n%-22s %8s | %s" % ("TRAINING median", "-",
                                "  ".join("%14.3f" % ref(k)[0] for k in keys)))
    print("%-22s %8s | %s" % ("training  5th pct", "-",
                              "  ".join("%14.3f" % ref(k)[1] for k in keys)))
    print("%-22s %8s | %s" % ("training 95th pct", "-",
                              "  ".join("%14.3f" % ref(k)[2] for k in keys)))

    print("\nwhere the two outliers fall relative to the training videos:")
    for r in rows:
        if not r["is_outlier"]:
            continue
        print("\n  %s  (error %.2f px)" % (r["video_id"], r["error_px"]))
        for k in keys:
            med, lo, hi = ref(k)
            vals = [t[k] for t in train_stats]
            pct = float((np.array(vals) < r[k]).mean() * 100)
            flag = ""
            if r[k] < lo or r[k] > hi:
                flag = "   OUTSIDE the 5-95%% range of training videos"
            print("    %-22s %10.3f   training median %10.3f   "
                  "percentile %5.1f%s" % (k, r[k], med, pct, flag))

    (RESULTS / "q4_failures.json").write_text(json.dumps(
        {"held_out": rows,
         "training_reference": {k: dict(zip(["median", "p5", "p95"], ref(k)))
                                for k in keys}},
        indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, key, title in zip(
            ax,
            ["shoulder_width_px", "left_hand_missing", "mean_hand_motion_px"],
            ["Body scale (shoulder width)", "Left-hand missingness",
             "Mean hand motion"]):
        xs = [r[key] for r in rows]
        ys = [r["error_px"] for r in rows]
        cols = ["tab:red" if r["is_outlier"] else "tab:blue" for r in rows]
        a.scatter(xs, ys, c=cols)
        med, lo, hi = ref(key)
        a.axvline(med, color="k", ls="--", lw=1, label="training median")
        a.axvspan(lo, hi, color="grey", alpha=.15, label="training 5-95%")
        for r in rows:
            if r["is_outlier"]:
                a.annotate(r["video_id"][:8], (r[key], r["error_px"]),
                           fontsize=7, xytext=(4, 4),
                           textcoords="offset points")
        a.set_xlabel(title)
        a.set_ylabel("unseen-video error at r=%d (px)" % R)
        a.grid(alpha=.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "q4_failures.png", dpi=150)
    print("\nwrote results/q4_failures.json, figures/q4_failures.png")


if __name__ == "__main__":
    main()
