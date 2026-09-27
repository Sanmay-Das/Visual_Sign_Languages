import json
from pathlib import Path

import numpy as np
import pandas as pd

import representation as rep
from q4_failures import raw_confidence
from normalization import fit_basis, local_scaling, normalize
from motifs import STRIDE, WIN, windows

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
R = 20
SEED = 0


def evaluate(M, U, mu, r, per_video):
    C = M - mu
    Z = U[:, :r].T @ C
    resid = C - U[:, :r] @ Z
    per_pt = np.sqrt((resid.reshape(rep.N_POINTS, 2, -1) ** 2).sum(axis=1))
    return float((Z ** 2).sum() / (C ** 2).sum()), per_pt.mean(axis=0)


def video_bootstrap(values_by_video, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    keys = list(values_by_video)
    boot = [np.mean(np.concatenate([values_by_video[k]
                                    for k in rng.choice(keys, len(keys))]))
            for _ in range(n)]
    return [float(x) for x in np.percentile(boot, [2.5, 97.5])]


def main():
    groups = json.loads((RESULTS / "signer_groups.json").read_text())
    seqs = [s for s in rep.load_all(verbose=False)
            if groups.get(s["video_id"]) in ("F", "M")]
    for s in seqs:
        s["N"], _ = normalize(s["X"], raw_confidence(rep.POSE_DIR / s["file"]))
        s["g"] = groups[s["video_id"]]

    male = sorted({s["video_id"] for s in seqs if s["g"] == "M"})
    female = sorted({s["video_id"] for s in seqs if s["g"] == "F"})
    rng = np.random.RandomState(SEED)
    held_m = set(rng.choice(male, len(female), replace=False))
    fit_v = set(male) - held_m
    print("fit on %d male-presenting videos; test on %d held-out male and "
          "%d female-presenting videos" % (len(fit_v), len(held_m), len(female)))

    out = {"n_fit_videos": len(fit_v), "n_heldout_male": len(held_m),
           "n_female": len(female), "excluded_unidentifiable": 2}

    Mfit = np.concatenate([s["N"] for s in seqs if s["video_id"] in fit_v]).T
    U, _, mu = fit_basis(Mfit, center=True)
    t1 = {}
    for name, vids in [("new recording (held-out male)", held_m),
                       ("new signer (female)", set(female))]:
        g = [s for s in seqs if s["video_id"] in vids]
        M = np.concatenate([s["N"] for s in g]).T
        row = {}
        for r in (10, 20):
            ve, err = evaluate(M, U, mu, r, None)
            row["variance_explained_r%d" % r] = ve
            row["keypoint_error_r%d" % r] = float(err.mean())
        by_v = {}
        for s in g:
            _, e = evaluate(s["N"].T, U, mu, 20, None)
            by_v.setdefault(s["video_id"], []).append(e)
        by_v = {k: np.concatenate(v) for k, v in by_v.items()}
        row["keypoint_error_r20_95ci_by_video"] = video_bootstrap(by_v)
        t1[name] = row
        print("  %-30s var r=20 %.2f%%  error r=20 %.4f sw  (95%% CI %.4f-%.4f)"
              % (name, 100 * row["variance_explained_r20"],
                 row["keypoint_error_r20"],
                 *row["keypoint_error_r20_95ci_by_video"]))
    out["new_signer_vs_new_recording"] = t1

    basis = np.load(RESULTS / "basis.npz", allow_pickle=True)
    train_v = set(basis["train_videos"].tolist())
    Mtr = np.concatenate([s["N"] for s in seqs if s["video_id"] in train_v]).T
    U2, _, mu2 = fit_basis(Mtr, center=True)
    W = [(w, v, s["g"]) for s in seqs
         for w, v in windows((s["N"] - mu2.T) @ U2[:, :R], s["video_id"])]
    X = np.stack([w for w, _, _ in W]).astype(np.float32)
    V = np.array([v for _, v, _ in W])
    G = np.array([g for _, _, g in W])
    motion = np.linalg.norm(np.diff(X, axis=1), axis=2).sum(1)
    keep = motion >= np.quantile(motion, 0.5)
    X, V, G = X[keep], V[keep], G[keep]

    q = G == "F"
    Q = X[q].reshape(q.sum(), -1)
    fwd = X[~q].reshape((~q).sum(), -1)
    rev = X[~q][:, ::-1].reshape((~q).sum(), -1)

    def nearest(Qm, Bm, chunk=512):
        best = np.empty(len(Qm), dtype=np.float32)
        Bsq = (Bm ** 2).sum(1)
        for i in range(0, len(Qm), chunk):
            qq = Qm[i:i + chunk]
            d2 = (qq ** 2).sum(1)[:, None] + Bsq[None, :] - 2 * qq @ Bm.T
            best[i:i + chunk] = np.sqrt(np.maximum(d2.min(1), 0))
        return best

    d_f, d_r = nearest(Q, fwd), nearest(Q, rev)
    closer = d_f < d_r
    qv = V[q]
    by_v = {v: closer[qv == v].astype(float) for v in np.unique(qv)}
    t2 = {"windows_female_queries": int(q.sum()),
          "windows_male_candidates": int((~q).sum()),
          "fraction_forward_closer": float(closer.mean()),
          "fraction_forward_closer_95ci_by_video": video_bootstrap(by_v),
          "window_frames": WIN, "stride": STRIDE}
    out["windows_across_signers"] = t2
    print("  windows, female matched only to male: forward closer %.1f%% "
          "(95%% CI %.1f-%.1f)" % (100 * t2["fraction_forward_closer"],
                                    *[100 * x for x in
                                      t2["fraction_forward_closer_95ci_by_video"]]))

    man = json.loads((rep.DATA / "manifest.json").read_text())
    all_g = [groups.get(m["video_id"], "X") for m in man]
    all_v = [m["video_id"] for m in man]
    t3 = {}
    for name, D in [("raw", np.load(RESULTS / "q3_distances.npz")["D_0"]),
                    ("normalized", np.load(RESULTS / "dtw_normalized.npy"))]:
        D = local_scaling(D).copy()
        for i in range(len(all_v)):      
            D[i, [j for j in range(len(all_v)) if all_v[j] == all_v[i]]] = np.inf
        idx = [i for i in range(len(all_g)) if all_g[i] in ("F", "M")]
        nn = np.argmin(D[np.ix_(idx, idx)], axis=1)
        gi = np.array([all_g[i] for i in idx])
        vi = np.array([all_v[i] for i in idx])
        hit = (gi[nn] == gi).astype(float)
        rate = float(hit.mean())
        pF = np.mean(gi == "F")
        base = float(pF ** 2 + (1 - pF) ** 2)
        ci = video_bootstrap({v: hit[vi == v] for v in np.unique(vi)})
        t3[name] = {"nn_same_signer_group": rate, "base_rate": base,
                    "nn_same_signer_group_95ci_by_video": ci}
        print("  %-10s nearest (other-video) neighbour shares signer group "
              "%.1f%% (95%% CI %.1f-%.1f; base rate %.1f%%)"
              % (name, 100 * rate, 100 * ci[0], 100 * ci[1], 100 * base))
    out["signer_identity_in_space"] = t3

    (RESULTS / "signers.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/signers.json")


if __name__ == "__main__":
    main()
