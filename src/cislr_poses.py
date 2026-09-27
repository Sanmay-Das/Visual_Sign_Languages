import json
import os
import random
import tempfile
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CISLR = ROOT / "data" / "cislr"
OUT = CISLR / "poses"

N_WORDS = 50
PER_WORD = 4
SEED = 0


def fetch_cislr():
    import shutil
    import urllib.request
    from fetch_data import hf_token

    base = ("https://huggingface.co/datasets/Exploration-Lab/CISLR/"
            "resolve/main/")
    files = {"dataset.csv": "dataset.csv",
             "videos.zip": "CISLR_v1.5-a_videos/CISLR_v1.5-a_videos.zip"}
    CISLR.mkdir(parents=True, exist_ok=True)
    token = hf_token()
    for local, remote in files.items():
        out = CISLR / local
        if out.is_file():
            continue
        print("downloading %s ..." % remote)
        req = urllib.request.Request(
            base + remote, headers={"Authorization": "Bearer " + token})
        with urllib.request.urlopen(req) as r, open(out, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 22)


def choose():
    d = pd.read_csv(CISLR / "dataset.csv")
    counts = d.gloss.value_counts()
    eligible = sorted(counts[counts >= PER_WORD].index)
    words = sorted(random.Random(SEED).sample(eligible, N_WORDS))
    rows = []
    for w in words:
        uids = sorted(d.loc[d.gloss == w, "uid"])[:PER_WORD]
        rows += [{"uid": u, "gloss": w} for u in uids]
    return rows


def extract(row):
    import cv2
    import mediapipe as mp

    out = OUT / (row["uid"] + ".npz")
    if out.is_file():
        return row["uid"], "cached"

    z = zipfile.ZipFile(CISLR / "videos.zip")
    tmp = Path(tempfile.gettempdir()) / ("cislr_%s.mp4" % row["uid"])
    tmp.write_bytes(z.read("CISLR_v1.5-a_videos/%s.mp4" % row["uid"]))

    cap = cv2.VideoCapture(str(tmp))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    hol = mp.solutions.holistic.Holistic(static_image_mode=False,
                                         model_complexity=1)
    xy, conf = [], []
    parts = [("pose_landmarks", 33), ("face_landmarks", 468),
             ("left_hand_landmarks", 21), ("right_hand_landmarks", 21)]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        f_xy = np.zeros((543, 2))
        f_c = np.zeros(543)
        off = 0
        for attr, n in parts:
            lm = getattr(r, attr)
            if lm is not None:
                pts = np.array([[p.x * W, p.y * H] for p in lm.landmark[:n]])
                f_xy[off:off + n] = pts
                f_c[off:off + n] = 1.0
            off += n
        xy.append(f_xy)
        conf.append(f_c)
    cap.release()
    hol.close()
    tmp.unlink(missing_ok=True)

    np.savez_compressed(out, xy=np.array(xy), conf=np.array(conf), fps=fps)
    return row["uid"], "%d frames" % len(xy)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fetch_cislr()
    rows = choose()
    (CISLR / "manifest.json").write_text(json.dumps(rows, indent=2))
    print("%d clips, %d words" % (len(rows), len({r["gloss"] for r in rows})))

    workers = max(1, (os.cpu_count() or 2) - 4)
    with Pool(workers) as pool:
        for i, (uid, msg) in enumerate(pool.imap_unordered(extract, rows), 1):
            print("  [%3d/%d] %s  %s" % (i, len(rows), uid, msg), flush=True)
    print("done")


if __name__ == "__main__":
    main()
