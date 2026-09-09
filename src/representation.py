import json
from pathlib import Path

import numpy as np
from pose_format import Pose

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
POSE_DIR = DATA / "poses"

# MediaPipe component order inside the stored keypoints.
OFF_BODY = 0     
OFF_FACE = 33    
OFF_LEFT = 501   
OFF_RIGHT = 522  
OFF_WORLD = 543 

BODY_IDX = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]
FACE_IDX = [
    61,                  # Mouthright
    291,                 # Mouthleft
    17,                  # LipsLowerOuter
    0,                   # LipsUpperOuter
    70, 105, 107,        # RightEyebrowUpper
    300, 334, 336,       # LeftEyebrowUpper
    161, 158,            # RightEyeUpper
    33, 163, 153, 133,   # RightEyeLower
    388, 385,            # LeftEyeUpper
    263, 390, 380, 362,  # LeftEyeLower
    9,                   # Nosetop
]

# Hands
LEFT_IDX = list(range(21))
RIGHT_IDX = list(range(21))

# Absolute positions of the 76 selected points
KEYPOINTS = (
    [OFF_BODY + i for i in BODY_IDX]
    + [OFF_FACE + i for i in FACE_IDX]
    + [OFF_LEFT + i for i in LEFT_IDX]
    + [OFF_RIGHT + i for i in RIGHT_IDX]
)

N_POINTS = len(KEYPOINTS)   # 76
D = N_POINTS * 2            # 152

N_HELD_OUT_VIDEOS = 20
SPLIT_SEED = 0

assert N_POINTS == 76, N_POINTS
assert len(set(KEYPOINTS)) == 76


def load_sequence(path):
    """Read one .pose file and return its pose vectors, shape (T, 152)."""
    pose = Pose.read(Path(path).read_bytes())
    data = pose.body.data                      
    arr = np.asarray(np.ma.getdata(data))      
    arr = arr[:, 0, :, :2]                     
    arr = arr[:, KEYPOINTS, :]                 
    return arr.reshape(arr.shape[0], D).astype(np.float64)


def load_all(verbose=True):
    """
    Returns a list of dicts: {file, video_id, X} with X of shape (T, 152).
    """
    manifest = json.loads((DATA / "manifest.json").read_text())
    seqs = []
    for i, m in enumerate(manifest):
        X = load_sequence(POSE_DIR / m["file"])
        seqs.append({"file": m["file"], "video_id": m["video_id"], "X": X})
        if verbose and (i + 1) % 50 == 0:
            print("  loaded %d/%d" % (i + 1, len(manifest)))
    return seqs


def split_by_video(seqs):
    """
    Split sequences into train / test, grouped by video_id.
    """
    videos = sorted({s["video_id"] for s in seqs})
    rng = np.random.RandomState(SPLIT_SEED)
    held = set(rng.choice(videos, N_HELD_OUT_VIDEOS, replace=False))
    train = [s for s in seqs if s["video_id"] not in held]
    test = [s for s in seqs if s["video_id"] in held]
    return train, test


def stack(seqs):
    return np.concatenate([s["X"] for s in seqs], axis=0).T


def describe(seqs, name):
    K = sum(s["X"].shape[0] for s in seqs)
    v = len({s["video_id"] for s in seqs})
    print("%-10s %3d sequences  %2d videos  %6d frames" % (name, len(seqs), v, K))
    return K
