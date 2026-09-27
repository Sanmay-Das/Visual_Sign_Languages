import json
import struct
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import fetch_data as fd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "signer_frames"
INDEX = DATA / ".cache" / "video_central_directory.json"

VIDEO_PARTS = ["iSign-videos_v1.1_part_aa", "iSign-videos_v1.1_part_ab"]


def video_index(rf):
    if INDEX.is_file():
        return json.loads(INDEX.read_text())
    cd_off, cd_size = fd.find_central_directory(rf)
    buf = bytearray()
    while len(buf) < cd_size:
        n = min(8 << 20, cd_size - len(buf))
        buf += rf.read(cd_off + len(buf), n)
    buf = bytes(buf)
    members, p = {}, 0
    while p + 46 <= len(buf) and buf[p:p + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", buf[p + 10:p + 12])[0]
        comp, uncomp = struct.unpack("<II", buf[p + 20:p + 28])
        n_name, n_extra, n_cmt = struct.unpack("<HHH", buf[p + 28:p + 34])
        offset = struct.unpack("<I", buf[p + 42:p + 46])[0]
        name = buf[p + 46:p + 46 + n_name].decode("utf-8", "replace")
        extra = buf[p + 46 + n_name:p + 46 + n_name + n_extra]
        comp, uncomp, offset = fd.parse_zip64_extra(extra, comp, uncomp, offset)
        if name.endswith(".mp4"):
            members[Path(name).stem] = {"name": name, "method": method,
                                        "comp": comp, "offset": offset}
        p += 46 + n_name + n_extra + n_cmt
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(members))
    return members


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fd.PARTS = VIDEO_PARTS
    rf = fd.ConcatRemote(fd.hf_token())
    members = video_index(rf)
    print("video members in archive: %d" % len(members))

    manifest = json.loads((DATA / "manifest.json").read_text())
    first = {}
    for m in manifest:
        first.setdefault(m["video_id"], Path(m["file"]).stem)

    for vid, stem in tqdm(sorted(first.items()), desc="frames"):
        out = OUT / (vid + ".jpg")
        if out.is_file() or stem not in members:
            continue
        tmp = Path(tempfile.gettempdir()) / ("isign_%s.mp4" % stem)
        tmp.write_bytes(fd.extract(rf, members[stem]))
        cap = cv2.VideoCapture(str(tmp))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
        ok, frame = cap.read()
        cap.release()
        tmp.unlink(missing_ok=True)
        if ok:
            cv2.imwrite(str(out), frame)

    vids = sorted(first)
    fig, axes = plt.subplots(10, 10, figsize=(22, 17))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(vids) and (OUT / (vids[i] + ".jpg")).is_file():
            img = cv2.cvtColor(cv2.imread(str(OUT / (vids[i] + ".jpg"))),
                               cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(str(i), fontsize=9)
    fig.tight_layout()
    fig.savefig(DATA / "signer_sheet.png", dpi=80)
    (DATA / "signer_sheet_order.json").write_text(json.dumps(vids, indent=1))
    print("frames: %d / %d" % (len(list(OUT.glob("*.jpg"))), len(vids)))


if __name__ == "__main__":
    main()
