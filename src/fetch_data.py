"""
Small subset of the iSign pose dataset.
"""

import json
import os
import random
import struct
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

REPO = "Exploration-Lab/iSign"
BASE = "https://huggingface.co/datasets/" + REPO + "/resolve/main"
PARTS = [
    "iSign-poses_v1.1_part_aa",
    "iSign-poses_v1.1_part_ab",
    "iSign-poses_v1.1_part_ac",
    "iSign-poses_v1.1_part_ad",
]
TEXT_CSV = "iSign_v1.1.csv"

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
POSE_DIR = DATA / "poses"
CACHE = DATA / ".cache"

N_VIDEOS = 100
N_PER_VIDEO = 3
SEED = 0


# auth
def hf_token():
    """Read the token from the environment or the huggingface-cli cache."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    for p in [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]:
        if p.is_file():
            return p.read_text().strip()
    sys.exit(
        "No Hugging Face token found. Run `huggingface-cli login` or set "
        "HF_TOKEN. A read token is sufficient."
    )


class ConcatRemote:

    def __init__(self, token):
        self.session = requests.Session()
        self.session.headers["Authorization"] = "Bearer " + token
        self.sizes = [self._size(p) for p in PARTS]
        self.starts = []
        off = 0
        for s in self.sizes:
            self.starts.append(off)
            off += s
        self.total = off

    def _size(self, name):
        r = self.session.head(BASE + "/" + name, allow_redirects=True, timeout=60)
        r.raise_for_status()
        return int(r.headers["Content-Length"])

    def read(self, offset, length):
        """Read `length` bytes starting at `offset` in the virtual file."""
        if offset < 0 or offset + length > self.total:
            raise ValueError("range outside virtual file")
        out = bytearray()
        remaining = length
        pos = offset
        while remaining > 0:
            i = max(j for j in range(len(PARTS)) if self.starts[j] <= pos)
            local = pos - self.starts[i]
            take = min(remaining, self.sizes[i] - local)
            headers = {"Range": "bytes=%d-%d" % (local, local + take - 1)}
            r = self.session.get(
                BASE + "/" + PARTS[i], headers=headers, timeout=300
            )
            r.raise_for_status()
            chunk = r.content
            if len(chunk) != take:
                raise IOError(
                    "short read: wanted %d, got %d (server may not support "
                    "range requests)" % (take, len(chunk))
                )
            out += chunk
            pos += take
            remaining -= take
        return bytes(out)

def find_central_directory(rf):
    tail_len = min(1 << 16, rf.total)
    tail = rf.read(rf.total - tail_len, tail_len)

    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise IOError("end-of-central-directory record not found")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])

    j = tail.rfind(b"PK\x06\x07")
    if j >= 0:
        z64_off = struct.unpack("<Q", tail[j + 8:j + 16])[0]
        rec = rf.read(z64_off, 56)
        if rec[:4] != b"PK\x06\x06":
            raise IOError("bad ZIP64 end-of-central-directory record")
        cd_size, cd_off = struct.unpack("<QQ", rec[40:56])

    return cd_off, cd_size


def parse_zip64_extra(extra, comp, uncomp, offset):
    p = 0
    while p + 4 <= len(extra):
        hid, hsz = struct.unpack("<HH", extra[p:p + 4])
        body = extra[p + 4:p + 4 + hsz]
        if hid == 0x0001:
            q = 0
            if uncomp == 0xFFFFFFFF and q + 8 <= len(body):
                uncomp = struct.unpack("<Q", body[q:q + 8])[0]
                q += 8
            if comp == 0xFFFFFFFF and q + 8 <= len(body):
                comp = struct.unpack("<Q", body[q:q + 8])[0]
                q += 8
            if offset == 0xFFFFFFFF and q + 8 <= len(body):
                offset = struct.unpack("<Q", body[q:q + 8])[0]
                q += 8
            break
        p += 4 + hsz
    return comp, uncomp, offset


def read_central_directory(rf):
    cache = CACHE / "central_directory.json"
    if cache.is_file():
        return json.loads(cache.read_text())

    cd_off, cd_size = find_central_directory(rf)
    print("central directory: %.1f MB at offset %d" % (cd_size / 1e6, cd_off))

    buf = bytearray()
    step = 8 << 20
    with tqdm(total=cd_size, unit="B", unit_scale=True,
              desc="central directory") as bar:
        while len(buf) < cd_size:
            n = min(step, cd_size - len(buf))
            buf += rf.read(cd_off + len(buf), n)
            bar.update(n)
    buf = bytes(buf)

    members = []
    p = 0
    while p + 46 <= len(buf) and buf[p:p + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", buf[p + 10:p + 12])[0]
        comp, uncomp = struct.unpack("<II", buf[p + 20:p + 28])
        n_name, n_extra, n_cmt = struct.unpack("<HHH", buf[p + 28:p + 34])
        offset = struct.unpack("<I", buf[p + 42:p + 46])[0]
        name = buf[p + 46:p + 46 + n_name].decode("utf-8", "replace")
        extra = buf[p + 46 + n_name:p + 46 + n_name + n_extra]
        comp, uncomp, offset = parse_zip64_extra(extra, comp, uncomp, offset)
        if name.endswith(".pose"):
            members.append({
                "name": name,
                "method": method,
                "comp": comp,
                "uncomp": uncomp,
                "offset": offset,
            })
        p += 46 + n_name + n_extra + n_cmt

    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(members))
    print("found %d .pose members" % len(members))
    return members


def extract(rf, m):
    head = rf.read(m["offset"], 30)
    if head[:4] != b"PK\x03\x04":
        raise IOError("bad local header for " + m["name"])
    n_name, n_extra = struct.unpack("<HH", head[26:30])
    data_at = m["offset"] + 30 + n_name + n_extra
    raw = rf.read(data_at, m["comp"])
    if m["method"] == 0:
        return raw
    if m["method"] == 8:
        return zlib.decompressobj(-zlib.MAX_WBITS).decompress(raw)
    raise IOError("unsupported compression method %d" % m["method"])


# subset selection
def video_id(name):
    stem = Path(name).stem
    return stem.rsplit("-", 1)[0] if "-" in stem[1:] else stem


def choose(members):
    by_video = defaultdict(list)
    for m in members:
        by_video[video_id(m["name"])].append(m)

    eligible = sorted(v for v, ms in by_video.items() if len(ms) >= N_PER_VIDEO)
    if len(eligible) < N_VIDEOS:
        raise SystemExit(
            "only %d videos have >= %d segments" % (len(eligible), N_PER_VIDEO)
        )

    rng = random.Random(SEED)
    picked = sorted(rng.sample(eligible, N_VIDEOS))

    chosen = []
    for v in picked:
        ms = sorted(by_video[v], key=lambda m: m["name"])
        chosen.extend(ms[:N_PER_VIDEO])
    return chosen


def fetch_text_csv(token):
    out = DATA / TEXT_CSV
    if out.is_file():
        return out
    r = requests.get(
        BASE + "/" + TEXT_CSV,
        headers={"Authorization": "Bearer " + token},
        timeout=300,
    )
    r.raise_for_status()
    DATA.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    print("saved %s (%.1f MB)" % (out.name, len(r.content) / 1e6))
    return out


def main():
    token = hf_token()
    DATA.mkdir(parents=True, exist_ok=True)
    POSE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    fetch_text_csv(token)

    rf = ConcatRemote(token)
    print("virtual archive: %.1f GB across %d parts" % (rf.total / 1e9, len(PARTS)))

    members = read_central_directory(rf)
    chosen = choose(members)
    print("selected %d sequences from %d videos (%.0f MB to download)" % (
        len(chosen), N_VIDEOS, sum(m["comp"] for m in chosen) / 1e6))

    manifest = []
    for m in tqdm(chosen, desc="poses"):
        out = POSE_DIR / Path(m["name"]).name
        if not out.is_file():
            out.write_bytes(extract(rf, m))
        manifest.append({
            "file": out.name,
            "video_id": video_id(m["name"]),
            "member": m["name"],
            "bytes": out.stat().st_size,
        })

    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nwrote %d .pose files to %s" % (len(manifest), POSE_DIR))
    print("manifest: %s" % (DATA / "manifest.json"))


if __name__ == "__main__":
    main()
