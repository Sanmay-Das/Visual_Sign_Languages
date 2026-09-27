# Tokenization in Visual Languages

## Directory Details

| Part | File |
|---|---|
| Data loading (download + `.pose` reading) | `src/fetch_data.py`, `src/representation.py` |
| Preprocessing / representation | `src/representation.py` |
| SVD computation, spectrum, reconstruction | `src/q1_dimensionality.py` |
| Reconstruction on unseen videos | `src/q2_generalization.py` |
| DTW and nearest-neighbour analysis | `src/q3_structure.py` |
| Failure-case diagnostics | `src/q4_failures.py` |
| Pose/skeleton plots | `src/visualise_poses.py` |
| Normalization and velocity | `src/normalization.py` |
| Short-window recurrence | `src/motifs.py` |
| CISLR pose extraction | `src/cislr_poses.py` |
| CISLR same-word retrieval | `src/cislr_retrieval.py` |
| One frame per iSign video (signer identification) | `src/signer_frames.py` |
| Cross-signer tests | `src/signers.py` |

## Data

The iSign dataset is gated, and its terms state that it must not be shared
further or uploaded elsewhere. `data/poses/` and `data/iSign_v1.1.csv` are
therefore excluded from this repository. `data/manifest.json` is included:
it lists the exact 300 files used.

To obtain the data:

1. Request access at
   [huggingface.co/datasets/Exploration-Lab/iSign](https://huggingface.co/datasets/Exploration-Lab/iSign)
   and accept the terms.

2. Authenticate with a **read** token:

   ```bash
   huggingface-cli login
   ```

   or set `HF_TOKEN`. No token is stored in this repository.

3. Run `python src/fetch_data.py`. It downloads ~695 MB, not the full dataset.

### How the fetch works

The iSign dataset page prescribes concatenating the four pose part files into
a single zip:

```bash
cat iSign-poses_v1.1_part_aa iSign-poses_v1.1_part_ab iSign-poses_v1.1_part_ac iSign-poses_v1.1_part_ad > iSign-poses_v1.1.zip
```

That zip is ~170 GB, which exceeded available disk here, and the task asks to
"start small enough to iterate quickly". `fetch_data.py` therefore treats the
four parts as one virtual file, reads the zip central directory from its end,
and issues HTTP range requests for only the members it keeps.

The extracted `.pose` files are byte-identical to those obtained via the
documented route, and they are read with `pose-format` exactly as the dataset
page specifies.

### CISLR

Request access at [huggingface.co/datasets/Exploration-Lab/CISLR](https://huggingface.co/datasets/Exploration-Lab/CISLR). `src/cislr_poses.py` downloads the videos and `dataset.csv` on first run. CISLR ships videos only, so poses
are extracted with MediaPipe Holistic, the estimator behind iSign's poses.
The subset is 50 words with at least 4 videos each, 4 videos per word
(200 clips, seeded).

### Signer frames

iSign has no signer labels, but its paper states that each video contains a
single signer. `src/signer_frames.py` fetches one segment per video from
iSign's own video archive and saves its middle frame. Videos were
grouped conservatively into female- and male-presenting signers, excluding
two where no signer was visible; the labels are in
`results/signer_groups.json`. The frames are not redistributed.

## Running

```
pip install -r requirements.txt
python src/fetch_data.py          
python src/q1_dimensionality.py   
python src/q2_generalization.py
python src/q3_structure.py        
python src/q4_failures.py
python src/visualise_poses.py     
```

`q1` must run before the others. It fits the shared basis and saves it to
`results/basis.npz`. `q2`, `q3` and `q4` load that basis and never refit it.

### Follow-up experiments

```
python src/normalization.py      # needs q1 and q3; writes the DTW matrices signers.py uses
python src/motifs.py
python src/cislr_poses.py        
python src/cislr_retrieval.py    # needs cislr_poses.py
python src/signer_frames.py      
python src/signers.py           # needs normalization.py
```

## Representation

Following the 76-keypoint layout named in the task and specified in
PoseStitch-SLT:

- 21 + 21 hand, 11 body, 23 face, selected from MediaPipe Holistic's 576
- x and y only, so `x_t` is in `R^152`
- no normalisation and no centering: the SVD is taken on the raw pose
  observation matrix `M = [x_1, ..., x_K]`, as the task specifies
- missing keypoints are left exactly as stored in the file

Train/unseen splitting is by `video_id`, as the iSign dataset page recommends.

The above describes the original analysis (`q1`-`q4`). The follow-up
experiments (`normalization.py` onwards) instead:

- center each frame on the shoulder midpoint and divide by the sequence's
  median shoulder width
- use a mean-centered SVD (i.e. PCA)
- fill missing keypoints from the nearest frame in which they were detected,
  as in PoseStitch-SLT
- compute velocity as the frame-to-frame difference of the normalized pose
- use local scaling (k = 10) in nearest-neighbour comparisons to correct for
  hubness

## Outputs

`results/` has the output:

| File | Contents |
|---|---|
| `q1.json` | singular values, variance explained, reconstruction vs r |
| `q2.json` | train vs unseen curves, per-video breakdown |
| `q3.json` | nearest-neighbour rates, chance levels, DTW medians |
| `q3_content_control.json` | content signal restricted to cross-video neighbours |
| `q4_failures.json` | per-video pose statistics and training reference ranges |
| `basis.npz` | the shared basis `U`, singular values, and the video split |
| `q3_distances.npz` | the 300x300 DTW distance matrices |
| `normalization.json` | raw vs normalized vs velocity: variance, generalization, nearest neighbours |
| `motifs.json` | short-window recurrence, forward vs reversed, with confidence intervals |
| `cislr_retrieval.json` | CISLR same-word retrieval and AUC per representation |
| `signers.json` | new signer vs new recording, windows across signers, signer information in the space |
| `signer_groups.json` | signer group per iSign video (from visual inspection) |

`figures/` has the plots. Those for the follow-up experiments are
`normalization.png`, `motifs.png`, `cislr_retrieval.png` and
`cislr_trajectories.png`.

