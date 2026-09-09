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

## Representation

Following the 76-keypoint layout named in the task and specified in
PoseStitch-SLT:

- 21 + 21 hand, 11 body, 23 face, selected from MediaPipe Holistic's 576
- x and y only, so `x_t` is in `R^152`
- no normalisation and no centering: the SVD is taken on the raw pose
  observation matrix `M = [x_1, ..., x_K]`, as the task specifies
- missing keypoints are left exactly as stored in the file

Train/unseen splitting is by `video_id`, as the iSign dataset page recommends.

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

`figures/` has the plots.

