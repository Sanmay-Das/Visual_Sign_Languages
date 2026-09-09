# Tokenization in Visual Languages: a shared SVD subspace on iSign

Code for the prerequisite task: does sign-language pose data contain a
low-dimensional structure, discoverable without linguistic annotation, that
could serve as a tokenization space?

A single shared, uncentered SVD basis is fitted on pose observations from
80 iSign videos and applied unchanged to 20 unseen videos. All numbers and
figures in the research note are produced by the scripts here.

## Where each part lives

| Part | File |
|---|---|
| Data loading (download + `.pose` reading) | `src/fetch_data.py`, `src/representation.py` |
| Preprocessing / representation | `src/representation.py` |
| SVD computation, spectrum, reconstruction | `src/q1_dimensionality.py` |
| Reconstruction on unseen videos | `src/q2_generalization.py` |
| DTW and nearest-neighbour analysis | `src/q3_structure.py` |
| Failure-case diagnostics | `src/q4_failures.py` |
| Pose/skeleton plots | `src/visualise_poses.py` |

## Data is not included

The iSign dataset is gated, and its terms state it must not be shared further
or uploaded elsewhere. `data/poses/` and `data/iSign_v1.1.csv` are therefore
excluded from this repository. `data/manifest.json` is included: it lists the
exact 300 files used, so the subset can be reproduced precisely.

To obtain the data:

1. Request access at https://huggingface.co/datasets/Exploration-Lab/iSign
   and accept the terms.
2. Authenticate with a **read** token:

   ```
   huggingface-cli login
   ```

   or set `HF_TOKEN`. No token is stored in this repository; it is read at
   runtime from the environment or the `huggingface-cli` cache.

3. Run the fetch script (below). It downloads ~670 MB, not the full dataset.

### How the fetch works

The iSign dataset page prescribes concatenating four part files into a single
~170 GB zip. That exceeded available disk here, and the task asks to "start
small enough to iterate quickly". `fetch_data.py` therefore treats the four
parts as one virtual file, reads the zip central directory from its end, and
issues HTTP range requests for only the members it keeps. The extracted
`.pose` files are byte-identical to those obtained via the documented route,
and they are read with `pose-format` exactly as the dataset page specifies.

Selection is deterministic (`SEED = 0`): 3 segments from each of 100 videos.

## Running

```
pip install -r requirements.txt
python src/fetch_data.py          # 300 .pose files + iSign_v1.1.csv
python src/q1_dimensionality.py   # must run first: writes results/basis.npz
python src/q2_generalization.py
python src/q3_structure.py        # ~4 minutes (pairwise DTW)
python src/q4_failures.py
python src/visualise_poses.py     # optional: skeleton plots
```

`q1` must run before the others: it fits the shared basis and saves it to
`results/basis.npz`. `q2`, `q3` and `q4` load that basis and never refit it.

Everything is seeded, so re-running reproduces identical numbers.

## Representation

Following the 76-keypoint layout named in the task and specified in
PoseStitch-SLT (Joshi et al. 2025, Appendix B.2):

- 21 + 21 hand, 11 body, 23 face, selected from MediaPipe Holistic's 576
- x and y only, so `x_t` is in `R^152`
- no normalisation and no centering: the SVD is taken on the raw pose
  observation matrix `M = [x_1, ..., x_K]`, as the task specifies
- missing keypoints are left exactly as stored in the file

Train/unseen splitting is by `video_id`, as the iSign dataset page recommends.

Note that PoseStitch interpolates low-confidence keypoints and normalises to
reduce inter-signer variance. Neither is done here, because the task
prescribes neither.

## Outputs

`results/` holds the numbers behind every claim in the note:

| File | Contents |
|---|---|
| `q1.json` | singular values, variance explained, reconstruction vs r |
| `q2.json` | train vs unseen curves, per-video breakdown |
| `q3.json` | nearest-neighbour rates, chance levels, DTW medians |
| `q3_content_control.json` | content signal restricted to cross-video neighbours |
| `q4_failures.json` | per-video pose statistics and training reference ranges |
| `basis.npz` | the shared basis `U`, singular values, and the video split |
| `q3_distances.npz` | the 300x300 DTW distance matrices |

`figures/` holds the corresponding plots.

## Notes on the implementation

- **DTW** is the standard three-way step pattern, evaluated one anti-diagonal
  at a time so the recurrence vectorises. It was verified to agree with the
  textbook row-by-row formulation to ten decimal places, and is ~11x faster.
  Distances are divided by `(n + m)` so sequences of different length remain
  comparable.
- **"Variance explained"** is reported as the task words it. Because the SVD
  is uncentered, this is the fraction of squared Frobenius norm captured at
  rank r. It should be read alongside the pixel reconstruction error: at
  r = 1 the basis captures 93.76% of variance while still being 45.4 px wrong,
  because the first component is essentially the mean pose.
