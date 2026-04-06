# cryosparc-2d-class-overlay

`cryosparc-2d-class-overlay` is a local-first command-line tool that takes one or more CryoSPARC overlay sources and projects their per-particle signal back onto the original micrographs or onto denoised micrographs.

Supported source types:

- CryoSPARC `select_2D` jobs, using the chosen 2D class averages
- CryoSPARC 3D refinement or reconstruct-only jobs such as `homo_refine_new`, `nonuniform_refine_new`, `new_local_refine`, and `homo_reconstruct`, using per-particle backprojections from the refined volume

It is the inverse visualization of a normal 2D classification or 3D refinement workflow:

- CryoSPARC stores, for each selected particle, the class assignment, in-plane rotation, translation, and micrograph coordinates.
- For refinement jobs, CryoSPARC stores per-particle 3D poses, shifts, and a refined 3D map.
- This tool reads those fields from the CryoSPARC `.cs` files and stamps either the selected class averages or per-particle 3D map backprojections back onto the corresponding micrographs.
- The result is a per-micrograph overlay PNG, a synthetic-only PNG, and a blink GIF by default.

This repository is designed to run locally on any machine with Python and filesystem access to the CryoSPARC project directory. It does not require Slurm and it does not call the CryoSPARC API.

## Example Output

GIF generated from two `select_2D` jobs overlaid onto a denoised micrograph, one in black and one in white.



![Example blink overlay](docs/media/j46_j98_j10_denoised_overlay_example.blink.gif)

## Acknowledgements and Prior Work

This repository is a new, independent implementation of the ReconSil concept for CryoSPARC overlay sources.

It is conceptually inspired by the ReconSil method described by Thomas C. R. Miller and colleagues in:

- Miller, T. C. R. et al. "Mechanism of head-to-head MCM double-hexamer formation revealed by cryo-EM." Nature 575, 704-710 (2019).
  https://www.nature.com/articles/s41586-019-1768-0
- Greiwe, J. F. et al. "In silico reconstitution of DNA replication. Lessons from single-molecule imaging and cryo-tomography applied to single-particle cryo-EM." Current Opinion in Structural Biology 72, 279-286 (2022).
  https://www.sciencedirect.com/science/chapter/bookseries/pii/S0076687922000830
  PubMed: https://pubmed.ncbi.nlm.nih.gov/35026552/

The particle re-projection step in this repository was also informed by the general approach used in RELION's `particle_reposition.cpp`:

- RELION source: https://github.com/3dem/relion/blob/master/src/apps/particle_reposition.cpp

This repository does not contain the original ReconSil source code. It should be understood as a fresh implementation of the same general idea, adapted for CryoSPARC `.cs` datasets, local Python environments, and direct command-line use.

## Features

- Works on CryoSPARC `select_2D` jobs and supported 3D refinement jobs directly from disk
- Supports one or more overlay sources at the same time
- Adds an automatic `JXX` color legend to multi-source overlay PNGs and GIFs
- Supports rendering onto denoised micrographs from a CryoSPARC denoise job
- Supports per-particle 3D refinement-map backprojections
- Supports cached 3D backprojections by quantized angular bins for speed
- Uses `alignments3D/object_pose` and `alignments3D/object_shift` automatically for CryoSPARC `new_local_refine` jobs
- Writes:
  - `.overlay.png`
  - `.synthetic.png`
  - `.blink.gif` by default
  - optional `.count.png`
- Can rank micrographs by particle abundance
- Includes a balanced ranking mode for multi-job overlays when one job is much rarer than the others
- Uses an auto-contrast synthetic background by default so black overlays remain visible

## Requirements

- Python 3.8+
- A local checkout or mounted path that can access the CryoSPARC project directory
- `cryosparc-tools` installed in the same Python environment

Important:

- `cryosparc-tools` should match your CryoSPARC minor release.
- If your CryoSPARC is `5.0.x`, install `cryosparc-tools~=5.0.0`.
- If your CryoSPARC is `4.7.x`, install `cryosparc-tools~=4.7.0`.

This follows the official CryoSPARC Tools guidance.

## Quick Start

### Option 1: bootstrap script

Create a virtual environment, install the matching `cryosparc-tools`, and install this package:

```bash
git clone https://github.com/mvorlander/cryosparc-2d-class-overlay.git
cd cryosparc-2d-class-overlay
./scripts/bootstrap.sh --cryosparc-version 5.0
source .venv/bin/activate
cryosparc-2d-class-overlay --help
```

For a CryoSPARC `4.7.x` installation:

```bash
./scripts/bootstrap.sh --cryosparc-version 4.7
```

### Option 2: manual installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "cryosparc-tools~=5.0.0"
python -m pip install .
```

Replace `5.0.0` with the minor release that matches your CryoSPARC installation.

## Usage

### Single Select 2D job

```bash
cryosparc-2d-class-overlay \
  --job-dir /path/to/CS-project/J119
```

### Multiple Select 2D jobs overlaid into the same micrographs

```bash
cryosparc-2d-class-overlay \
  --job-dir /path/to/CS-project/J46 \
  --job-dir /path/to/CS-project/J98 \
  --overlay-color black \
  --overlay-color red
```

### Single 3D refinement job

```bash
cryosparc-2d-class-overlay \
  --job-dir /path/to/CS-project/J95
```

### 3D refinement job onto denoised micrographs

```bash
cryosparc-2d-class-overlay \
  --job-dir /path/to/CS-project/J95 \
  --denoise-job-dir /path/to/CS-project/J10 \
  --projection-angle-step-deg 5
```

### Render onto denoised micrographs

```bash
cryosparc-2d-class-overlay \
  --job-dir /path/to/CS-project/J46 \
  --denoise-job-dir /path/to/CS-project/J10
```

### Rank by the most balanced overlap across several jobs

```bash
cryosparc-2d-class-overlay \
  --job-dir /path/to/CS-project/J46 \
  --job-dir /path/to/CS-project/J98 \
  --job-dir /path/to/CS-project/J95 \
  --top-micrographs 10 \
  --top-micrographs-mode balanced
```

### Mix 2D and 3D sources in one render

```bash
cryosparc-2d-class-overlay \
  --job-dir /path/to/CS-project/J46 \
  --job-dir /path/to/CS-project/J98 \
  --job-dir /path/to/CS-project/J95 \
  --overlay-color black \
  --overlay-color red \
  --overlay-color cyan
```

### Disable GIF output

```bash
cryosparc-2d-class-overlay \
  --job-dir /path/to/CS-project/J46 \
  --no-write-gifs
```

## Output

By default the tool writes into:

```text
<job-dir>/<subset>_2d_class_overlay
```

If additional `--job-dir` sources are used, their job names are appended in order.
If `--denoise-job-dir` is used, the denoise job name is appended.
If any source is a 3D refinement job, the default base folder changes to `particle_reprojection_overlay` instead of `<subset>_2d_class_overlay`.

Each rendered micrograph produces:

- `<micrograph>.overlay.png`
- `<micrograph>.synthetic.png`
- `<micrograph>.blink.gif`
- optional `<micrograph>.count.png`

The output folder also contains `overlay_summary.tsv` with total and per-job particle counts.

## Notes

- This tool is file-based and does not require CryoSPARC credentials.
- The project directory must be readable from the machine where you run the command.
- For the tested CryoSPARC outputs, the best-matching transform convention is:
  - rotate by `+alignments2D/pose`
  - shift by `-alignments2D/shift`
- For tested CryoSPARC refinement outputs, the best-matching 3D convention is:
  - rotate by `-alignments3D/pose`
  - shift by `-alignments3D/shift`
- For tested CryoSPARC `new_local_refine` outputs, the reprojection should use `alignments3D/object_pose` and `alignments3D/object_shift` rather than the generic `alignments3D/pose` and `alignments3D/shift`.
- For 3D refinement sources, `--projection-angle-step-deg` controls an on-demand projection cache. The default `5` degree binning is much faster than exact per-particle projection; set it to `0` to disable quantization.
- PNG and GIF outputs are full resolution by default. Use `--png-downsample` or `--gif-downsample` only when you explicitly want smaller review files.
- CryoSPARC motion-corrected micrographs may use MRC mode `12` half-floats; this is supported.

## Development

Run the CLI module directly:

```bash
python -m cryosparc_2d_class_overlay --help
```

Install development dependencies:

```bash
python -m pip install -e .[dev]
```

Run tests:

```bash
pytest
```
