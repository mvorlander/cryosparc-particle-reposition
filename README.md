# cryosparc-particle-reposition

`cryosparc-particle-reposition` is a local-first command-line tool that takes one or more CryoSPARC overlay sources and projects their per-particle signal back onto the original micrographs or onto denoised micrographs.

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

GIF generated from two particle-subset-derived reconstruct-only jobs overlaid onto a denoised micrograph. Black and red correspond to different particle scale-factor subsets.

![Particle subset scale-factor example](docs/media/j100_j101_particle_subset_scale_factor_example.blink.gif)

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

The preferred CLI name is `cryosparc-particle-reposition`. The older `cryosparc-2d-class-overlay` command remains available as a legacy alias.

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

- Python 3.10+ (Python 3.11 or 3.12 recommended)
- A local checkout or mounted path that can access the CryoSPARC project directory
- An existing **CryoSPARC Tools (`cryosparc-tools`, sometimes called cstools) installation in the same Python environment** as this tool. A Conda environment merely named `cstools` is not sufficient unless the package is installed in it.
- NumPy, SciPy, and Pillow (installed automatically with this package)
- Read access to job metadata, `.cs` datasets, referenced class stacks/3D maps, and micrographs; write access to your chosen output directory

Important:

- `cryosparc-tools` should match your CryoSPARC minor release.
- If your CryoSPARC is `5.0.x`, install `cryosparc-tools~=5.0.0`.
- If your CryoSPARC is `4.7.x`, install `cryosparc-tools~=4.7.0`.

See the official [CryoSPARC Tools installation instructions](https://tools.cryosparc.com/#installation), [Python environment guidance](https://tools.cryosparc.com/#python-environment), and [source repository](https://github.com/cryoem-uoft/cryosparc-tools).

Use a dedicated environment outside the CryoSPARC server installation. This tool only uses the dataset reader: no CryoSPARC server URL, credentials, API login, GPU, CUDA, Slurm, or site-specific software path is needed. Linux and macOS are the primary environments; on Windows use WSL with the project storage mounted there. Memory requirements depend on micrograph/map size and the 3D projection cache; start with one micrograph.

## Quick Start

### Option 1: use an existing CryoSPARC Tools environment

Activate your own virtualenv or Conda environment first (its name/location is up to you):

```bash
# For example: conda activate YOUR_ENVIRONMENT
python -c "from cryosparc.dataset import Dataset; import sys; print(sys.executable)"
python -m pip install "git+https://github.com/mvorlander/cryosparc-particle-reposition.git"
python -m cryosparc_2d_class_overlay --help
```

The Git URL installation requires Git. Alternatively clone/download this repository,
enter its directory, and run `python -m pip install .`.
The package deliberately does not select a `cryosparc-tools` version automatically:
your CryoSPARC minor release determines that version.

### Option 2: bootstrap script

Create a virtual environment, install the matching `cryosparc-tools`, and install this package:

```bash
git clone https://github.com/mvorlander/cryosparc-particle-reposition.git
cd cryosparc-particle-reposition
./scripts/bootstrap.sh --cryosparc-version 5.0
source .venv/bin/activate
cryosparc-particle-reposition --help
```

For a CryoSPARC `4.7.x` installation:

```bash
./scripts/bootstrap.sh --cryosparc-version 4.7
```

### Option 3: manual installation

From a clone/download of this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "cryosparc-tools~=5.0.0"
python -m pip install .
```

Replace `5.0.0` with the minor release that matches your CryoSPARC installation.

## Optional configuration file

Copy [examples/reposition.json](examples/reposition.json) to `reposition.local.json`,
edit the example paths, then run:

```bash
cp examples/reposition.json reposition.local.json
# Edit reposition.local.json to use your own job and output paths.
cryosparc-particle-reposition --config reposition.local.json
# Override settings for a quick test:
cryosparc-particle-reposition --config reposition.local.json --top-micrographs 1 --no-write-gifs
```

Configuration is plain JSON (no comments). Keys are CLI option names with underscores
instead of hyphens. All rendering options shown by `--help` are supported.
Repeatable options (`job_dir`, `subset`, `overlay_color`) use nonempty arrays;
flags use JSON booleans; numeric settings use JSON numbers. Unknown keys, invalid
types, and invalid choices are rejected. `config`, `help`, and `version` are not
configuration keys.

Precedence is **explicit CLI option > config > built-in default**. Repeated CLI
options replace the entire corresponding config array. Use the same option name
when overriding; legacy aliases retain their historical behavior.
Relative job/output directory paths in JSON resolve against the config file's
folder; CLI paths resolve against the current working directory. `~` is expanded;
environment variables are not interpolated. No config is loaded automatically.
`reposition.local.json` is git-ignored to keep local storage paths out of commits.
There is no cstools installation-path setting: select its Python environment instead.

## Storage and compute environments

Keep the CryoSPARC project directory layout (`project/Jxx/...`) intact. Dataset
paths relative to the project are resolved against each job's parent directory.
Absolute paths recorded in `.cs` files are used as recorded; this tool does not
rewrite them. When moving projects, make those locations available through your
mount/container bind configuration and preserve symlink targets. Copying only a
job folder may omit referenced upstream micrographs, stacks, or maps.

For read-only project mounts, always specify a writable `--output-dir`. For example:

```bash
cryosparc-particle-reposition --job-dir /mounted/project/J46 \
  --output-dir "$HOME/reposition-output" --max-micrographs 1
```

On a cluster, activate the environment inside your site's batch script and run the
same command on a CPU compute node. Request memory/time according to data size;
there are no built-in scheduler, module, queue, or Conda-prefix assumptions. In a
container, both the project and all external symlink targets must be accessible.

## Usage

### Single Select 2D job

```bash
cryosparc-particle-reposition \
  --job-dir /path/to/CS-project/J119
```

### Multiple Select 2D jobs overlaid into the same micrographs

```bash
cryosparc-particle-reposition \
  --job-dir /path/to/CS-project/J46 \
  --job-dir /path/to/CS-project/J98 \
  --overlay-color black \
  --overlay-color red
```

### Single 3D refinement job

```bash
cryosparc-particle-reposition \
  --job-dir /path/to/CS-project/J95
```

### 3D refinement job onto denoised micrographs

```bash
cryosparc-particle-reposition \
  --job-dir /path/to/CS-project/J95 \
  --denoise-job-dir /path/to/CS-project/J10 \
  --projection-angle-step-deg 5
```

### Render onto denoised micrographs

```bash
cryosparc-particle-reposition \
  --job-dir /path/to/CS-project/J46 \
  --denoise-job-dir /path/to/CS-project/J10
```

### Rank by the most balanced overlap across several jobs

```bash
cryosparc-particle-reposition \
  --job-dir /path/to/CS-project/J46 \
  --job-dir /path/to/CS-project/J98 \
  --job-dir /path/to/CS-project/J95 \
  --top-micrographs 10 \
  --top-micrographs-mode balanced
```

### Mix 2D and 3D sources in one render

```bash
cryosparc-particle-reposition \
  --job-dir /path/to/CS-project/J46 \
  --job-dir /path/to/CS-project/J98 \
  --job-dir /path/to/CS-project/J95 \
  --overlay-color black \
  --overlay-color red \
  --overlay-color cyan
```

### Disable GIF output

```bash
cryosparc-particle-reposition \
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

## Troubleshooting

- **Cannot import `cryosparc.dataset`:** activate the environment containing
  CryoSPARC Tools, install the matching version there, and use
  `python -m cryosparc_2d_class_overlay` to ensure the same interpreter is used.
  Check `python -m pip show cryosparc-tools` and `python -m pip check`.
- **Missing `job.json`, `.cs`, MRC, or symlink target:** check the full project
  layout and storage mounts described above. An exported particle dataset alone
  is not a supported job directory.
- **Permission denied for output:** set `--output-dir` to a writable location.
- **Memory/time pressure:** begin with `--max-micrographs 1`; use a nonzero
  `--projection-angle-step-deg` for cached 3D projections. Output downsampling
  reduces image sizes but does not eliminate full-resolution processing memory.

## Development

Run the CLI module directly:

```bash
python -m cryosparc_2d_class_overlay --help
```

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Contributor and coding-agent guidance: [AGENTS.md](AGENTS.md).
