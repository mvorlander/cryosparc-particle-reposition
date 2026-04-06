#!/usr/bin/env python3
"""Overlay CryoSPARC Select 2D class averages back onto micrographs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageColor
from scipy import ndimage
from scipy.spatial.transform import Rotation

from . import __version__


DEFAULT_OVERLAY_COLOR = "black"
DEFAULT_SECONDARY_OVERLAY_COLOR = "red"
DEFAULT_SYNTHETIC_BACKGROUND_COLOR = "auto"
DEFAULT_MICROGRAPH_OPACITY = 1.0
DEFAULT_CLASS_OPACITY = 0.70
DEFAULT_MASK_RADIUS_FRACTION = 0.48
DEFAULT_POSE_SIGN = -1
DEFAULT_SHIFT_SIGN = -1
DEFAULT_IMBALANCE_WARNING_RATIO = 3.0
DEFAULT_TOP_MICROGRAPHS_MODE_SINGLE = "sum"
DEFAULT_TOP_MICROGRAPHS_MODE_MULTI = "balanced"
DEFAULT_TOP_MICROGRAPHS_MODES = ("sum", "min", "balanced")
DEFAULT_PROJECTION_ANGLE_STEP_DEG = 5.0
DEFAULT_PNG_DOWNSAMPLE = 1
DEFAULT_GIF_DOWNSAMPLE = 1
DEFAULT_GIF_FRAME_MS = 500
SELECT2D_JOB_TYPE = "select_2D"
LOCAL_REFINE_JOB_TYPE = "new_local_refine"
REFINE3D_JOB_TYPES = (
    "homo_refine_new",
    "nonuniform_refine_new",
    "new_local_refine",
    "homo_reconstruct",
)
SOURCE_KIND_SELECT2D = "select2d"
SOURCE_KIND_REFINE3D = "refine3d"


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def log(message: str) -> None:
    print(f"[cryosparc-2d-class-overlay] {message}")


def as_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_rgb_color(value: str) -> np.ndarray:
    try:
        rgb = ImageColor.getrgb(value)
    except ValueError as exc:
        fail(f"Invalid overlay color '{value}': {exc}")
    return np.asarray(rgb, dtype=np.float32) / 255.0


def resolve_synthetic_background_color(
    value: str,
    overlay_color_rgbs: list[np.ndarray],
) -> np.ndarray:
    if value != "auto":
        return parse_rgb_color(value)
    if not overlay_color_rgbs:
        return parse_rgb_color("white")
    luminances = [
        float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
        for rgb in overlay_color_rgbs
    ]
    background_name = "white" if min(luminances) < 0.45 else "black"
    return parse_rgb_color(background_name)


def load_dataset_class():
    try:
        from cryosparc.dataset import Dataset
    except ImportError as exc:
        fail(
            "cryosparc-tools is required to read CryoSPARC .cs datasets. "
            "Install a version that matches your CryoSPARC minor release, for example "
            "'pip install \"cryosparc-tools~=5.0.0\"' for CryoSPARC 5.0.x or "
            "'pip install \"cryosparc-tools~=4.7.0\"' for CryoSPARC 4.7.x."
        )
    return Dataset


def normalize_field_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return label or "job"


def resolve_path(project_dir: Path, value: str) -> Path:
    path = Path(as_text(value))
    if path.is_absolute():
        return path
    return project_dir / path


def load_job_metadata(job_dir: Path) -> dict:
    job_json = job_dir / "job.json"
    if not job_json.exists():
        fail(f"job.json not found in {job_dir}")
    with job_json.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def detect_source_kind(job_dir: Path) -> str:
    job_type = str(load_job_metadata(job_dir).get("type", ""))
    if job_type == SELECT2D_JOB_TYPE:
        return SOURCE_KIND_SELECT2D
    if job_type in REFINE3D_JOB_TYPES:
        return SOURCE_KIND_REFINE3D
    fail(
        f"{job_dir} has unsupported CryoSPARC job type {job_type!r}. "
        f"Supported types are {SELECT2D_JOB_TYPE!r} and {', '.join(REFINE3D_JOB_TYPES)}."
    )


def refinement_alignment_fields(job_type: str) -> tuple[str, str]:
    if job_type == LOCAL_REFINE_JOB_TYPE:
        return ("alignments3D/object_pose", "alignments3D/object_shift")
    return ("alignments3D/pose", "alignments3D/shift")


def source_kind_label(source_kind: str) -> str:
    if source_kind == SOURCE_KIND_SELECT2D:
        return "select_2D"
    if source_kind == SOURCE_KIND_REFINE3D:
        return "refine3D"
    return source_kind


def find_latest_job_file(job_dir: Path, basename: str) -> Path:
    iter_pattern = re.compile(rf"^{re.escape(job_dir.name)}_(\d+)_({re.escape(basename)})$")
    iter_candidates: list[tuple[int, Path]] = []
    for candidate in job_dir.iterdir():
        match = iter_pattern.fullmatch(candidate.name)
        if match:
            iter_candidates.append((int(match.group(1)), candidate))
    if iter_candidates:
        iter_candidates.sort(key=lambda item: item[0])
        return iter_candidates[-1][1]

    fallbacks = (
        job_dir / f"{job_dir.name}_{basename}",
        job_dir / basename,
    )
    for candidate in fallbacks:
        if candidate.exists():
            return candidate
    fail(f"Could not find {basename} in {job_dir}")


@dataclass(frozen=True)
class TemplateRecord:
    original_class_idx: int
    selected_stack_idx: int
    selected_stack_path: Path
    pixel_size_angstrom: float


@dataclass(frozen=True)
class ParticleRecord:
    class_idx: int | None
    pose_rad: float | None
    pose3d: tuple[float, float, float] | None
    shift_x_px: float
    shift_y_px: float
    align_pixel_size_angstrom: float
    micrograph_uid: int
    raw_micrograph_path: Path
    center_x_frac: float
    center_y_frac: float
    raw_micrograph_pixel_size_angstrom: float


@dataclass(frozen=True)
class OverlaySource:
    label: str
    job_dir: Path
    subset: str | None
    source_kind: str
    overlay_color_name: str
    overlay_color_rgb: np.ndarray
    templates: dict[int, TemplateRecord] | None
    volume: "VolumeRecord | None"
    particle_groups: dict[Path, list[ParticleRecord]]


@dataclass(frozen=True)
class VolumeRecord:
    map_path: Path
    pixel_size_angstrom: float


@dataclass(frozen=True)
class TargetMicrographRecord:
    path: Path
    pixel_size_angstrom: float


@dataclass(frozen=True)
class MrcHeader:
    nx: int
    ny: int
    nz: int
    mode: int
    data_offset: int


class MrcReader:
    DTYPE_BY_MODE = {
        0: np.int8,
        1: np.int16,
        2: np.float32,
        6: np.uint16,
        12: np.float16,
    }

    def __init__(self) -> None:
        self._headers: dict[Path, MrcHeader] = {}

    def header(self, path: Path) -> MrcHeader:
        path = path.resolve()
        header = self._headers.get(path)
        if header is not None:
            return header
        with path.open("rb") as handle:
            raw = handle.read(1024)
        if len(raw) < 1024:
            fail(f"MRC header too short: {path}")
        nx, ny, nz, mode = struct.unpack("<4i", raw[:16])
        nsymbt = struct.unpack("<i", raw[92:96])[0]
        header = MrcHeader(nx=nx, ny=ny, nz=nz, mode=mode, data_offset=1024 + nsymbt)
        self._headers[path] = header
        return header

    def read_slice(self, path: Path, index: int = 0) -> np.ndarray:
        header = self.header(path)
        dtype = self.DTYPE_BY_MODE.get(header.mode)
        if dtype is None:
            fail(
                f"Unsupported MRC mode {header.mode} in {path}; "
                f"supported modes are {', '.join(str(mode) for mode in sorted(self.DTYPE_BY_MODE))}."
            )
        if not (0 <= index < max(header.nz, 1)):
            fail(f"Slice index {index} is out of range for {path}.")
        bytes_per_value = np.dtype(dtype).itemsize
        offset = header.data_offset + index * header.nx * header.ny * bytes_per_value
        array = np.memmap(
            path,
            dtype=dtype,
            mode="r",
            offset=offset,
            shape=(header.ny, header.nx),
        )
        return np.array(array, dtype=np.float32, copy=True)

    def read_volume(self, path: Path) -> np.ndarray:
        header = self.header(path)
        dtype = self.DTYPE_BY_MODE.get(header.mode)
        if dtype is None:
            fail(
                f"Unsupported MRC mode {header.mode} in {path}; "
                f"supported modes are {', '.join(str(mode) for mode in sorted(self.DTYPE_BY_MODE))}."
            )
        array = np.memmap(
            path,
            dtype=dtype,
            mode="r",
            offset=header.data_offset,
            shape=(max(header.nz, 1), header.ny, header.nx),
        )
        return np.array(array, dtype=np.float32, copy=True)


def load_select2d_files(job_dir: Path, subset: str) -> tuple[Path, Path, Path]:
    particles_path = job_dir / f"particles_{subset}.cs"
    passthrough_path = job_dir / f"{job_dir.name}_passthrough_particles_{subset}.cs"
    templates_path = job_dir / f"templates_{subset}.cs"

    missing = [str(path) for path in (particles_path, passthrough_path, templates_path) if not path.exists()]
    if missing:
        fail(
            "Select 2D job is missing expected files for subset "
            f"'{subset}': {', '.join(missing)}"
        )
    return particles_path, passthrough_path, templates_path


def validate_select2d_job(job_dir: Path) -> None:
    job = load_job_metadata(job_dir)
    if job.get("type") != SELECT2D_JOB_TYPE:
        fail(f"{job_dir} is not a CryoSPARC select_2D job directory.")


def validate_denoise_job(job_dir: Path) -> None:
    job = load_job_metadata(job_dir)
    job_type = str(job.get("type", ""))
    if "denoise" not in job_type:
        fail(f"{job_dir} is not a CryoSPARC denoise job directory.")


def load_denoised_targets(denoise_job_dir: Path) -> dict[int, TargetMicrographRecord]:
    Dataset = load_dataset_class()
    passthrough_path = denoise_job_dir / f"{denoise_job_dir.name}_passthrough_exposures.cs"
    denoised_path = denoise_job_dir / "micrographs_denoised.cs"
    missing = [str(path) for path in (passthrough_path, denoised_path) if not path.exists()]
    if missing:
        fail(
            "Denoise job is missing expected files: "
            + ", ".join(missing)
        )

    passthrough = Dataset.load(str(passthrough_path))
    denoised = Dataset.load(str(denoised_path))
    if len(passthrough) != len(denoised):
        fail("Denoise passthrough and denoised output datasets have different row counts.")
    if not np.array_equal(np.asarray(passthrough["uid"]), np.asarray(denoised["uid"])):
        fail("Denoise passthrough and denoised output datasets are not aligned by UID.")

    required = {
        "uid",
        "micrograph_blob_denoised/path",
        "micrograph_blob_denoised/psize_A",
    }
    missing_fields = sorted(required - set(denoised.fields()))
    if missing_fields:
        fail(
            "Denoised micrograph dataset is missing required fields: "
            + ", ".join(missing_fields)
        )

    project_dir = denoise_job_dir.parent
    targets: dict[int, TargetMicrographRecord] = {}
    for uid, path, pixel_size in zip(
        np.asarray(denoised["uid"]),
        np.asarray(denoised["micrograph_blob_denoised/path"]),
        np.asarray(denoised["micrograph_blob_denoised/psize_A"], dtype=np.float32),
        strict=True,
    ):
        targets[int(uid)] = TargetMicrographRecord(
            path=resolve_path(project_dir, as_text(path)),
            pixel_size_angstrom=float(pixel_size),
        )
    if not targets:
        fail(f"No denoised micrographs were found in {denoise_job_dir}")
    return targets


def load_templates(project_dir: Path, templates_path: Path) -> dict[int, TemplateRecord]:
    Dataset = load_dataset_class()
    dataset = Dataset.load(str(templates_path))

    if "blob_selected/path" in dataset.fields():
        selected_path_field = "blob_selected/path"
        selected_idx_field = "blob_selected/idx"
        psize_field = "blob_selected/psize_A"
    else:
        selected_path_field = "blob/path"
        selected_idx_field = "blob/idx"
        psize_field = "blob/psize_A"

    templates: dict[int, TemplateRecord] = {}
    original_indices = np.asarray(dataset["blob/idx"])
    selected_indices = np.asarray(dataset[selected_idx_field])
    selected_paths = np.asarray(dataset[selected_path_field])
    pixel_sizes = np.asarray(dataset[psize_field], dtype=np.float32)

    for original_idx, selected_idx, selected_path, pixel_size in zip(
        original_indices, selected_indices, selected_paths, pixel_sizes, strict=True
    ):
        original_class_idx = int(original_idx)
        templates[original_class_idx] = TemplateRecord(
            original_class_idx=original_class_idx,
            selected_stack_idx=int(selected_idx),
            selected_stack_path=resolve_path(project_dir, as_text(selected_path)),
            pixel_size_angstrom=float(pixel_size),
        )

    if not templates:
        fail(f"No templates were found in {templates_path}")
    return templates


def load_refine_files(job_dir: Path) -> tuple[Path, Path, Path]:
    particles_path = find_latest_job_file(job_dir, "particles.cs")
    passthrough_path = job_dir / f"{job_dir.name}_passthrough_particles.cs"
    volume_path = find_latest_job_file(job_dir, "volume_map.cs")
    missing = [str(path) for path in (particles_path, passthrough_path, volume_path) if not path.exists()]
    if missing:
        fail(
            "3D refinement job is missing expected files: "
            + ", ".join(missing)
        )
    return particles_path, passthrough_path, volume_path


def load_refinement_volume(project_dir: Path, volume_cs_path: Path) -> VolumeRecord:
    Dataset = load_dataset_class()
    dataset = Dataset.load(str(volume_cs_path))
    required = {"map/path", "map/psize_A"}
    missing = sorted(required - set(dataset.fields()))
    if missing:
        fail(
            f"Refinement volume dataset {volume_cs_path} is missing required fields: "
            + ", ".join(missing)
        )
    if len(dataset) != 1:
        fail(f"Expected exactly one volume row in {volume_cs_path}, found {len(dataset)}")
    return VolumeRecord(
        map_path=resolve_path(project_dir, as_text(dataset["map/path"][0])),
        pixel_size_angstrom=float(np.asarray(dataset["map/psize_A"], dtype=np.float32)[0]),
    )


def load_select2d_particles(
    project_dir: Path,
    particles_path: Path,
    passthrough_path: Path,
    templates: dict[int, TemplateRecord],
) -> dict[Path, list[ParticleRecord]]:
    Dataset = load_dataset_class()
    particles = Dataset.load(str(particles_path))
    passthrough = Dataset.load(str(passthrough_path))

    if len(particles) != len(passthrough):
        fail("Main and passthrough particle datasets have different row counts.")
    if not np.array_equal(np.asarray(particles["uid"]), np.asarray(passthrough["uid"])):
        fail("Main and passthrough particle datasets are not aligned by UID.")

    required_main = {
        "alignments2D/class",
        "alignments2D/pose",
        "alignments2D/shift",
        "alignments2D/psize_A",
    }
    required_pass = {
        "location/micrograph_path",
        "location/micrograph_uid",
        "location/center_x_frac",
        "location/center_y_frac",
        "location/micrograph_psize_A",
    }
    missing_main = sorted(required_main - set(particles.fields()))
    missing_pass = sorted(required_pass - set(passthrough.fields()))
    if missing_main:
        fail(f"Main particle dataset is missing required fields: {', '.join(missing_main)}")
    if missing_pass:
        fail(
            "Passthrough particle dataset is missing required fields: "
            + ", ".join(missing_pass)
        )

    particle_groups: dict[Path, list[ParticleRecord]] = defaultdict(list)
    class_indices = np.asarray(particles["alignments2D/class"], dtype=np.int64)
    poses = np.asarray(particles["alignments2D/pose"], dtype=np.float32)
    shifts = np.asarray(particles["alignments2D/shift"], dtype=np.float32)
    align_psizes = np.asarray(particles["alignments2D/psize_A"], dtype=np.float32)
    micro_uids = np.asarray(passthrough["location/micrograph_uid"], dtype=np.uint64)
    micro_paths = np.asarray(passthrough["location/micrograph_path"])
    micro_psizes = np.asarray(passthrough["location/micrograph_psize_A"], dtype=np.float32)
    center_x_frac = np.asarray(passthrough["location/center_x_frac"], dtype=np.float32)
    center_y_frac = np.asarray(passthrough["location/center_y_frac"], dtype=np.float32)

    missing_classes = set()
    for class_idx, pose, shift, align_psize, micro_uid, micro_path, micrograph_psize, x_frac, y_frac in zip(
        class_indices,
        poses,
        shifts,
        align_psizes,
        micro_uids,
        micro_paths,
        micro_psizes,
        center_x_frac,
        center_y_frac,
        strict=True,
    ):
        class_idx = int(class_idx)
        if class_idx not in templates:
            missing_classes.add(class_idx)
            continue

        particle_groups[resolve_path(project_dir, as_text(micro_path))].append(
            ParticleRecord(
                class_idx=class_idx,
                pose_rad=float(pose),
                pose3d=None,
                shift_x_px=float(shift[0]),
                shift_y_px=float(shift[1]),
                align_pixel_size_angstrom=float(align_psize),
                micrograph_uid=int(micro_uid),
                raw_micrograph_path=resolve_path(project_dir, as_text(micro_path)),
                center_x_frac=float(x_frac),
                center_y_frac=float(y_frac),
                raw_micrograph_pixel_size_angstrom=float(micrograph_psize),
            )
        )

    if missing_classes:
        preview = ", ".join(str(value) for value in sorted(missing_classes)[:10])
        fail(
            "Some particle class IDs are not present in the selected template set: "
            f"{preview}"
        )
    if not particle_groups:
        fail("No particles were loaded from the selected subset.")
    return particle_groups


def load_refinement_particles(
    project_dir: Path,
    particles_path: Path,
    passthrough_path: Path,
    pose_field: str,
    shift_field: str,
) -> dict[Path, list[ParticleRecord]]:
    Dataset = load_dataset_class()
    particles = Dataset.load(str(particles_path))
    passthrough = Dataset.load(str(passthrough_path))

    if len(particles) != len(passthrough):
        fail("Main and passthrough particle datasets have different row counts.")
    if not np.array_equal(np.asarray(particles["uid"]), np.asarray(passthrough["uid"])):
        fail("Main and passthrough particle datasets are not aligned by UID.")

    required_main = {
        pose_field,
        shift_field,
        "alignments3D/psize_A",
    }
    required_pass = {
        "location/micrograph_path",
        "location/micrograph_uid",
        "location/center_x_frac",
        "location/center_y_frac",
        "location/micrograph_psize_A",
    }
    missing_main = sorted(required_main - set(particles.fields()))
    missing_pass = sorted(required_pass - set(passthrough.fields()))
    if missing_main:
        fail(f"Main particle dataset is missing required fields: {', '.join(missing_main)}")
    if missing_pass:
        fail(
            "Passthrough particle dataset is missing required fields: "
            + ", ".join(missing_pass)
        )

    particle_groups: dict[Path, list[ParticleRecord]] = defaultdict(list)
    poses = np.asarray(particles[pose_field], dtype=np.float32)
    shifts = np.asarray(particles[shift_field], dtype=np.float32)
    align_psizes = np.asarray(particles["alignments3D/psize_A"], dtype=np.float32)
    micro_uids = np.asarray(passthrough["location/micrograph_uid"], dtype=np.uint64)
    micro_paths = np.asarray(passthrough["location/micrograph_path"])
    micro_psizes = np.asarray(passthrough["location/micrograph_psize_A"], dtype=np.float32)
    center_x_frac = np.asarray(passthrough["location/center_x_frac"], dtype=np.float32)
    center_y_frac = np.asarray(passthrough["location/center_y_frac"], dtype=np.float32)

    for pose, shift, align_psize, micro_uid, micro_path, micrograph_psize, x_frac, y_frac in zip(
        poses,
        shifts,
        align_psizes,
        micro_uids,
        micro_paths,
        micro_psizes,
        center_x_frac,
        center_y_frac,
        strict=True,
    ):
        particle_groups[resolve_path(project_dir, as_text(micro_path))].append(
            ParticleRecord(
                class_idx=None,
                pose_rad=None,
                pose3d=(float(pose[0]), float(pose[1]), float(pose[2])),
                shift_x_px=float(shift[0]),
                shift_y_px=float(shift[1]),
                align_pixel_size_angstrom=float(align_psize),
                micrograph_uid=int(micro_uid),
                raw_micrograph_path=resolve_path(project_dir, as_text(micro_path)),
                center_x_frac=float(x_frac),
                center_y_frac=float(y_frac),
                raw_micrograph_pixel_size_angstrom=float(micrograph_psize),
            )
        )

    if not particle_groups:
        fail("No particles were loaded from the refinement particle dataset.")
    return particle_groups


def load_overlay_source(job_dir: Path, subset: str, overlay_color_name: str) -> OverlaySource:
    job_metadata = load_job_metadata(job_dir)
    job_type = str(job_metadata.get("type", ""))
    source_kind = detect_source_kind(job_dir)
    project_dir = job_dir.parent
    templates = None
    volume = None
    if source_kind == SOURCE_KIND_SELECT2D:
        validate_select2d_job(job_dir)
        particles_path, passthrough_path, templates_path = load_select2d_files(job_dir, subset)
        log(
            f"Loading 2D particle data for {job_dir.name} from "
            f"{particles_path.name} and {passthrough_path.name}"
        )
        templates = load_templates(project_dir, templates_path)
        particle_groups = load_select2d_particles(project_dir, particles_path, passthrough_path, templates)
        source_subset: str | None = subset
    else:
        particles_path, passthrough_path, volume_cs_path = load_refine_files(job_dir)
        pose_field, shift_field = refinement_alignment_fields(job_type)
        log(
            f"Loading 3D refinement particle data for {job_dir.name} from "
            f"{particles_path.name}, {passthrough_path.name}, and {volume_cs_path.name}"
        )
        if job_type == LOCAL_REFINE_JOB_TYPE:
            log(
                "Using local-refine object alignment fields for reprojection: "
                f"{pose_field} and {shift_field}"
            )
        volume = load_refinement_volume(project_dir, volume_cs_path)
        particle_groups = load_refinement_particles(
            project_dir,
            particles_path,
            passthrough_path,
            pose_field=pose_field,
            shift_field=shift_field,
        )
        source_subset = None

    return OverlaySource(
        label=job_dir.name,
        job_dir=job_dir,
        subset=source_subset,
        source_kind=source_kind,
        overlay_color_name=overlay_color_name,
        overlay_color_rgb=parse_rgb_color(overlay_color_name),
        templates=templates,
        volume=volume,
        particle_groups=particle_groups,
    )


def combine_particle_groups(
    sources: list[OverlaySource],
) -> dict[Path, list[list[ParticleRecord]]]:
    combined: dict[Path, list[list[ParticleRecord]]] = {}
    source_count = len(sources)
    for source_index, source in enumerate(sources):
        for micrograph_path, particles in source.particle_groups.items():
            slots = combined.get(micrograph_path)
            if slots is None:
                slots = [[] for _ in range(source_count)]
                combined[micrograph_path] = slots
            slots[source_index] = particles
    if not combined:
        fail("No particles were loaded from any requested overlay source.")
    return combined


def first_particle(
    particle_lists: list[list[ParticleRecord]],
) -> ParticleRecord:
    for particles in particle_lists:
        if particles:
            return particles[0]
    fail("Encountered an internal error: micrograph entry has no particles.")


def particle_counts_by_source(
    particle_lists: list[list[ParticleRecord]],
) -> list[int]:
    return [len(particles) for particles in particle_lists]


def particle_totals_by_source(
    items: list[tuple[Path, list[list[ParticleRecord]]]],
    source_count: int,
) -> list[int]:
    totals = [0] * source_count
    for _, particle_lists in items:
        for source_index, particles in enumerate(particle_lists):
            totals[source_index] += len(particles)
    return totals


def harmonic_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values or any(value <= 0.0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def warn_on_particle_imbalance(
    sources: list[OverlaySource],
    totals: list[int],
    ratio_threshold: float,
    ranking_mode: str,
) -> None:
    positive_totals = [value for value in totals if value > 0]
    if len(positive_totals) < 2:
        return
    ratio = max(positive_totals) / min(positive_totals)
    if ratio < ratio_threshold:
        return
    detail = ", ".join(
        f"{source.label}={total}"
        for source, total in zip(sources, totals, strict=True)
    )
    recommendation = (
        "Balanced ranking is active and will prioritize micrographs that retain the rarer job."
        if ranking_mode == "balanced"
        else "Consider --top-micrographs-mode balanced to prioritize the rarer job."
    )
    log(
        "WARNING: particle counts are imbalanced across overlay jobs after filtering: "
        f"{detail} ({ratio:.2f}x ratio). {recommendation}"
    )


def rank_micrograph_items(
    items: list[tuple[Path, list[list[ParticleRecord]]]],
    sources: list[OverlaySource],
    ranking_mode: str,
    top_micrographs: int,
) -> list[tuple[Path, list[list[ParticleRecord]]]]:
    source_count = len(sources)
    if source_count == 1:
        ranked = sorted(
            items,
            key=lambda item: (-len(item[1][0]), item[0].name),
        )
        return ranked[:top_micrographs]

    totals = particle_totals_by_source(items, source_count)
    require_all_jobs = ranking_mode in {"balanced", "min"}
    if require_all_jobs:
        overlap_items = [
            item
            for item in items
            if all(len(particles) > 0 for particles in item[1])
        ]
        if not overlap_items:
            fail(
                "No micrographs contain particles from all requested overlay sources "
                f"after filtering, so --top-micrographs-mode {ranking_mode!r} cannot be used."
            )
        if len(overlap_items) < top_micrographs:
            log(
                "WARNING: only "
                f"{len(overlap_items)} micrographs contain particles from all "
                f"{source_count} jobs; rendering those instead of the requested top "
                f"{top_micrographs}."
            )
        items = overlap_items

    def score(item: tuple[Path, list[list[ParticleRecord]]]) -> tuple[float, ...]:
        counts = particle_counts_by_source(item[1])
        total_count = sum(counts)
        min_count = min(counts)
        if ranking_mode == "sum":
            return (
                float(total_count),
                float(min_count),
                harmonic_mean(float(count) for count in counts if count > 0),
            )
        if ranking_mode == "min":
            return (
                float(min_count),
                harmonic_mean(float(count) for count in counts),
                float(total_count),
            )
        normalized = [
            (count / total) if total > 0 else 0.0
            for count, total in zip(counts, totals, strict=True)
        ]
        return (
            min(normalized),
            harmonic_mean(normalized),
            sum(normalized),
            float(min_count),
            float(total_count),
        )

    ranked = sorted(
        items,
        key=lambda item: tuple(-value for value in score(item)) + (item[0].name,),
    )
    return ranked[:top_micrographs]


def filter_micrographs(
    particle_groups: dict[Path, list[list[ParticleRecord]]],
    explicit_micrographs: list[str] | None,
    max_micrographs: int | None,
    top_micrographs: int | None,
    denoised_targets: dict[int, TargetMicrographRecord] | None,
    sources: list[OverlaySource],
    top_micrographs_mode: str,
    imbalance_warning_ratio: float,
) -> list[tuple[Path, list[list[ParticleRecord]]]]:
    items = sorted(particle_groups.items(), key=lambda item: item[0].name)

    if explicit_micrographs:
        wanted = {item for item in explicit_micrographs}
        items = [
            item
            for item in items
            if item[0].name in wanted or str(item[0]) in wanted or item[0].stem in wanted
        ]
        if not items:
            fail("None of the requested micrographs matched the particle dataset.")

    if denoised_targets is not None:
        kept_items: list[tuple[Path, list[list[ParticleRecord]]]] = []
        skipped_items: list[tuple[Path, list[list[ParticleRecord]]]] = []
        for item in items:
            particle = first_particle(item[1])
            if particle.micrograph_uid in denoised_targets:
                kept_items.append(item)
            else:
                skipped_items.append(item)
        if skipped_items:
            skipped_particles = sum(
                sum(len(particles) for particles in particle_lists)
                for _, particle_lists in skipped_items
            )
            preview = ", ".join(path.name for path, _ in skipped_items[:3])
            suffix = "" if len(skipped_items) <= 3 else ", ..."
            log(
                "Skipping "
                f"{len(skipped_items)} micrographs ({skipped_particles} particles) "
                "because no denoised micrograph is available: "
                f"{preview}{suffix}"
            )
        items = kept_items
        if not items:
            fail(
                "No matching denoised micrographs remain after filtering. "
                "Nothing can be rendered for the requested selection."
            )

    if len(sources) > 1 and top_micrographs is not None:
        totals = particle_totals_by_source(items, len(sources))
        warn_on_particle_imbalance(
            sources=sources,
            totals=totals,
            ratio_threshold=imbalance_warning_ratio,
            ranking_mode=top_micrographs_mode,
        )

    if top_micrographs is not None:
        items = rank_micrograph_items(
            items=items,
            sources=sources,
            ranking_mode=top_micrographs_mode,
            top_micrographs=top_micrographs,
        )

    if max_micrographs is not None:
        items = items[:max_micrographs]
    return items


def get_target_record(
    particle_lists: list[list[ParticleRecord]],
    denoised_targets: dict[int, TargetMicrographRecord] | None,
) -> TargetMicrographRecord:
    particle = first_particle(particle_lists)
    if denoised_targets is None:
        return TargetMicrographRecord(
            path=particle.raw_micrograph_path,
            pixel_size_angstrom=particle.raw_micrograph_pixel_size_angstrom,
        )
    return denoised_targets[particle.micrograph_uid]


def circular_mask(shape: tuple[int, int], radius_fraction: float) -> np.ndarray:
    height, width = shape
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    radius = min(height, width) * radius_fraction
    yy, xx = np.ogrid[:height, :width]
    return ((yy - center_y) ** 2 + (xx - center_x) ** 2) <= radius**2


def make_select2d_stamp(
    class_image: np.ndarray,
    pose_rad: float,
    shift_x_px: float,
    shift_y_px: float,
    pose_sign: int,
    shift_sign: int,
    template_pixel_size_angstrom: float,
    shift_pixel_size_angstrom: float,
    target_pixel_size_angstrom: float,
) -> np.ndarray:
    fill_value = float(class_image.mean())
    stamp = ndimage.rotate(
        class_image,
        np.degrees(pose_sign * pose_rad),
        reshape=False,
        order=1,
        mode="constant",
        cval=fill_value,
    )
    shift_scale = shift_pixel_size_angstrom / template_pixel_size_angstrom
    stamp = ndimage.shift(
        stamp,
        shift=(
            shift_sign * shift_y_px * shift_scale,
            shift_sign * shift_x_px * shift_scale,
        ),
        order=1,
        mode="constant",
        cval=fill_value,
    )

    scale_factor = template_pixel_size_angstrom / target_pixel_size_angstrom
    if not math.isclose(scale_factor, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        stamp = ndimage.zoom(stamp, zoom=scale_factor, order=1)
    return stamp.astype(np.float32, copy=False)


def resample_volume_to_pixel_size(
    volume: np.ndarray,
    source_pixel_size_angstrom: float,
    target_pixel_size_angstrom: float,
) -> np.ndarray:
    scale_factor = source_pixel_size_angstrom / target_pixel_size_angstrom
    if math.isclose(scale_factor, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        return volume.astype(np.float32, copy=False)
    return ndimage.zoom(volume, zoom=scale_factor, order=1).astype(np.float32, copy=False)


def project_volume(
    volume: np.ndarray,
    pose3d: tuple[float, float, float],
    pose_sign: int,
) -> np.ndarray:
    fill_value = float(volume.mean())
    rotvec_xyz = pose_sign * np.asarray(pose3d, dtype=np.float32)
    rotation_xyz = Rotation.from_rotvec(rotvec_xyz)
    matrix_xyz = rotation_xyz.inv().as_matrix()
    matrix_zyx = matrix_xyz[::-1, ::-1]
    center = (np.asarray(volume.shape, dtype=np.float32) - 1.0) / 2.0
    offset = center - matrix_zyx @ center
    rotated = ndimage.affine_transform(
        volume,
        matrix_zyx,
        offset=offset,
        order=1,
        mode="constant",
        cval=fill_value,
    )
    return rotated.sum(axis=0).astype(np.float32, copy=False)


def quantize_pose3d(
    pose3d: tuple[float, float, float],
    angle_step_deg: float,
) -> tuple[float, float, float]:
    pose_vec = np.asarray(pose3d, dtype=np.float32)
    if angle_step_deg <= 0.0:
        return tuple(float(value) for value in pose_vec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        euler_deg = Rotation.from_rotvec(pose_vec).as_euler("ZYZ", degrees=True)
    quantized_euler_deg = np.round(euler_deg / angle_step_deg) * angle_step_deg
    quantized_pose = Rotation.from_euler("ZYZ", quantized_euler_deg, degrees=True).as_rotvec()
    return tuple(float(value) for value in quantized_pose)


def make_refine3d_stamp_from_projection(
    projection: np.ndarray,
    shift_x_px: float,
    shift_y_px: float,
    shift_sign: int,
    volume_pixel_size_angstrom: float,
    shift_pixel_size_angstrom: float,
    target_pixel_size_angstrom: float,
) -> np.ndarray:
    fill_value = float(projection.mean())
    shift_scale = shift_pixel_size_angstrom / volume_pixel_size_angstrom
    stamp = ndimage.shift(
        projection,
        shift=(
            shift_sign * shift_y_px * shift_scale,
            shift_sign * shift_x_px * shift_scale,
        ),
        order=1,
        mode="constant",
        cval=fill_value,
    )
    scale_factor = volume_pixel_size_angstrom / target_pixel_size_angstrom
    if not math.isclose(scale_factor, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        stamp = ndimage.zoom(stamp, zoom=scale_factor, order=1)
    return stamp.astype(np.float32, copy=False)


def paste_stamp(
    overlay_sum: np.ndarray,
    overlay_weight: np.ndarray,
    stamp: np.ndarray,
    center_x: float,
    center_y: float,
    mask_radius_fraction: float,
) -> None:
    stamp = stamp.astype(np.float32, copy=False)
    mask = circular_mask(stamp.shape, mask_radius_fraction)

    rounded_center_x = int(round(center_x))
    rounded_center_y = int(round(center_y))
    frac_shift_x = center_x - rounded_center_x
    frac_shift_y = center_y - rounded_center_y
    if abs(frac_shift_x) > 1e-6 or abs(frac_shift_y) > 1e-6:
        fill_value = float(stamp.mean())
        stamp = ndimage.shift(
            stamp,
            shift=(frac_shift_y, frac_shift_x),
            order=1,
            mode="constant",
            cval=fill_value,
        )

    height, width = stamp.shape
    x0 = rounded_center_x - width // 2
    y0 = rounded_center_y - height // 2
    x1 = x0 + width
    y1 = y0 + height

    clip_x0 = max(0, x0)
    clip_y0 = max(0, y0)
    clip_x1 = min(overlay_sum.shape[1], x1)
    clip_y1 = min(overlay_sum.shape[0], y1)
    if clip_x0 >= clip_x1 or clip_y0 >= clip_y1:
        return

    src_x0 = clip_x0 - x0
    src_y0 = clip_y0 - y0
    src_x1 = src_x0 + (clip_x1 - clip_x0)
    src_y1 = src_y0 + (clip_y1 - clip_y0)

    stamp_view = stamp[src_y0:src_y1, src_x0:src_x1]
    mask_view = mask[src_y0:src_y1, src_x0:src_x1]
    if not np.any(mask_view):
        return

    overlay_sum[clip_y0:clip_y1, clip_x0:clip_x1][mask_view] += stamp_view[mask_view]
    overlay_weight[clip_y0:clip_y1, clip_x0:clip_x1][mask_view] += 1.0


def robust_scale(array: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    lo = float(np.percentile(array, low))
    hi = float(np.percentile(array, high))
    if math.isclose(lo, hi):
        return np.zeros_like(array, dtype=np.float32)
    scaled = (array - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32, copy=False)


def normalize_overlay(overlay: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if not np.any(valid_mask):
        return np.zeros_like(overlay, dtype=np.float32)
    values = overlay[valid_mask]
    scale = float(np.percentile(np.abs(values), 99.5))
    if math.isclose(scale, 0.0):
        return np.zeros_like(overlay, dtype=np.float32)
    normalized = overlay / scale
    return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)


def downsample_image(array: np.ndarray, factor: int, order: int = 1) -> np.ndarray:
    if factor <= 1:
        return array
    return ndimage.zoom(array, zoom=1.0 / factor, order=order)


def render_overlay(
    micrograph: np.ndarray,
    overlay_avgs: list[np.ndarray],
    overlay_weights: list[np.ndarray],
    overlay_color_rgbs: list[np.ndarray],
    synthetic_background_rgb: np.ndarray,
    micrograph_opacity: float,
    class_opacity: float,
    downsample: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    micro_scaled = robust_scale(micrograph)
    micro_scaled = downsample_image(micro_scaled, downsample)

    base = np.repeat(micro_scaled[:, :, None], 3, axis=2)
    base_weight = np.full(base.shape[:2], micrograph_opacity, dtype=np.float32)
    total_weight = np.array(base_weight, copy=True)
    overlay_strength_total = np.zeros(base.shape[:2], dtype=np.float32)
    color_sum = np.zeros_like(base, dtype=np.float32)
    composite_numerator = base * base_weight[..., None]
    count_sum = np.zeros(base.shape[:2], dtype=np.float32)

    for overlay_avg, overlay_weight, overlay_color_rgb in zip(
        overlay_avgs,
        overlay_weights,
        overlay_color_rgbs,
        strict=True,
    ):
        valid_mask = overlay_weight > 0
        overlay_norm = normalize_overlay(overlay_avg, valid_mask)
        overlay_norm = downsample_image(overlay_norm, downsample)
        valid_mask = downsample_image(valid_mask.astype(np.float32), downsample, order=0) > 0.5
        count_sum += downsample_image(overlay_weight, downsample)
        overlay_strength = np.clip(np.abs(overlay_norm), 0.0, 1.0)
        overlay_strength *= valid_mask.astype(np.float32)
        overlay_weight_map = class_opacity * overlay_strength
        overlay_rgb = np.broadcast_to(overlay_color_rgb.reshape(1, 1, 3), base.shape)
        composite_numerator += overlay_rgb * overlay_weight_map[..., None]
        color_sum += overlay_rgb * overlay_weight_map[..., None]
        total_weight += overlay_weight_map
        overlay_strength_total += overlay_weight_map

    composite = composite_numerator / np.maximum(total_weight[..., None], 1e-6)
    average_color = color_sum / np.maximum(overlay_strength_total[..., None], 1e-6)
    synthetic_alpha = np.clip(overlay_strength_total[..., None], 0.0, 1.0)
    synthetic_background = np.broadcast_to(
        synthetic_background_rgb.reshape(1, 1, 3),
        base.shape,
    )
    synthetic = (
        synthetic_background * (1.0 - synthetic_alpha)
        + average_color * synthetic_alpha
    )
    count_image = robust_scale(count_sum)
    count_rgb = np.repeat(count_image[:, :, None], 3, axis=2)

    return (
        np.clip(base * 255.0, 0.0, 255.0).astype(np.uint8),
        np.clip(composite * 255.0, 0.0, 255.0).astype(np.uint8),
        np.clip(synthetic * 255.0, 0.0, 255.0).astype(np.uint8),
        np.clip(count_rgb * 255.0, 0.0, 255.0).astype(np.uint8),
    )


def save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)


def save_blink_gif(
    path: Path,
    base_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    frame_duration_ms: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        Image.fromarray(base_rgb, mode="RGB"),
        Image.fromarray(overlay_rgb, mode="RGB"),
    ]
    frames[0].save(
        path,
        save_all=True,
        append_images=[frames[1], frames[0], frames[1]],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )


def format_source_summary(source: OverlaySource) -> str:
    if source.source_kind == SOURCE_KIND_SELECT2D:
        return f"{source.label}({source_kind_label(source.source_kind)}, {source.subset}, {source.overlay_color_name})"
    return f"{source.label}({source_kind_label(source.source_kind)}, {source.overlay_color_name})"


def default_output_name(
    sources: list[OverlaySource],
    primary_subset: str,
    denoise_job_dir: Path | None,
    job_dir_2: Path | None,
) -> str:
    if all(source.source_kind == SOURCE_KIND_SELECT2D for source in sources):
        output_name = f"{primary_subset}_2d_class_overlay"
    else:
        output_name = "particle_reprojection_overlay"
    if job_dir_2 is not None:
        output_name += f"_{job_dir_2.name}"
    if denoise_job_dir is not None:
        output_name += f"_{denoise_job_dir.name}"
    return output_name


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Take one or two CryoSPARC overlay sources and place their per-particle signal "
            "back onto the source micrographs. Supported sources are select_2D jobs "
            "(2D class averages) and 3D refinement jobs (per-particle map projections)."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    required = parser.add_argument_group("Required")
    required.add_argument(
        "--job-dir",
        required=True,
        help=(
            "Path to a CryoSPARC select_2D or 3D refinement job directory, "
            "for example /path/to/J119 or /path/to/J95"
        ),
    )
    required.add_argument(
        "--job-dir-2",
        help=(
            "Optional second CryoSPARC select_2D or 3D refinement job directory. "
            "When set, both sources are projected back into the same target micrographs at once."
        ),
    )

    main = parser.add_argument_group("Common")
    main.add_argument(
        "--subset",
        choices=("selected", "excluded"),
        default="selected",
        help=(
            "Subset for select_2D jobs only: use particles_selected/templates_selected "
            "or the excluded counterparts (default: selected)"
        ),
    )
    main.add_argument(
        "--subset-2",
        choices=("selected", "excluded"),
        help="Subset for --job-dir-2 (default: same value as --subset)",
    )
    main.add_argument(
        "--denoise-job-dir",
        help=(
            "Optional CryoSPARC denoise job directory. When set, the class averages are "
            "placed onto the denoised micrographs instead of the original motion-corrected ones."
        ),
    )
    main.add_argument(
        "--output-dir",
        help=(
            "Output directory (default: keep the historical <subset>_2d_class_overlay name "
            "for pure Select 2D runs, otherwise use particle_reprojection_overlay; optional "
            "_<job-dir-2> and _<denoise-job> suffixes are appended when those inputs are used)"
        ),
    )
    main.add_argument(
        "--overlay-color",
        default=DEFAULT_OVERLAY_COLOR,
        help=(
            "Overlay color for the primary overlay source. "
            f"Default: {DEFAULT_OVERLAY_COLOR}"
        ),
    )
    main.add_argument(
        "--overlay-color-2",
        default=DEFAULT_SECONDARY_OVERLAY_COLOR,
        help=(
            "Overlay color for --job-dir-2. Ignored unless a second job is supplied. "
            f"Default: {DEFAULT_SECONDARY_OVERLAY_COLOR}"
        ),
    )
    main.add_argument(
        "--synthetic-background-color",
        default=DEFAULT_SYNTHETIC_BACKGROUND_COLOR,
        help=(
            "Background color for the synthetic PNG. Use a CSS color name or hex code, "
            f"or 'auto' to choose white for dark overlays and black for bright ones "
            f"(default: {DEFAULT_SYNTHETIC_BACKGROUND_COLOR})"
        ),
    )
    main.add_argument(
        "--micrograph-opacity",
        type=float,
        default=DEFAULT_MICROGRAPH_OPACITY,
        help=(
            "Relative opacity weight of the background micrograph in the composite "
            f"(default: {DEFAULT_MICROGRAPH_OPACITY})"
        ),
    )
    main.add_argument(
        "--class-opacity",
        type=float,
        default=DEFAULT_CLASS_OPACITY,
        help=(
            "Relative opacity weight of the overlaid particle signal in the composite "
            f"(default: {DEFAULT_CLASS_OPACITY})"
        ),
    )
    main.add_argument(
        "--alpha",
        type=float,
        help="Legacy alias for --class-opacity",
    )
    main.add_argument(
        "--png-downsample",
        type=int,
        default=DEFAULT_PNG_DOWNSAMPLE,
        help=(
            "Downsample PNG outputs by this integer factor. "
            f"1 keeps full resolution (default: {DEFAULT_PNG_DOWNSAMPLE})"
        ),
    )
    main.add_argument(
        "--gif-downsample",
        type=int,
        default=DEFAULT_GIF_DOWNSAMPLE,
        help=(
            "Downsample GIF outputs by this integer factor. "
            f"1 keeps full resolution (default: {DEFAULT_GIF_DOWNSAMPLE})"
        ),
    )
    main.add_argument(
        "--downsample",
        type=int,
        help="Legacy alias that sets both --png-downsample and --gif-downsample",
    )
    main.add_argument(
        "--write-count-images",
        action="store_true",
        help="Also write per-micrograph hit-count PNGs that show overlap density",
    )
    main.add_argument(
        "--write-gifs",
        dest="write_gifs",
        action="store_true",
        help=(
            "Write blink GIFs that alternate between the plain micrograph and the overlay "
            "composite. This is enabled by default."
        ),
    )
    main.add_argument(
        "--no-write-gifs",
        dest="write_gifs",
        action="store_false",
        help="Disable blink GIF output",
    )
    main.add_argument(
        "--gif-frame-ms",
        type=int,
        default=DEFAULT_GIF_FRAME_MS,
        help=(
            "Frame duration for blink GIFs in milliseconds "
            f"(default: {DEFAULT_GIF_FRAME_MS})"
        ),
    )
    parser.set_defaults(write_gifs=True)

    advanced = parser.add_argument_group("Advanced")
    advanced.add_argument(
        "--micrographs",
        help=(
            "Comma-separated list of micrograph basenames, stems, or full paths to render. "
            "Useful for fast tests."
        ),
    )
    advanced.add_argument(
        "--max-micrographs",
        type=int,
        help="Only render the first N matching micrographs after sorting by basename",
    )
    advanced.add_argument(
        "--top-micrographs",
        type=int,
        help=(
            "Only render the N most populated micrographs. With two jobs, the ranking can "
            "prioritize balanced overlap instead of plain total count."
        ),
    )
    advanced.add_argument(
        "--top-micrographs-mode",
        choices=DEFAULT_TOP_MICROGRAPHS_MODES,
        help=(
            "Ranking mode for --top-micrographs when multiple jobs are supplied: "
            "'sum' ranks by total particles across jobs, 'min' ranks by the lower raw "
            "per-job count, and 'balanced' ranks by the lower normalized per-job "
            "contribution and is recommended for imbalanced jobs."
        ),
    )
    advanced.add_argument(
        "--imbalance-warning-ratio",
        type=float,
        default=DEFAULT_IMBALANCE_WARNING_RATIO,
        help=(
            "Warn when the total contributing particle counts between jobs differ by at "
            f"least this factor (default: {DEFAULT_IMBALANCE_WARNING_RATIO})"
        ),
    )
    advanced.add_argument(
        "--pose-sign",
        type=int,
        choices=(-1, 1),
        default=DEFAULT_POSE_SIGN,
        help=(
            "Sign applied to alignment poses before rotation. "
            "For 2D sources this affects alignments2D/pose; for 3D sources it affects "
            f"the 3-vector alignments3D/pose rotation. Default: {DEFAULT_POSE_SIGN}."
        ),
    )
    advanced.add_argument(
        "--shift-sign",
        type=int,
        choices=(-1, 1),
        default=DEFAULT_SHIFT_SIGN,
        help=(
            "Sign applied to alignment shifts before stamping. "
            "For 2D sources this affects alignments2D/shift; for 3D sources it affects "
            f"alignments3D/shift. Default: {DEFAULT_SHIFT_SIGN}."
        ),
    )
    advanced.add_argument(
        "--projection-angle-step-deg",
        type=float,
        default=DEFAULT_PROJECTION_ANGLE_STEP_DEG,
        help=(
            "For 3D refinement sources, quantize particle orientations into Euler-angle bins "
            "of this size and reuse the cached backprojection for each bin. Use 0 to disable "
            f"quantization and project every particle exactly. Default: {DEFAULT_PROJECTION_ANGLE_STEP_DEG}"
        ),
    )
    advanced.add_argument(
        "--mask-radius-fraction",
        type=float,
        default=DEFAULT_MASK_RADIUS_FRACTION,
        help=(
            "Radius of the circular stamp mask as a fraction of the transformed box size "
            f"(default: {DEFAULT_MASK_RADIUS_FRACTION})"
        ),
    )

    parser.epilog = """Examples:
  Render all selected micrographs from a Select 2D job:
    cryosparc-2d-class-overlay --job-dir /path/to/J119

  Render onto denoised micrographs from J10:
    cryosparc-2d-class-overlay \
      --job-dir /path/to/J46 \
      --denoise-job-dir /path/to/J10

  Overlay two Select 2D jobs into the same micrographs:
    cryosparc-2d-class-overlay \
      --job-dir /path/to/J46 \
      --job-dir-2 /path/to/J52 \
      --overlay-color black \
      --overlay-color-2 red

  Render per-particle 3D map backprojections from a refinement job:
    cryosparc-2d-class-overlay \
      --job-dir /path/to/J95 \
      --projection-angle-step-deg 5 \
      --top-micrographs 1

  Combine a Select 2D source with a 3D refinement source:
    cryosparc-2d-class-overlay \
      --job-dir /path/to/J46 \
      --job-dir-2 /path/to/J95 \
      --overlay-color black \
      --overlay-color-2 red

  Force a white background for the synthetic PNGs:
    cryosparc-2d-class-overlay \
      --job-dir /path/to/J46 \
      --synthetic-background-color white

  Prioritize micrographs that keep both jobs well represented:
    cryosparc-2d-class-overlay \
      --job-dir /path/to/J46 \
      --job-dir-2 /path/to/J52 \
      --top-micrographs 10 \
      --top-micrographs-mode balanced

  Render black overlays with a stronger class contribution:
    cryosparc-2d-class-overlay \
      --job-dir /path/to/J119 \
      --overlay-color black \
      --micrograph-opacity 0.8 \
      --class-opacity 1.2

  Test only three micrographs and downsample the PNGs:
    cryosparc-2d-class-overlay \\
      --job-dir /path/to/J119 \\
      --micrographs mic1_patch_aligned_doseweighted,mic2_patch_aligned_doseweighted,mic3 \\
      --png-downsample 2

  Write lower-resolution blink GIFs while keeping full-resolution PNGs:
    cryosparc-2d-class-overlay \
      --job-dir /path/to/J119 \
      --png-downsample 1 \
      --gif-downsample 4

  Disable GIF output entirely:
    cryosparc-2d-class-overlay \
      --job-dir /path/to/J119 \
      --no-write-gifs

  Render only the ten most populated micrographs:
    cryosparc-2d-class-overlay \\
      --job-dir /path/to/J119 \\
      --top-micrographs 10
"""
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    job_dir = Path(args.job_dir).expanduser().resolve()
    if not job_dir.exists():
        fail(f"Job directory not found: {job_dir}")
    job_dir_2 = None
    if args.job_dir_2:
        job_dir_2 = Path(args.job_dir_2).expanduser().resolve()
        if not job_dir_2.exists():
            fail(f"Second job directory not found: {job_dir_2}")
    denoise_job_dir = None
    if args.denoise_job_dir:
        denoise_job_dir = Path(args.denoise_job_dir).expanduser().resolve()
        if not denoise_job_dir.exists():
            fail(f"Denoise job directory not found: {denoise_job_dir}")
        validate_denoise_job(denoise_job_dir)

    subset_2 = args.subset_2 or args.subset
    if args.downsample is not None:
        if args.downsample < 1:
            fail("--downsample must be at least 1")
        args.png_downsample = args.downsample
        args.gif_downsample = args.downsample
    if args.alpha is not None:
        args.class_opacity = args.alpha
    if args.png_downsample < 1:
        fail("--png-downsample must be at least 1")
    if args.gif_downsample < 1:
        fail("--gif-downsample must be at least 1")
    if args.gif_frame_ms < 1:
        fail("--gif-frame-ms must be at least 1")
    if args.micrograph_opacity <= 0.0:
        fail("--micrograph-opacity must be greater than 0")
    if args.class_opacity < 0.0:
        fail("--class-opacity must be greater than or equal to 0")
    if args.max_micrographs is not None and args.max_micrographs < 1:
        fail("--max-micrographs must be at least 1")
    if args.top_micrographs is not None and args.top_micrographs < 1:
        fail("--top-micrographs must be at least 1")
    if args.max_micrographs is not None and args.top_micrographs is not None:
        fail("Use either --max-micrographs or --top-micrographs, not both.")
    if args.imbalance_warning_ratio <= 1.0:
        fail("--imbalance-warning-ratio must be greater than 1")
    if args.projection_angle_step_deg < 0.0:
        fail("--projection-angle-step-deg must be greater than or equal to 0")
    if not (0.0 < args.mask_radius_fraction <= 0.5):
        fail("--mask-radius-fraction must be in the range (0, 0.5]")

    sources = [
        load_overlay_source(job_dir, args.subset, args.overlay_color),
    ]
    if job_dir_2 is not None:
        sources.append(load_overlay_source(job_dir_2, subset_2, args.overlay_color_2))
    output_name = default_output_name(sources, args.subset, denoise_job_dir, job_dir_2)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else job_dir / output_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    top_micrographs_mode = args.top_micrographs_mode or (
        DEFAULT_TOP_MICROGRAPHS_MODE_MULTI
        if len(sources) > 1
        else DEFAULT_TOP_MICROGRAPHS_MODE_SINGLE
    )
    synthetic_background_rgb = resolve_synthetic_background_color(
        args.synthetic_background_color,
        [source.overlay_color_rgb for source in sources],
    )
    particle_groups = combine_particle_groups(sources)
    denoised_targets = load_denoised_targets(denoise_job_dir) if denoise_job_dir is not None else None

    selected_micrographs = filter_micrographs(
        particle_groups,
        parse_csv(args.micrographs) if args.micrographs else None,
        args.max_micrographs,
        args.top_micrographs,
        denoised_targets,
        sources,
        top_micrographs_mode,
        args.imbalance_warning_ratio,
    )
    source_summary = ", ".join(
        format_source_summary(source) for source in sources
    )
    log(
        f"Rendering {len(selected_micrographs)} micrographs from {source_summary} "
        f"into {output_dir}"
    )
    if args.top_micrographs is not None and len(sources) > 1:
        log(f"Top-micrograph ranking mode: {top_micrographs_mode}")
    if denoise_job_dir is not None:
        log(f"Rendering onto denoised targets from {denoise_job_dir.name}")
    if any(source.source_kind == SOURCE_KIND_REFINE3D for source in sources):
        if args.projection_angle_step_deg > 0.0:
            log(
                "Quantizing 3D refinement orientations for projection caching at "
                f"{args.projection_angle_step_deg:.2f} degree bins"
            )
        else:
            log("3D refinement projection caching is disabled; using exact per-particle orientations")

    mrc_reader = MrcReader()
    template_cache: dict[tuple[Path, int], np.ndarray] = {}
    volume_cache: dict[Path, np.ndarray] = {}
    resampled_volume_cache: dict[tuple[Path, float], np.ndarray] = {}
    projection_cache: dict[tuple[Path, float, tuple[float, float, float]], np.ndarray] = {}
    summary_rows: list[dict[str, str | int | float]] = []
    per_source_count_fields = [
        f"particle_count_{normalize_field_label(source.label)}"
        for source in sources
    ]

    for index, (micrograph_path, particle_lists) in enumerate(selected_micrographs, start=1):
        target = get_target_record(particle_lists, denoised_targets)
        if not target.path.exists():
            fail(f"Target micrograph file not found: {target.path}")
        micrograph = mrc_reader.read_slice(target.path, 0)
        target_height, target_width = micrograph.shape
        overlay_sums: list[np.ndarray] = []
        overlay_weights: list[np.ndarray] = []

        for source_index, particles in enumerate(particle_lists):
            overlay_sum = np.zeros_like(micrograph, dtype=np.float32)
            overlay_weight = np.zeros_like(micrograph, dtype=np.float32)
            if particles:
                source = sources[source_index]
                for particle in particles:
                    if source.source_kind == SOURCE_KIND_SELECT2D:
                        if source.templates is None or particle.class_idx is None or particle.pose_rad is None:
                            fail("Encountered an internal error: Select 2D particle is missing template data.")
                        template = source.templates[particle.class_idx]
                        template_key = (template.selected_stack_path, template.selected_stack_idx)
                        class_image = template_cache.get(template_key)
                        if class_image is None:
                            class_image = mrc_reader.read_slice(
                                template.selected_stack_path, template.selected_stack_idx
                            )
                            template_cache[template_key] = class_image

                        stamp = make_select2d_stamp(
                            class_image=class_image,
                            pose_rad=particle.pose_rad,
                            shift_x_px=particle.shift_x_px,
                            shift_y_px=particle.shift_y_px,
                            pose_sign=args.pose_sign,
                            shift_sign=args.shift_sign,
                            template_pixel_size_angstrom=template.pixel_size_angstrom,
                            shift_pixel_size_angstrom=particle.align_pixel_size_angstrom,
                            target_pixel_size_angstrom=target.pixel_size_angstrom,
                        )
                    else:
                        if source.volume is None or particle.pose3d is None:
                            fail("Encountered an internal error: 3D refinement particle is missing volume data.")
                        raw_volume = volume_cache.get(source.volume.map_path)
                        if raw_volume is None:
                            raw_volume = mrc_reader.read_volume(source.volume.map_path)
                            volume_cache[source.volume.map_path] = raw_volume
                        resampled_key = (source.volume.map_path, round(target.pixel_size_angstrom, 6))
                        projection_volume = resampled_volume_cache.get(resampled_key)
                        if projection_volume is None:
                            projection_volume = resample_volume_to_pixel_size(
                                raw_volume,
                                source_pixel_size_angstrom=source.volume.pixel_size_angstrom,
                                target_pixel_size_angstrom=target.pixel_size_angstrom,
                            )
                            resampled_volume_cache[resampled_key] = projection_volume
                        quantized_pose3d = quantize_pose3d(
                            particle.pose3d,
                            args.projection_angle_step_deg,
                        )
                        projection_key = (
                            source.volume.map_path,
                            round(target.pixel_size_angstrom, 6),
                            tuple(round(value, 6) for value in quantized_pose3d),
                        )
                        projection = projection_cache.get(projection_key)
                        if projection is None:
                            projection = project_volume(
                                volume=projection_volume,
                                pose3d=quantized_pose3d,
                                pose_sign=args.pose_sign,
                            )
                            projection_cache[projection_key] = projection
                        stamp = make_refine3d_stamp_from_projection(
                            projection=projection,
                            shift_x_px=particle.shift_x_px,
                            shift_y_px=particle.shift_y_px,
                            shift_sign=args.shift_sign,
                            volume_pixel_size_angstrom=target.pixel_size_angstrom,
                            shift_pixel_size_angstrom=particle.align_pixel_size_angstrom,
                            target_pixel_size_angstrom=target.pixel_size_angstrom,
                        )
                    paste_stamp(
                        overlay_sum=overlay_sum,
                        overlay_weight=overlay_weight,
                        stamp=stamp,
                        center_x=particle.center_x_frac * target_width,
                        center_y=particle.center_y_frac * target_height,
                        mask_radius_fraction=args.mask_radius_fraction,
                    )
            overlay_sums.append(overlay_sum)
            overlay_weights.append(overlay_weight)

        overlay_avgs: list[np.ndarray] = []
        for overlay_sum, overlay_weight in zip(overlay_sums, overlay_weights, strict=True):
            overlay_avg = np.zeros_like(overlay_sum)
            np.divide(overlay_sum, overlay_weight, out=overlay_avg, where=overlay_weight > 0)
            overlay_avgs.append(overlay_avg)

        base_png, overlay_png, synthetic_png, count_png = render_overlay(
            micrograph=micrograph,
            overlay_avgs=overlay_avgs,
            overlay_weights=overlay_weights,
            overlay_color_rgbs=[source.overlay_color_rgb for source in sources],
            synthetic_background_rgb=synthetic_background_rgb,
            micrograph_opacity=args.micrograph_opacity,
            class_opacity=args.class_opacity,
            downsample=args.png_downsample,
        )
        gif_base = gif_overlay = None
        if args.write_gifs:
            gif_base, gif_overlay, _, _ = render_overlay(
                micrograph=micrograph,
                overlay_avgs=overlay_avgs,
                overlay_weights=overlay_weights,
                overlay_color_rgbs=[source.overlay_color_rgb for source in sources],
                synthetic_background_rgb=synthetic_background_rgb,
                micrograph_opacity=args.micrograph_opacity,
                class_opacity=args.class_opacity,
                downsample=args.gif_downsample,
            )

        stem = target.path.stem
        overlay_path = output_dir / f"{stem}.overlay.png"
        synthetic_path = output_dir / f"{stem}.synthetic.png"
        save_png(overlay_path, overlay_png)
        save_png(synthetic_path, synthetic_png)

        count_path = ""
        if args.write_count_images:
            count_path = str(output_dir / f"{stem}.count.png")
            save_png(Path(count_path), count_png)

        gif_path = ""
        if args.write_gifs and gif_base is not None and gif_overlay is not None:
            gif_path = str(output_dir / f"{stem}.blink.gif")
            save_blink_gif(Path(gif_path), gif_base, gif_overlay, args.gif_frame_ms)

        per_source_counts = particle_counts_by_source(particle_lists)
        total_particles = sum(per_source_counts)
        overlay_pixel_count = int(np.count_nonzero(sum(overlay_weights)))
        summary_rows.append(
            {
                "raw_micrograph_path": str(micrograph_path),
                "rendered_micrograph_path": str(target.path),
                "particle_count": total_particles,
                "overlay_png": str(overlay_path),
                "synthetic_png": str(synthetic_path),
                "count_png": count_path,
                "blink_gif": gif_path,
                "overlay_pixel_count": overlay_pixel_count,
                "rendered_pixel_size_angstrom": target.pixel_size_angstrom,
                **{
                    field_name: count
                    for field_name, count in zip(
                        per_source_count_fields,
                        per_source_counts,
                        strict=True,
                    )
                },
            }
        )
        count_detail = ", ".join(
            f"{source.label}={count}"
            for source, count in zip(sources, per_source_counts, strict=True)
        )
        log(
            f"[{index}/{len(selected_micrographs)}] Wrote overlays for {target.path.name} "
            f"using {total_particles} particles ({count_detail})"
        )

    summary_path = output_dir / "overlay_summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "raw_micrograph_path",
                "rendered_micrograph_path",
                "particle_count",
                *per_source_count_fields,
                "overlay_png",
                "synthetic_png",
                "count_png",
                "blink_gif",
                "overlay_pixel_count",
                "rendered_pixel_size_angstrom",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    log(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
