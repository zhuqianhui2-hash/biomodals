"""ProtenixScore Modal wrapper for structure scoring.

## Overview

- Modal wrapper around ProtenixScore for scoring existing protein structures.
- Input is an existing `PDB`/`CIF`; output is confidence/interface scoring.
- Scoring only: no structure generation, no diffusion, no AF3-style prediction run.

## Upstream

- Protenix: <https://github.com/bytedance/Protenix>
- ProtenixScore: <https://github.com/cytokineking/ProtenixScore>
- Protenix is the upstream open-source structure prediction/model family; this script only wraps its confidence head for scoring existing coordinates.

## When To Use

- Post-design filtering for binder, nanobody, or PPIFlow outputs.
- Batch quality evaluation for existing structure sets.
- Interface / complex quality scoring for multi-chain inputs.

## When NOT To Use

- Not a structure prediction tool.
- Not an AF3 / diffusion generation pipeline.
- Does not generate de novo structures.
- Does not create new `pLDDT` / `iPTM` semantics beyond what the internal scorer already writes.

## MSA

- Current wrapper supports MSA only indirectly through `--extra-args`; there are no dedicated top-level Modal flags such as `--use-msa` or `--msa-path`.
- In practice, design workflows usually run without meaningful MSA. If needed, pass ProtenixScore CLI flags through `--extra-args`.
- Typical no-MSA cases: de novo binders, RFdiffusion / PPIFlow backbones, ProteinMPNN-designed sequences.
- Cases that may benefit from real MSA: natural proteins, evolution-supported proteins, native complexes.
- If enabling MSA, it must match the scored sequence exactly; designed proteins often do not have a valid MSA.
- Actual downstream flags are:
  - `--use_msas both|target|binder|false`
  - `--msa_map_csv <csv>`
  - `--target_msa_shared_dir <dir>`
  - `--binder_msa_shared_dir <dir>`
- For no-MSA scoring, use `--extra-args '--use_msas false'`.

## Input / Output

- Input: one `PDB`/`CIF` file or a directory of structures.
- Output: score summaries, per-sample logs/JSON outputs, and `failed_records.txt` when failures occur.
- Default aggregate output includes `summary.csv`; per-sample outputs are written under the output directory.

## CLI

Single file:

```bash
modal run protenixscore_app.py \
  --input-pdb test.pdb \
  --out-dir outputs
```

Batch directory:

```bash
modal run protenixscore_app.py \
  --input-pdb test_pdbs \
  --out-dir outputs
```

MSA via forwarded CLI args:

```bash
modal run protenixscore_app.py \
  --input-pdb test_pdbs \
  --out-dir outputs \
  --extra-args '--use_msas target --target_msa_shared_dir msa_dir'
```

## Notes

- This script is a scoring wrapper, not the model codebase itself.
- Inputs should be valid all-atom structures when possible; missing atoms may affect scoring behavior.
- Chain IDs should be correct and stable.
- Interface metrics only make sense for complexes / multi-chain inputs.
- Failed samples are recorded in `failed_records.txt`.
- Checkpoints are resolved from the default volume path unless `--checkpoint-dir` is provided.
- GPU type is controlled by `GPU` (default `L40S`).
- There is no explicit resume flag in this wrapper; reruns depend on the chosen output directory / run name behavior.
"""

# Ignore ruff warnings about import location and subprocess usage in image build.
# ruff: noqa: PLC0415

import io
import importlib.util
import os
import shlex
import sys
import tarfile
import uuid
from pathlib import Path

from modal import App, Image, Volume

##########################################
# Modal configs
##########################################
GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", "7200"))
APP_NAME = os.environ.get("MODAL_APP", "ProtenixScore")

REPO_DIR = "/opt/ProtenixScore"
PROTENIX_DIR = f"{REPO_DIR}/Protenix_fork"
REPO_URL = os.environ.get(
    "PROTENIXSCORE_REPO_URL", "https://github.com/cytokineking/ProtenixScore.git"
)
REPO_COMMIT = os.environ.get(
    "PROTENIXSCORE_REPO_COMMIT", "64303b57aa1d4874e01fe8ca59ed9735283c24fa"
)
PROTENIX_FORK_URL = os.environ.get(
    "PROTENIX_FORK_URL", "https://github.com/cytokineking/Protenix.git"
)
PROTENIX_FORK_COMMIT = os.environ.get(
    "PROTENIX_FORK_COMMIT", "6f3a6175f372161d668b505356cff0a7d91fe520"
)

CHECKPOINTS_VOLUME_NAME = "protenixscore-checkpoints"
CHECKPOINTS_VOLUME = Volume.from_name(
    CHECKPOINTS_VOLUME_NAME, create_if_missing=True, version=2
)
CHECKPOINTS_DIR = "/vol/protenix/checkpoints"

DATA_VOLUME_NAME = "protenixscore-data"
DATA_VOLUME = Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True, version=2)
DATA_DIR = "/vol/protenix/data"

OUTPUTS_VOLUME_NAME = "protenixscore-outputs"
OUTPUTS_VOLUME = Volume.from_name(OUTPUTS_VOLUME_NAME, create_if_missing=True, version=2)
OUTPUTS_DIR = "/vol/protenix/outputs"

DEFAULT_MODEL_NAME = "protenix_base_default_v1.0.0"
OBSOLETE_RELEASE_DATE_URL = (
    "https://protenix.tos-cn-beijing.volces.com/common/obsolete_release_date.csv"
)

##########################################
# Image and app definitions
##########################################
runtime_image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "build-essential")
    .run_commands("python -m pip install --upgrade pip setuptools wheel ninja")
    .run_commands(
        f"git clone {shlex.quote(REPO_URL)} {shlex.quote(REPO_DIR)}",
        f"git -C {shlex.quote(REPO_DIR)} checkout --detach {shlex.quote(REPO_COMMIT)}",
        f"git clone {shlex.quote(PROTENIX_FORK_URL)} {shlex.quote(PROTENIX_DIR)}",
        f"git -C {shlex.quote(PROTENIX_DIR)} checkout --detach {shlex.quote(PROTENIX_FORK_COMMIT)}",
        f"python -m pip install -e {shlex.quote(PROTENIX_DIR)}",
    )
    .workdir(REPO_DIR)
    .env(
        {
            "PROTENIX_CHECKPOINT_DIR": CHECKPOINTS_DIR,
            "PROTENIX_DATA_ROOT_DIR": DATA_DIR,
            "LAYERNORM_TYPE": "torch",
        }
    )
)

app = App(APP_NAME, image=runtime_image)


##########################################
# Helper functions
##########################################
def _safe_extract_tar(tar_bytes: bytes, destination: Path) -> None:
    """Extract a tarball while preventing path traversal."""
    destination = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            member_path = (destination / member.name).resolve()
            if not str(member_path).startswith(f"{destination}{os.sep}") and member_path != destination:
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(destination)


def _build_input_tar(input_path: Path) -> tuple[bytes, str]:
    """Package a local input file or directory for remote execution."""
    buffer = io.BytesIO()
    arcname = input_path.name
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        if input_path.is_file():
            tar.add(input_path, arcname=arcname)
        else:
            tar.add(input_path, arcname=arcname)
    return buffer.getvalue(), arcname


def _ensure_local_input_path(input_path: Path) -> Path:
    """Validate and normalize the user-provided input path."""
    normalized = input_path.expanduser().resolve()
    if not normalized.exists():
        raise FileNotFoundError(
            f"Input path does not exist: {normalized}\n"
            "Please pass a valid absolute or relative path to a .pdb/.cif file or a directory."
        )
    if not os.access(normalized, os.R_OK):
        raise PermissionError(f"Input path is not readable: {normalized}")
    if normalized.is_file() and normalized.suffix.lower() not in {".pdb", ".cif"}:
        raise ValueError(f"Input file must end with .pdb or .cif: {normalized}")
    if not normalized.is_file() and not normalized.is_dir():
        raise ValueError(f"Input path must be a file or directory: {normalized}")
    return normalized


def _ensure_local_output_dir(output_path: Path) -> Path:
    """Create and validate the local output directory."""
    normalized = output_path.expanduser().resolve()
    if normalized.exists() and not normalized.is_dir():
        raise NotADirectoryError(
            f"Output path exists but is not a directory: {normalized}"
        )
    normalized.mkdir(parents=True, exist_ok=True)
    if not os.access(normalized, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {normalized}")
    return normalized


def _validate_checkpoint_dir_arg(checkpoint_dir: str) -> str | None:
    """Reject host-local checkpoint paths that are not visible inside Modal."""
    value = checkpoint_dir.strip()
    if not value:
        return None

    checkpoint_path = Path(value).expanduser()
    if checkpoint_path.exists():
        resolved = checkpoint_path.resolve()
        raise ValueError(
            "The provided --checkpoint-dir points to a local host path, but Modal functions "
            "cannot read arbitrary host-local directories.\n"
            f"Received local path: {resolved}\n"
            f"Use the default checkpoint volume ({CHECKPOINTS_DIR}) or pass a path that already "
            "exists inside the Modal container."
        )
    return value


def _bootstrap_local_package(repo_dir: str) -> None:
    """Make the repo importable as `protenixscore` regardless of checkout dir name."""
    if "protenixscore" in sys.modules:
        return

    package_init = Path(repo_dir) / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "protenixscore",
        package_init,
        submodule_search_locations=[repo_dir],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {package_init}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["protenixscore"] = module
    spec.loader.exec_module(module)


##########################################
# Remote functions
##########################################
@app.function(
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={
        CHECKPOINTS_DIR: CHECKPOINTS_VOLUME,
        DATA_DIR: DATA_VOLUME,
        OUTPUTS_DIR: OUTPUTS_VOLUME,
    },
    image=runtime_image,
)
def run_protenixscore(
    *,
    input_name: str,
    input_tar_bytes: bytes,
    run_name: str,
    model_name: str = DEFAULT_MODEL_NAME,
    extra_args: str | None = None,
    checkpoint_dir: str | None = None,
) -> bytes:
    """Run ProtenixScore remotely and return outputs as a tar.gz archive."""
    import pickle
    import shutil
    import urllib.request

    _bootstrap_local_package(REPO_DIR)

    from protenix.web_service.dependency_url import URL
    from protenixscore.cli import _parse_globs, build_parser
    from protenixscore.score import _sanitize_name, run_score

    def _validate_nonempty_file(path: Path, min_size: int = 1) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size < min_size:
            raise RuntimeError(f"File is smaller than expected: {path}")

    def _validate_pickle_file(path: Path) -> None:
        _validate_nonempty_file(path, min_size=1024)
        with path.open("rb") as handle:
            pickle.load(handle)

    def download_file(url: str, destination: Path, validator=None) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"🧬 Downloading asset: {url} -> {destination}")
        tmp_path = destination.with_name(f"{destination.name}.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        if validator is not None:
            validator(tmp_path)
        tmp_path.replace(destination)

    def ensure_valid_asset(
        *,
        label: str,
        url: str,
        destination: Path,
        validator,
    ) -> bool:
        if destination.exists():
            try:
                validator(destination)
                print(f"🧬 Reusing data file: {destination}")
                return False
            except Exception as exc:
                print(f"🧬 Corrupted {label}, re-downloading {destination}: {exc}")
                destination.unlink()

        download_file(url, destination, validator=validator)
        return True

    def materialize_runtime_data_root(data_root: Path) -> None:
        runtime_root = Path("/root/common")
        runtime_root.mkdir(parents=True, exist_ok=True)

        runtime_map = {
            data_root / "components.v20240608.cif": runtime_root / "components.cif",
            data_root / "components.v20240608.cif.rdkit_mol.pkl": runtime_root / "components.cif.rdkit_mol.pkl",
            data_root / "clusters-by-entity-40.txt": runtime_root / "clusters-by-entity-40.txt",
            data_root / "obsolete_release_date.csv": runtime_root / "obsolete_release_date.csv",
        }
        for source_path, target_path in runtime_map.items():
            if target_path.exists() or target_path.is_symlink():
                target_path.unlink()
            try:
                target_path.symlink_to(source_path)
            except FileExistsError:
                continue
            except OSError:
                shutil.copy2(source_path, target_path)

    def materialize_runtime_checkpoint_root(selected_model: str, checkpoint_root: Path) -> None:
        runtime_root = Path("/root/checkpoint")
        runtime_root.mkdir(parents=True, exist_ok=True)

        source_path = checkpoint_root / f"{selected_model}.pt"
        target_path = runtime_root / source_path.name
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()
        try:
            target_path.symlink_to(source_path)
        except FileExistsError:
            return
        except OSError:
            shutil.copy2(source_path, target_path)

    def ensure_runtime_assets(selected_model: str, selected_checkpoint_dir: str) -> None:
        checkpoint_root = Path(selected_checkpoint_dir)
        checkpoint_root.mkdir(parents=True, exist_ok=True)

        model_path = checkpoint_root / f"{selected_model}.pt"
        if not model_path.exists():
            if selected_model not in URL:
                raise KeyError(f"Unknown Protenix model name: {selected_model}")
            download_file(URL[selected_model], model_path)
            CHECKPOINTS_VOLUME.commit()
        else:
            print(f"🧬 Reusing checkpoint: {model_path}")
        materialize_runtime_checkpoint_root(selected_model, checkpoint_root)

        data_root = Path(DATA_DIR)
        data_root.mkdir(parents=True, exist_ok=True)
        data_files = {
            "ccd_components_file": (
                data_root / "components.v20240608.cif",
                lambda path: _validate_nonempty_file(path, min_size=1024),
            ),
            "ccd_components_rdkit_mol_file": (
                data_root / "components.v20240608.cif.rdkit_mol.pkl",
                _validate_pickle_file,
            ),
            "pdb_cluster_file": (
                data_root / "clusters-by-entity-40.txt",
                lambda path: _validate_nonempty_file(path, min_size=64),
            ),
        }
        downloaded_any = False
        for key, (target_path, validator) in data_files.items():
            downloaded_any = ensure_valid_asset(
                label=key,
                url=URL[key],
                destination=target_path,
                validator=validator,
            ) or downloaded_any
        obsolete_release_date_path = data_root / "obsolete_release_date.csv"
        downloaded_any = ensure_valid_asset(
            label="obsolete_release_date.csv",
            url=OBSOLETE_RELEASE_DATE_URL,
            destination=obsolete_release_date_path,
            validator=lambda path: _validate_nonempty_file(path, min_size=64),
        ) or downloaded_any
        if downloaded_any:
            DATA_VOLUME.commit()
        materialize_runtime_data_root(data_root)

    checkpoint_root = checkpoint_dir or CHECKPOINTS_DIR
    ensure_runtime_assets(model_name, checkpoint_root)

    remote_root = Path(OUTPUTS_DIR) / run_name
    remote_input_dir = remote_root / "inputs"
    remote_output_dir = remote_root / "results"
    remote_input_dir.mkdir(parents=True, exist_ok=True)
    remote_output_dir.mkdir(parents=True, exist_ok=True)

    _safe_extract_tar(input_tar_bytes, remote_input_dir)
    remote_input_path = remote_input_dir / Path(input_name).name

    parser = build_parser()
    argv = [
        "score",
        "--input",
        str(remote_input_path),
        "--output",
        str(remote_output_dir),
        "--model_name",
        model_name,
        "--checkpoint_dir",
        checkpoint_root,
    ]
    if extra_args:
        argv.extend(shlex.split(extra_args))

    print(f"🧬 Input path: {remote_input_path}")
    print(f"🧬 Output path: {remote_output_dir}")
    print(f"🧬 Executing command: python -m protenixscore {' '.join(shlex.quote(arg) for arg in argv)}")

    try:
        args = parser.parse_args(argv)
        args.glob = _parse_globs(args.glob)
        if args.failed_log is None:
            args.failed_log = str(Path(args.output) / "failed_records.txt")
        if args.aggregate_csv is None:
            args.aggregate_csv = str(Path(args.output) / "summary.csv")

        run_score(args)
    except Exception as exc:
        print(f"🧬 ProtenixScore failed for input {remote_input_path}: {exc}")
        raise RuntimeError(f"ProtenixScore execution failed: {exc}") from exc

    sample_name = _sanitize_name(remote_input_path.stem)
    sample_dir = remote_output_dir / sample_name
    failed_log_path = Path(args.failed_log)
    if failed_log_path.exists() and not sample_dir.exists():
        failure_text = failed_log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            "ProtenixScore did not produce a successful result.\n"
            f"Failed log: {failed_log_path}\n{failure_text}"
        )

    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as tar:
        for child in sorted(remote_output_dir.iterdir()):
            tar.add(child, arcname=child.name)

    OUTPUTS_VOLUME.commit()
    return archive_buffer.getvalue()


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def main(
    input_pdb: str,
    out_dir: str,
    model_name: str = DEFAULT_MODEL_NAME,
    extra_args: str = "",
    checkpoint_dir: str = "",
) -> None:
    """Run ProtenixScore on Modal and download outputs into `out_dir`."""
    input_path = _ensure_local_input_path(Path(input_pdb))
    local_out_dir = _ensure_local_output_dir(Path(out_dir))
    checkpoint_dir_value = _validate_checkpoint_dir_arg(checkpoint_dir)

    run_name = f"{input_path.stem}-{uuid.uuid4().hex[:8]}"
    print(f"🧬 Local input path: {input_path}")
    print(f"🧬 Local output path: {local_out_dir}")
    if extra_args:
        print(f"🧬 Extra score args: {extra_args}")
    if checkpoint_dir_value:
        print(f"🧬 Remote checkpoint dir override: {checkpoint_dir_value}")

    input_tar_bytes, input_name = _build_input_tar(input_path)

    try:
        result_tar = run_protenixscore.remote(
            input_name=input_name,
            input_tar_bytes=input_tar_bytes,
            run_name=run_name,
            model_name=model_name,
            extra_args=extra_args or None,
            checkpoint_dir=checkpoint_dir_value,
        )
    except Exception as exc:
        raise RuntimeError(
            "Modal execution failed before results could be downloaded.\n"
            f"Input: {input_path}\n"
            f"Output directory: {local_out_dir}\n"
            f"Original error: {exc}"
        ) from exc

    try:
        _safe_extract_tar(result_tar, local_out_dir)
    except Exception as exc:
        raise RuntimeError(
            "Modal finished, but extracting results into the local output directory failed.\n"
            f"Output directory: {local_out_dir}\n"
            f"Original error: {exc}"
        ) from exc

    print(f"🧬 Results saved to: {local_out_dir}")
