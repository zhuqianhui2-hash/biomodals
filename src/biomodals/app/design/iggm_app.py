"""IgGM source repo: <https://github.com/TencentAI4S/IgGM>.

IgGM is a generative model for functional antibody and nanobody design.

## Input FASTA notes

The FASTA file should contain one to three sequences. Use `X` characters to mark
regions for design, such as CDR loops or framework regions. When using an
antigen structure, the last FASTA sequence header should specify the antigen
chain ID from the PDB file, for example `>A` or `>B`.

## Outputs

Results are saved in the `IgGM-outputs` Modal volume under `<run-name>/`.
If `out_dir` is provided, that run directory is also downloaded locally.

For multiple-chain antigens, use `--task merge_chains` first, then use the
generated merged PDB and FASTA for design.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415
import os
from pathlib import Path

import modal

from biomodals.app.config import AppConfig
from biomodals.app.constant import MODEL_VOLUME
from biomodals.helper import patch_image_for_helper
from biomodals.helper.io import resolve_local_output_dir
from biomodals.helper.shell import (
    run_command,
    run_command_with_log,
    sanitize_filename,
    softlink_dir,
)
from biomodals.helper.volume_run import build_volume_run_paths
from biomodals.helper.web import download_files

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="IgGM",
    repo_url="https://github.com/TencentAI4S/IgGM",
    repo_commit_hash="6b1bd91abdf6785d9bbf59c51d6e0c8880aceb8a",
    package_name="iggm",
    python_version="3.11",
    cuda_version="cu121",
    gpu=os.environ.get("GPU", "A10G"),
    timeout=int(os.environ.get("TIMEOUT", "18000")),
)

REPO_DIR = CONF.git_clone_dir
OUTPUTS_VOLUME = CONF.get_out_volume()
OUTPUTS_VOLUME_NAME = OUTPUTS_VOLUME.name or f"{CONF.name}-outputs"
OUTPUTS_DIR = CONF.output_volume_mountpoint
MODEL_NAMES = (
    "esm_ppi_650m_ab",
    "antibody_design_trunk",
    "antibody_inverse_design_trunk",
    "antibody_fr_design_trunk",
    "igso3_buffer",
)
MODEL_MD5 = {
    "antibody_design_trunk": "975baa1f0f5d9ae5cb7afdd4ed179da7",
    "antibody_fr_design_trunk": "be06510ce1562603f3a76e01a92f8a63",
    "antibody_inverse_design_trunk": "204670ca95ff1bd8116148c1bb28ba95",
    "esm_ppi_650m_ab": "9f332b21296d8182c6159ba7833d3a74",
    "igso3_buffer": "8963fa425002a5a65c0b13ddaa443e9e",
}
VALID_TASKS = {
    "design",
    "inverse_design",
    "affinity_maturation",
    "fr_design",
}
MERGE_CHAINS_TASK = "merge_chains"

##########################################
# Image and app definitions
##########################################
runtime_image = patch_image_for_helper(
    modal.Image.debian_slim(python_version=CONF.python_version)
    .apt_install("git", "wget", "build-essential", "zstd")
    .env(CONF.default_env)
    .uv_pip_install(
        "torch==2.1.2+cu121",
        "torchvision==0.16.2+cu121",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .uv_pip_install(
        "torch_geometric==2.5.2",
        "pyg_lib",
        "torch_scatter",
        "torch_sparse",
        "torch_cluster",
        "torch_spline_conv",
        find_links="https://data.pyg.org/whl/torch-2.1.0+cu121.html",
    )
    .uv_pip_install(
        "numpy==1.23.5",
        "pandas==2.2.2",
        "scipy==1.14.0",
        "scikit-learn==1.5.1",
        "matplotlib==3.9.1",
        "seaborn==0.13.2",
        "termcolor==3.1.0",
        "absl-py==2.1.0",
        "biopython==1.85",
        "polars==1.19.0",
        "openmm==8.3.1",
        "pdbfixer==1.12.0",
        "ml-collections",
        "prody==2.6.1",
    )
    .run_commands(
        f"git clone {CONF.repo_url} {REPO_DIR}",
        f"cd {REPO_DIR} && git checkout {CONF.repo_commit_hash}",
        "cd "
        f"{REPO_DIR} && sed -i "
        "'s/torch.from_numpy(residue\\[atom_name\\].get_coord())/"
        "torch.tensor(residue[atom_name].get_coord(), dtype=torch.float32)/' "
        "IgGM/protein/parser/pdb_parser.py",
    )
)

app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


def parse_epitope(epitope: str | None) -> list[int] | None:
    """Parse a comma-separated epitope residue list."""
    if epitope is None or not epitope.strip():
        return None
    return [int(residue.strip()) for residue in epitope.split(",") if residue.strip()]


def _file_md5(path: Path) -> str:
    """Calculate the MD5 checksum for a local file."""
    import hashlib

    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


##########################################
# Fetch model weights
##########################################
@app.function(
    cpu=(0.125, 8.125),
    timeout=CONF.timeout,
    volumes={CONF.model_volume_mountpoint: MODEL_VOLUME},
    image=runtime_image,
)
def download_iggm_models(force: bool = False) -> None:
    """Download IgGM model weights into the shared model volume."""
    MODEL_VOLUME.reload()
    model_urls = {}
    for name in MODEL_NAMES:
        path = CONF.model_dir / f"{name}.pth"
        if path.exists() and not force:
            actual_md5 = _file_md5(path)
            if actual_md5 == MODEL_MD5[name]:
                continue
            print(f"💊 Removing incomplete IgGM model weight: {path.name}")
            path.unlink()

        model_urls[
            f"https://zenodo.org/records/16909543/files/{name}.pth?download=1"
        ] = path

    if model_urls:
        download_files(
            model_urls,
            force=force,
            progress_bar_desc="Downloading IgGM models",
        )
        for path in model_urls.values():
            name = Path(path).stem
            actual_md5 = _file_md5(Path(path))
            if actual_md5 != MODEL_MD5[name]:
                raise RuntimeError(
                    f"Checksum mismatch for {Path(path).name}: "
                    f"expected {MODEL_MD5[name]}, got {actual_md5}"
                )
    else:
        print(f"💊 IgGM model weights already exist under {CONF.model_dir}")
    MODEL_VOLUME.commit()
    print(f"💊 IgGM model weights are available under {CONF.model_dir}")


def _ensure_model_symlink() -> None:
    """Link IgGM's expected checkpoints directory to the mounted model volume."""
    missing_models = [
        path
        for name in MODEL_NAMES
        if not (path := CONF.model_dir / f"{name}.pth").exists()
    ]
    if missing_models:
        missing_names = ", ".join(path.name for path in missing_models)
        raise FileNotFoundError(
            f"Missing IgGM model weights in {CONF.model_dir}: {missing_names}. "
            "Run this app with --download-models first."
        )

    softlink_dir(CONF.model_dir, REPO_DIR / "checkpoints")


def _download_remote_run(remote_run_dir: str, out_dir: str, label: str) -> None:
    """Download one persisted Modal volume run directory to a local directory."""
    local_out_dir = resolve_local_output_dir(out_dir)
    local_out_dir.mkdir(parents=True, exist_ok=True)
    local_run_dir = local_out_dir / remote_run_dir
    if local_run_dir.exists():
        raise FileExistsError(f"Local output directory already exists: {local_run_dir}")
    run_command(
        ["modal", "volume", "get", OUTPUTS_VOLUME_NAME, remote_run_dir],
        cwd=local_out_dir,
    )
    print(f"🧬 {label} downloaded to: {local_run_dir}")


##########################################
# Inference functions
##########################################
@app.function(
    cpu=(0.125, 8.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes={OUTPUTS_DIR: OUTPUTS_VOLUME},
    image=runtime_image,
)
def merge_pdb_chains(
    antigen_pdb_bytes: bytes,
    run_name: str,
    antibody_ids: str = "H_L",
    merge_ids: str = "A",
) -> str:
    """Merge multiple antigen chains and return the Modal volume run path."""
    OUTPUTS_VOLUME.reload()

    run_paths = build_volume_run_paths(OUTPUTS_DIR, run_name)
    workdir = run_paths["run_root"]
    input_dir = run_paths["inputs_dir"]
    output_dir = run_paths["output_dir"]
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    antigen_path = input_dir / "antigen.pdb"
    antigen_path.write_bytes(antigen_pdb_bytes)

    cmd = [
        "python",
        "scripts/merge_chains.py",
        "--antigen",
        str(antigen_path),
        "--output",
        str(output_dir),
        "--antibody_ids",
        antibody_ids,
        "--merge_ids",
        merge_ids,
    ]

    log_path = workdir / "iggm_merge_chains.log"
    OUTPUTS_VOLUME.commit()
    print("💊 Merging IgGM antigen chains...")
    try:
        run_command_with_log(
            cmd,
            log_file=log_path,
            verbose=True,
            cwd=REPO_DIR,
            env={"PYTHONPATH": str(REPO_DIR)},
        )
    finally:
        OUTPUTS_VOLUME.commit()

    relative_workdir = workdir.relative_to(OUTPUTS_DIR)
    print(
        f"💊 Merged chain results are available at '{relative_workdir}' "
        f"in volume '{OUTPUTS_VOLUME_NAME}'."
    )
    return str(relative_workdir)


@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes={
        CONF.model_volume_mountpoint: MODEL_VOLUME.read_only(),
        OUTPUTS_DIR: OUTPUTS_VOLUME,
    },
    image=runtime_image,
)
def iggm_inference(
    input_fasta_bytes: bytes,
    task: str,
    run_name: str,
    antigen_pdb_bytes: bytes | None = None,
    epitope: list[int] | None = None,
    fasta_origin_bytes: bytes | None = None,
    num_samples: int | None = None,
    relax: bool = False,
    max_antigen_size: int | None = None,
) -> str:
    """Run IgGM and return the Modal volume run path."""
    if task not in VALID_TASKS:
        raise ValueError(f"Task must be one of {sorted(VALID_TASKS)}.")

    _ensure_model_symlink()
    OUTPUTS_VOLUME.reload()

    run_paths = build_volume_run_paths(OUTPUTS_DIR, run_name)
    workdir = run_paths["run_root"]
    input_dir = run_paths["inputs_dir"]
    output_dir = run_paths["output_dir"]
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = input_dir / "input.fasta"
    fasta_path.write_bytes(input_fasta_bytes)

    cmd = ["python", "design.py", "--fasta", str(fasta_path)]

    if antigen_pdb_bytes is not None:
        antigen_path = input_dir / "antigen.pdb"
        antigen_path.write_bytes(antigen_pdb_bytes)
        cmd.extend(["--antigen", str(antigen_path)])

    if epitope is not None:
        cmd.extend(["--epitope", *[str(residue) for residue in epitope]])

    if fasta_origin_bytes is not None:
        origin_path = input_dir / "original.fasta"
        origin_path.write_bytes(fasta_origin_bytes)
        cmd.extend(["--fasta_origin", str(origin_path)])

    cmd.extend(["--run_task", task])

    if num_samples is not None:
        cmd.extend(["--num_samples", str(num_samples)])
    if relax:
        cmd.append("--relax")
    if max_antigen_size is not None:
        cmd.extend(["--max_antigen_size", str(max_antigen_size)])

    cmd.extend(["--output", str(output_dir)])

    log_path = workdir / "iggm.log"
    OUTPUTS_VOLUME.commit()
    print("💊 Running IgGM...")
    try:
        run_command_with_log(
            cmd,
            log_file=log_path,
            verbose=True,
            cwd=REPO_DIR,
            env={"PYTHONPATH": str(REPO_DIR)},
        )
    finally:
        OUTPUTS_VOLUME.commit()

    relative_workdir = workdir.relative_to(OUTPUTS_DIR)
    print(
        f"💊 Results are available at '{relative_workdir}' "
        f"in volume '{OUTPUTS_VOLUME_NAME}'."
    )
    return str(relative_workdir)


##########################################
# Local entrypoints
##########################################
@app.local_entrypoint()
def submit_iggm_task(
    input_fasta: str | None = None,
    task: str = "design",
    antigen: str | None = None,
    epitope: str | None = None,
    fasta_origin: str | None = None,
    antibody_ids: str = "H_L",
    merge_ids: str = "A",
    num_samples: int | None = None,
    relax: bool = False,
    max_antigen_size: int | None = None,
    run_name: str | None = None,
    out_dir: str | None = None,
    download_models: bool = False,
    force_redownload: bool = False,
) -> None:
    """Run IgGM antibody or nanobody design on Modal.

    Args:
        input_fasta: Path to the input antibody or nanobody FASTA file. Use `X`
            characters to mark regions for design. Not required when
            `download_models` is true or `task` is `merge_chains`.
        task: IgGM task to run. One of `design`, `inverse_design`,
            `affinity_maturation`, `fr_design`, or `merge_chains`.
        antigen: Optional path to an antigen PDB file. When provided, the last
            FASTA sequence header should name the antigen chain ID from this PDB.
        epitope: Optional comma-separated antigen epitope residue numbers, for
            example `41,42,43`.
        fasta_origin: Optional path to the original FASTA file for affinity maturation.
        antibody_ids: Underscore-separated antibody chain IDs to keep when
            `task` is `merge_chains`, for example `H_L`.
        merge_ids: Underscore-separated antigen chain IDs to merge into chain
            `A` when `task` is `merge_chains`, for example `A_B_C`.
        num_samples: Optional number of samples to generate. For
            `affinity_maturation`, this is the number of samples per masked
            position, so total candidates are approximately `num_samples`
            multiplied by the number of `X` positions.
        relax: Whether to run IgGM structure relaxation.
        max_antigen_size: Optional maximum antigen length to consider.
        run_name: Name used for the local output archive. Defaults to the input
            FASTA stem.
        out_dir: Optional local directory to download the persisted Modal volume
            run directory after inference. If omitted, results remain in the
            `IgGM-outputs` Modal volume.
        download_models: Download IgGM model weights and exit without inference.
        force_redownload: Force re-download of model weights even if cached files exist.

    """
    if download_models:
        print("🧬 Downloading IgGM model weights...")
        download_iggm_models.remote(force=force_redownload)
        return
    if force_redownload:
        raise ValueError("force_redownload can only be used with download_models.")

    print("🧬 Checking IgGM inputs...")
    if task not in VALID_TASKS | {MERGE_CHAINS_TASK}:
        raise ValueError(
            f"Task must be one of {sorted(VALID_TASKS | {MERGE_CHAINS_TASK})}."
        )

    if task == MERGE_CHAINS_TASK:
        if antigen is None:
            raise ValueError("antigen must be provided when task is merge_chains.")
        antigen_path = Path(antigen).expanduser().resolve()
        if not antigen_path.exists():
            raise FileNotFoundError(f"Antigen PDB file not found: {antigen_path}")
        if run_name is None:
            run_name = f"{antigen_path.stem}_merge_chains"
        run_name = sanitize_filename(run_name)

        print("🧬 Merging IgGM antigen chains on Modal...")
        remote_run_dir = merge_pdb_chains.remote(
            antigen_pdb_bytes=antigen_path.read_bytes(),
            run_name=run_name,
            antibody_ids=antibody_ids,
            merge_ids=merge_ids,
        )

        print(
            f"🧬 Merge-chain results saved to '{remote_run_dir}' "
            f"in Modal volume '{OUTPUTS_VOLUME_NAME}'."
        )
        print(
            f"🧬 Download with: modal volume get {OUTPUTS_VOLUME_NAME} {remote_run_dir}"
        )

        if out_dir is not None:
            _download_remote_run(remote_run_dir, out_dir, "Merge-chain results")
        return

    if input_fasta is None:
        raise ValueError("input_fasta must be provided unless download_models is true.")
    input_fasta_path = Path(input_fasta).expanduser().resolve()
    if not input_fasta_path.exists():
        raise FileNotFoundError(f"Input FASTA file not found: {input_fasta_path}")
    antigen_bytes = None
    if antigen is not None:
        antigen_path = Path(antigen).expanduser().resolve()
        if not antigen_path.exists():
            raise FileNotFoundError(f"Antigen PDB file not found: {antigen_path}")
        antigen_bytes = antigen_path.read_bytes()

    fasta_origin_bytes = None
    if fasta_origin is not None:
        fasta_origin_path = Path(fasta_origin).expanduser().resolve()
        if not fasta_origin_path.exists():
            raise FileNotFoundError(f"Origin FASTA file not found: {fasta_origin_path}")
        fasta_origin_bytes = fasta_origin_path.read_bytes()

    if run_name is None:
        run_name = input_fasta_path.stem
    run_name = sanitize_filename(run_name)

    print("🧬 Running IgGM on Modal...")
    remote_run_dir = iggm_inference.remote(
        input_fasta_bytes=input_fasta_path.read_bytes(),
        task=task,
        run_name=run_name,
        antigen_pdb_bytes=antigen_bytes,
        epitope=parse_epitope(epitope),
        fasta_origin_bytes=fasta_origin_bytes,
        num_samples=num_samples,
        relax=relax,
        max_antigen_size=max_antigen_size,
    )

    print(
        f"🧬 Results saved to '{remote_run_dir}' "
        f"in Modal volume '{OUTPUTS_VOLUME_NAME}'."
    )
    print(f"🧬 Download with: modal volume get {OUTPUTS_VOLUME_NAME} {remote_run_dir}")

    if out_dir is not None:
        _download_remote_run(remote_run_dir, out_dir, "Results")
