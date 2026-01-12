"""LigandMPNN source repo: <https://github.com/dauparas/LigandMPNN>.

## Model checkpoints

See <https://github.com/dauparas/LigandMPNN#available-models> for details.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--input-struct` | **Required** | Path to structure coordinates file. |
| `--out-dir` | `$CWD` | Optional local output directory. If not specified, outputs will be saved in a Modal volume only. |

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `MODAL_APP` | `LigandMPNN` | Name of the Modal app to use. |
| `GPU` | `L40S` | Type of GPU to use. See https://modal.com/docs/guide/gpu for details. |
| `TIMEOUT` | `1800` | Timeout for each Modal function in seconds. |

## Outputs

* Results will be saved to the specified `--out-dir` under a subdirectory named after the `--run-name`.
"""

# Ignore ruff warnings about import location and unsafe subprocess usage
# ruff: noqa: PLC0415, S603
import os
from pathlib import Path

from modal import App, Image, Volume

##########################################
# Modal configs
##########################################
# https://modal.com/docs/guide/gpu
GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", "1800"))  # for inputs and startup in seconds
APP_NAME = os.environ.get("MODAL_APP", "LigandMPNN")

# Volume for model cache
LIGANDMPNN_VOLUME_NAME = "ligandmpnn-models"
LIGANDMPNN_VOLUME = Volume.from_name(LIGANDMPNN_VOLUME_NAME, create_if_missing=True)

# Volume for outputs
OUTPUTS_VOLUME_NAME = "ligandmpnn-outputs"
OUTPUTS_VOLUME = Volume.from_name(
    OUTPUTS_VOLUME_NAME, create_if_missing=True, version=2
)
OUTPUTS_DIR = "/ligandmpnn-outputs"

# Repositories and commit hashes
REPO_URL = "https://github.com/dauparas/LigandMPNN"
REPO_COMMIT = "26ec57ac976ade5379920dbd43c7f97a91cf82de"
REPO_DIR = "/opt/LigandMPNN"

##########################################
# Image and app definitions
##########################################
runtime_image = (
    Image.debian_slim()
    .apt_install("git", "build-essential", "zstd")
    .env(
        {
            # "UV_COMPILE_BYTECODE": "1",  # slower image build, faster runtime
            # https://modal.com/docs/guide/cuda
            "UV_TORCH_BACKEND": "cu121",  # find best torch and CUDA versions
        }
    )
    .run_commands(
        " && ".join(
            (
                f"git clone {REPO_URL} {REPO_DIR}",
                f"cd {REPO_DIR}",
                f"git checkout {REPO_COMMIT}",
                "uv venv --python 3.11",
                "uv pip install -r requirements.txt",
            ),
        )
    )
    .env({"PATH": f"{REPO_DIR}/.venv/bin:$PATH"})
    .run_commands("uv pip install polars[pandas,numpy,calamine,xlsxwriter] tqdm")
    .apt_install("wget", "fd-find")
    .workdir(REPO_DIR)
)

app = App(APP_NAME, image=runtime_image)


##########################################
# Helper functions
##########################################
def run_command(cmd: list[str], **kwargs) -> None:
    """Run a shell command and stream output to stdout."""
    import subprocess as sp

    print(f"Running command: {' '.join(cmd)}")
    # Set default kwargs for sp.Popen
    kwargs.setdefault("stdout", sp.PIPE)
    kwargs.setdefault("stderr", sp.STDOUT)
    kwargs.setdefault("bufsize", 1)
    kwargs.setdefault("encoding", "utf-8")

    with sp.Popen(cmd, **kwargs) as p:
        if p.stdout is None:
            raise RuntimeError("Failed to capture stdout from the command.")

        buffered_output = None
        while (buffered_output := p.stdout.readline()) != "" or p.poll() is None:
            print(buffered_output, end="", flush=True)

        if p.returncode != 0:
            raise sp.CalledProcessError(p.returncode, cmd, buffered_output)


##########################################
# Fetch model weights
##########################################
@app.function(
    volumes={f"{REPO_DIR}/model_params": LIGANDMPNN_VOLUME},
    timeout=TIMEOUT,
    image=runtime_image,
)
def download_weights() -> None:
    """Download ProteinMPNN models into the mounted volume."""
    print("Downloading boltzgen models...")
    cmd = ["bash", f"{REPO_DIR}/get_model_params.sh", f"{REPO_DIR}/model_params"]

    run_command(cmd, cwd=REPO_DIR)
    LIGANDMPNN_VOLUME.commit()
    print("Model download complete")


##########################################
# Inference functions
##########################################
@app.function(
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=86400,
    volumes={OUTPUTS_DIR: OUTPUTS_VOLUME},
    image=runtime_image,
)
def collect_ligandmpnn_data(input_struct_bytes: bytes, run_args: dict[str, str]) -> str:
    """Collect BoltzGen output data from multiple runs."""
    from datetime import UTC, datetime
    from uuid import uuid4

    outdir = Path(OUTPUTS_DIR) / run_name / "outputs"
    if salvage_mode:
        all_run_dirs = [d for d in outdir.iterdir() if d.is_dir()]
        run_dirs = [
            d
            for d in all_run_dirs
            if not (
                (d_final_dir := (d / "final_ranked_designs")).exists()
                and (d_final_dir / "results_overview.pdf").exists()
            )
        ]
        run_ids = [d.name for d in all_run_dirs]
    else:
        today: str = datetime.now(UTC).strftime("%Y%m%d")
        run_dirs = [outdir / f"{today}-{uuid4().hex}" for _ in range(num_parallel_runs)]
        run_ids = [d.name for d in run_dirs]

    kwargs = {
        "input_yaml_path": str(
            outdir.parent / "inputs" / "config" / f"{run_name}.yaml"
        ),
        "protocol": protocol,
        "num_designs": num_designs,
        "steps": steps,
        "extra_args": extra_args,
    }
    cli_args_json_path = outdir.parent / "inputs" / "config" / "cli-args.json"
    if not cli_args_json_path.exists():
        import json

        # Save a copy of the CLI args for reference
        with cli_args_json_path.open("w") as f:
            json.dump(kwargs, f, indent=2)

    if run_dirs:
        for boltzgen_dir in ligandmpnn_run.map(run_dirs, kwargs=kwargs):
            print(f"BoltzGen run completed: {boltzgen_dir}")

    OUTPUTS_VOLUME.reload()
    if refilter_results:
        # Rerun BoltzGen filters on all run IDs, and only download the designs
        # that passed all filters (also limited by the `budget`)
        print("Collecting BoltzGen outputs...")
        combine_multiple_runs.remote(run_name)
        refilter_designs.remote(run_name)
        OUTPUTS_VOLUME.reload()

        tarball_bytes = package_outputs.remote(
            outdir,
            run_ids,
            tar_args=[
                "--exclude",
                "intermediate_designs",  # intermediate_designs_inverse_folded is enough
                "--exclude",
                "lightning_logs",
                "--exclude",
                "metrics_tmp",  # design_seq, ca_coords, ca_coords_refolded
            ],
        )
        print("Packaging complete.")
        return tarball_bytes
    else:
        print("Skipping refiltering of BoltzGen outputs.")
        print(
            f"Results are available at: '{outdir.relative_to(OUTPUTS_DIR)}' in volume '{OUTPUTS_VOLUME_NAME}'."
        )
        return run_ids


@app.function(
    gpu=GPU,
    cpu=1.125,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=86400,
    volumes={
        f"{REPO_DIR}/model_params": LIGANDMPNN_VOLUME.read_only(),
        OUTPUTS_DIR: OUTPUTS_VOLUME,
    },
    image=runtime_image,
)
def ligandmpnn_run(cli_args: dict[str, str]) -> str:
    """Run BoltzGen on a yaml specification.

    Args:
        out_dir: Output directory path
        input_yaml_path: Path to YAML design specification file
        protocol: Design protocol (protein-anything, peptide-anything, etc.)
        num_designs: Number of designs to generate
        steps: Specific pipeline steps to run (e.g. "design inverse_folding")
        extra_args: Additional CLI arguments as string

    Returns:
        Path to output directory as a string.
    """
    import subprocess as sp
    import time
    from datetime import UTC, datetime

    # Build command
    cmd = ["python", f"{REPO_DIR}/run.py"]
    for arg, val in cli_args.items():
        cmd.extend([f"--{arg.replace('_', '-')}", str(val)])

    out_path = Path(cli_args["out_folder"])
    out_path.mkdir(parents=True, exist_ok=True)
    log_path = out_path / "ligandmpnn-run.log"
    print(f"Running LigandMPNN, saving logs to {log_path}")
    with (
        sp.Popen(
            cmd,
            bufsize=8,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            encoding="utf-8",
            cwd=REPO_DIR,
        ) as p,
        open(log_path, "a", buffering=1) as log_file,
    ):
        now = time.time()
        banner = "=" * 100
        log_file.write(f"\n{banner}\nTime: {str(datetime.now(UTC))}\n")
        log_file.write(f"Running command: {' '.join(cmd)}\n{banner}\n")

        while (buffered_output := p.stdout.readline()) != "" or p.poll() is None:
            log_file.write(buffered_output)  # not realtime without volume commit
            print(buffered_output)

        log_file.write(f"\n{banner}\nFinished at: {str(datetime.now(UTC))}\n")
        log_file.write(f"Elapsed time: {time.time() - now:.2f} seconds\n")

        if p.returncode != 0:
            print(f"BoltzGen run failed. Error log is in {log_path}")
            raise sp.CalledProcessError(p.returncode, cmd)

    OUTPUTS_VOLUME.commit()
    return str(out_dir)


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_ligandmpnn_task(
    # Input and output
    input_struct: str,
    out_dir: str,
    run_name: str | None = None,
    num_designs: int = 10,
    download_models: bool = False,
    # Run configuration
    model_type: str = "soluble_mpnn",
    checkpoint: str | None = None,
    fixed_residues: str | None = None,
    redesigned_residues: str | None = None,
    bias_aa: str | None = None,
) -> None:
    """Run a variant of the MPNN models with results saved to `out_dir`.

    Args:
        input_struct: Path to YAML design specification file
        out_dir: Local output directory; defaults to $PWD
        num_designs: Number of designs to generate
        download_models: Whether to download model weights and skip running

        protocol: One of: protein_mpnn, ligand_mpnn, per_residue_label_membrane_mpnn,
            global_label_membrane_mpnn, soluble_mpnn
        checkpoint: Optional path to model weights. Note that the name should match
            the `model_type` specified.
        fixed_residues: Space-separated list of residue to keep fixed, e.g. "A12 A13 A14 B2 B25"
        redesigned_residues: Space-separated list of residues to redesign, e.g. "A15 A16 A17 B3 B4".
            Everything else will be fixed.
        bias_aa: Bias generation of amino acids, e.g. "A:-1.024,P:2.34,C:-12.34"


    """
    from pathlib import Path

    if download_models:
        download_weights.remote()
        return

    print("Running BoltzGen...")
    input_path = Path(input_struct).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input structure file not found: {input_path}")
    if run_name is None:
        run_name = input_path.stem
    struct_bytes = input_path.read_bytes()
    remote_results_dir = collect_ligandmpnn_data.remote(
        run_name=run_name,
        protocol=protocol,
        num_designs=num_designs,
    )
    local_out_dir = Path(out_dir).expanduser().resolve()
    local_out_dir.mkdir(parents=True, exist_ok=True)
    for run_id in outputs:
        run_out_dir: Path = local_out_dir / "outputs" / run_id
        run_out_dir.mkdir(parents=True, exist_ok=True)
        remote_root_dir = f"{run_name}/outputs/{run_id}"
        print(f"Downloading results for run ID {run_id}...")
        for subdir in (
            "boltzgen-run.log",
            f"{run_name}.cif",
            "final_ranked_designs",
            "intermediate_designs_inverse_folded",
        ):
            if (run_out_dir / subdir).exists():
                continue

            run_command(
                [
                    "modal",
                    "volume",
                    "get",
                    OUTPUTS_VOLUME_NAME,
                    f"{remote_root_dir}/{subdir}",
                ],
                cwd=run_out_dir,
            )

    print(f"Results saved to: {local_out_dir}")
