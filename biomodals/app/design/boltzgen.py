"""BoltzGen for VHH generation."""
# Ignore ruff warnings about import location and unsafe subprocess usage
# ruff: noqa: PLC0415, S603

import os
from datetime import UTC
from pathlib import Path

from modal import App, Image, Volume

##########################################
# https://modal.com/docs/guide/gpu
GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", "120"))  # for inputs and startup in seconds

# Volume for model cache
BOLTZGEN_VOLUME_NAME = "boltzgen-models"
BOLTZGEN_VOLUME = Volume.from_name(BOLTZGEN_VOLUME_NAME, create_if_missing=True)
BOLTZGEN_MODEL_DIR = "/boltzgen-models"

# Volume for outputs
OUTPUTS_VOLUME_NAME = "boltzgen-outputs"
OUTPUTS_VOLUME = Volume.from_name(OUTPUTS_VOLUME_NAME, create_if_missing=True)
OUTPUTS_DIR = "/boltzgen-outputs"

# Repositories and commit hashes
BOLTZGEN_REPO = "https://github.com/HannesStark/boltzgen"
BOLTZGEN_COMMIT = "6a82850a6e8f8b334d8202822395f95725c02904"
BOLTZGEN_REPO_DIR = "/opt/boltzgen"

##########################################

download_image = (
    Image.debian_slim()
    .pip_install("huggingface_hub[hf_transfer]==0.26.3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # speed up downloads
)

runtime_image = (
    Image.debian_slim()
    .apt_install("git", "build-essential", "wget")
    .env(
        {
            # "UV_COMPILE_BYTECODE": "1",  # slower image build, faster runtime
            # https://modal.com/docs/guide/cuda
            "UV_TORCH_BACKEND": "cu128",  # find best torch and CUDA versions
            "HF_HOME": str(BOLTZGEN_MODEL_DIR),  # store boltzgen model weights
        }
    )
    .run_commands(
        " && ".join(
            (
                f"git clone {BOLTZGEN_REPO} {BOLTZGEN_REPO_DIR}",
                f"cd {BOLTZGEN_REPO_DIR}",
                f"git checkout {BOLTZGEN_COMMIT}",
                "uv venv --python 3.12",
                "uv pip install .",
            ),
        )
    )
    .env({"PATH": f"{BOLTZGEN_REPO_DIR}/.venv/bin:$PATH"})
    .workdir(BOLTZGEN_REPO_DIR)
)

app = App("BoltzGen", image=runtime_image)


@app.function(
    volumes={BOLTZGEN_MODEL_DIR: BOLTZGEN_VOLUME}, timeout=TIMEOUT, image=download_image
)
def boltzgen_download(force: bool = False) -> None:
    """Download BoltzGen models into the mounted volume."""
    import subprocess

    # Download all artifacts (~/.cache overridden to volume mount)
    print("Downloading boltzgen models...")
    cmd = ["boltzgen", "download", "all", "--cache", BOLTZGEN_MODEL_DIR]
    if force:
        cmd.append("--force_download")
    subprocess.run(cmd, check=True, cwd=BOLTZGEN_REPO_DIR)
    print("Model download complete")

    BOLTZGEN_VOLUME.commit()


@app.function(timeout=TIMEOUT, volumes={OUTPUTS_DIR: OUTPUTS_VOLUME})
def prepare_boltzgen_run(
    yaml_content: bytes, run_name: str, additional_files: dict[str, bytes]
) -> None:
    """Prepare BoltzGen input and output directories."""
    workdir = Path(OUTPUTS_DIR) / run_name
    for d in ("inputs", "outputs"):
        (workdir / d).mkdir(parents=True, exist_ok=True)

    # Write yaml to file
    conf_path = workdir / "inputs" / "config"
    conf_path.mkdir(parents=True, exist_ok=True)
    (conf_path / "design-specs.yaml").write_bytes(yaml_content)

    # Write any additional files (e.g., .cif files referenced in yaml)
    for rel_path, content in additional_files.items():
        file_path = conf_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)


@app.function(cpu=1.0, timeout=TIMEOUT, volumes={OUTPUTS_DIR: OUTPUTS_VOLUME})
def collect_boltzgen_data(run_name: str, num_parallel_runs: int):
    """Collect BoltzGen output data from multiple runs."""
    from uuid import uuid4

    workdir = Path(OUTPUTS_DIR) / run_name
    run_ids = [uuid4().hex for _ in range(num_parallel_runs)]

    # TODO: submit tasks to `boltzgen_run` and collect outputs


@app.function(
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={OUTPUTS_DIR: OUTPUTS_VOLUME, BOLTZGEN_MODEL_DIR: BOLTZGEN_VOLUME},
)
def boltzgen_run(
    out_dir: str | Path,
    input_yaml_path: str | Path,
    protocol: str = "protein-anything",
    num_designs: int = 10,
    steps: str | None = None,
    extra_args: str | None = None,
) -> list:
    """Run BoltzGen on a yaml specification.

    Args:
        out_dir: Output directory path
        input_yaml_path: Path to YAML design specification file
        protocol: Design protocol (protein-anything, peptide-anything, etc.)
        num_designs: Number of designs to generate
        steps: Specific pipeline steps to run (e.g. "design inverse_folding")
        extra_args: Additional CLI arguments as string

    Returns:
    -------
        List of (path, content) tuples for all output files
    """
    from subprocess import run

    # Build command
    cmd = [
        "boltzgen",
        "run",
        str(input_yaml_path),
        "--protocol",
        protocol,
        "--output",
        out_dir,
        "--num_designs",
        str(num_designs),
    ]

    if steps:
        cmd.extend(["--steps", *steps.split()])
    if extra_args:
        cmd.extend(extra_args.split())

    print(f"Running: {' '.join(cmd)}")
    run(cmd, check=True, cwd=BOLTZGEN_REPO_DIR)


class YAMLReferenceLoader:
    """Class to load referenced files from YAML files.

    BoltzGen configs might reference other cif or yaml files.
    We need to recursively parse all yaml files to find all used cif templates.

    The file paths need to be relative to the parent directory of the
    input yaml, because we need to recreate the file structure on the remote.
    """

    def __init__(self, input_yaml_file: str | Path) -> None:
        """Initialize the loader with the input YAML file path."""
        self.input_path = Path(input_yaml_file).expanduser().resolve()
        self.ref_dir = self.input_path.parent

        # key: relative path to self.ref_dir, value: file content bytes
        self.additional_files: dict[str, bytes] = {}

        # absolute paths for tracking and recursive parsing
        self.parsed_files: set[Path] = set()
        self.queue: set[Path] = set()
        self.queue.add(self.input_path)
        self.load()

    def load(self) -> None:
        """Load referenced files from a YAML."""
        while self.queue:
            file = self.queue.pop()
            if file in self.parsed_files:
                continue

            new_ref_files = self.find_paths_from_yaml(file)
            for ref_file in new_ref_files:
                ref_path = file.parent.joinpath(ref_file).resolve()
                if ref_path.exists():
                    rel_path = ref_path.relative_to(self.ref_dir, walk_up=True)
                    self.additional_files[str(rel_path)] = ref_path.read_bytes()
                if (
                    ref_path.suffix in {".yaml", ".yml"}
                    and ref_path not in self.parsed_files
                ):
                    self.queue.add(ref_path)

    def find_paths_from_yaml(self, yaml_file: Path) -> set[Path]:
        """Load referenced files from a YAML."""
        import yaml

        yaml_path = Path(yaml_file).expanduser().resolve()
        if yaml_path in self.parsed_files:
            return set()

        with yaml_path.open() as f:
            conf = yaml.safe_load(f)

        file_refs: set[Path] = set()
        self.find_paths_in_dict(conf, yaml_path.parent, file_refs)
        self.parsed_files.add(yaml_path)
        return file_refs

    def find_paths_in_dict(
        self, yaml_content: dict, ref_dir: Path, file_refs: set[Path]
    ) -> None:
        """Recursively find all file references in the yaml content."""
        for v in yaml_content.values():
            if isinstance(v, str):
                if (p := (ref_dir / v)).exists():
                    file_refs.add(p)
            elif isinstance(v, list):
                self.find_paths_in_list(v, ref_dir, file_refs)
            elif isinstance(v, dict):
                self.find_paths_in_dict(v, ref_dir, file_refs)
            else:
                continue

    def find_paths_in_list(
        self, sublist: list, ref_dir: Path, file_refs: set[Path]
    ) -> None:
        """Recursively find all file references in the yaml content."""
        for item in sublist:
            if isinstance(item, str):
                if (p := (ref_dir / item)).exists():
                    file_refs.add(p)
            elif isinstance(item, dict):
                self.find_paths_in_dict(item, ref_dir, file_refs)
            elif isinstance(item, list):
                self.find_paths_in_list(item, ref_dir, file_refs)
            else:
                continue


def submit_boltzgen_task(
    input_yaml: str,
    out_dir: str | Path | None = None,
    num_parallel_runs: int = 1,
    download_models: bool = False,
    force_redownload: bool = False,
    protocol: str = "nanobody-anything",
    num_designs: int = 10,
    steps: str | None = None,
    extra_args: str | None = None,
) -> None:
    """Run BoltzGen locally with results saved to out_dir.

    Args:
        input_yaml: Path to YAML design specification file
        out_dir: Local output directory; defaults to $PWD
        num_parallel_runs: Number of parallel runs to submit
        download_models: Whether to download model weights before running
        force_redownload: Whether to force re-download of model weights
        protocol: Design protocol, one of: protein-anything, peptide-anything,
            protein-small_molecule, or nanobody-anything
        num_designs: Number of designs to generate
        steps: Specific pipeline steps to run (e.g. "design inverse_folding")
        extra_args: Additional CLI arguments as string
    """
    from datetime import datetime

    if download_models:
        boltzgen_download.remote(force=force_redownload)

    # Find any file references in the yaml (path: something.cif)
    # File paths in yaml are relative to the yaml file location
    print("Checking if input yaml references additional files...")
    yaml_path = Path(input_yaml)
    yml_parser = YAMLReferenceLoader(yaml_path)
    if yml_parser.additional_files:
        print(
            f"Including additional referenced files: {list(yml_parser.additional_files.keys())}"
        )

    print(f"Submitting BoltzGen run for yaml: {input_yaml}")
    yaml_str = yaml_path.read_bytes()
    today: str = datetime.now(UTC).strftime("%Y%m%d")
    run_name = f"{today}-{yaml_path.stem}"

    if out_dir is None:
        out_dir = Path.cwd()

    local_out_dir = Path(out_dir).expanduser().resolve() / run_name
    local_out_dir.mkdir(parents=True, exist_ok=True)

    prepare_boltzgen_run.remote(
        yaml_content=yaml_str,
        run_name=run_name,
        additional_files=yml_parser.additional_files,
    )

    print("Running BoltzGen...")
    outputs = boltzgen_run.remote()

    for out_file, out_content in outputs:
        output_path = local_out_dir / out_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(out_content)

    print(f"Results saved to: {local_out_dir}")


main = app.local_entrypoint()(submit_boltzgen_task)
