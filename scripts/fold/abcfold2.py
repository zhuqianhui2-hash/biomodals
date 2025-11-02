"""Run ABCFold on Modal GPU instances."""
# Ignore ruff warnings about import location and unsafe subprocess usage
# ruff: noqa: PLC0415, S603

import os
from datetime import UTC
from pathlib import Path

from modal import App, Image, Volume

##########################################
# Modal configs
# T4: 16GB, L4: 24GB, A10G: 24GB, L40S: 48GB, A100-40G, A100-80G, H100: 80GB
GPU = os.environ.get("GPU", "A10G")
TIMEOUT = int(os.environ.get("TIMEOUT", "1200"))  # seconds

# Volume for model cache
CHAI_VOLUME_NAME = "chai-models"
CHAI_VOLUME = Volume.from_name(CHAI_VOLUME_NAME, create_if_missing=True)
CHAI_MODEL_DIR = "/chai-models"

BOLTZ_VOLUME_NAME = "boltz-models"
BOLTZ_VOLUME = Volume.from_name(BOLTZ_VOLUME_NAME, create_if_missing=True)
BOLTZ_MODEL_DIR = "/boltz-models"

# Volume for outputs
OUTPUTS_VOLUME_NAME = "abcfold2-outputs"
OUTPUTS_VOLUME = Volume.from_name(OUTPUTS_VOLUME_NAME, create_if_missing=True)
OUTPUTS_DIR = "/abcfold2-outputs"

# Repositories and commit hashes
ABCFOLD_DIR = "/opt/ABCFold"
ABCFOLD_REPO = "https://github.com/y1zhou/ABCFold"
ABCFOLD_COMMIT = "987b1a722e998ca2cddfc28a25a65cd727b7bf10"

CHAI_DIR = "/opt/chai-lab"
CHAI_REPO = "https://github.com/y1zhou/chai-lab"
CHAI_COMMIT = "0ac68311911bfcd28b118fc289437bf3eff8ac97"

BOLTZ_DIR = "/opt/boltz"
BOLTZ_REPO = "https://github.com/jwohlwend/boltz"
BOLTZ_COMMIT = "cb04aeccdd480fd4db707f0bbafde538397fa2ac"
BOLTZ_MODEL_HASH = "6fdef46d763fee7fbb83ca5501ccceff43b85607"  # HF revision
##########################################

download_image = (
    Image.debian_slim()
    .pip_install("huggingface_hub[hf_transfer]==0.26.3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # speed up downloads
)

runtime_image = (
    Image.debian_slim()
    .apt_install("git", "build-essential", "kalign")  # kalign for Chai templates
    .env(
        {
            # "UV_COMPILE_BYTECODE": "1",  # slower image build, faster runtime
            "UV_TORCH_BACKEND": "auto",  # find best torch and CUDA versions
            "CHAI_DOWNLOADS_DIR": str(CHAI_MODEL_DIR),  # store chai model weights
            "BOLTZ_CACHE_DIR": str(BOLTZ_MODEL_DIR),  # store boltz model weights
        }
    )
    .run_commands(
        " && ".join(
            (
                # Clone Boltz and Chai
                f"git clone {BOLTZ_REPO} {BOLTZ_DIR}",
                f"cd {BOLTZ_DIR}",
                f"git checkout {BOLTZ_COMMIT}",
                f"git clone {CHAI_REPO} {CHAI_DIR}",
                f"cd {CHAI_DIR}",
                f"git checkout {CHAI_COMMIT}",
                # Setup ABCFold2 environment
                f"git clone {ABCFOLD_REPO} {ABCFOLD_DIR}",
                f"cd {ABCFOLD_DIR}",
                f"git checkout {ABCFOLD_COMMIT}",
                "uv venv --python 3.12",
                f"uv pip install {BOLTZ_DIR}[cuda] {CHAI_DIR}",
                "uv pip install .",
            ),
        )
    )
    .workdir(ABCFOLD_DIR)
)

app = App("ABCFold2", image=runtime_image)


@app.function(
    volumes={BOLTZ_MODEL_DIR: BOLTZ_VOLUME}, timeout=TIMEOUT, image=download_image
)
def download_boltz_models(force: bool = False) -> None:
    """Download Boltz models into the mounted volume.

    From: https://modal.com/docs/examples/boltz_predict.
    """
    import tarfile

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="boltz-community/boltz-2",
        revision=BOLTZ_MODEL_HASH,
        local_dir=BOLTZ_MODEL_DIR,
        force_download=force,
    )
    tar_mols = Path(BOLTZ_MODEL_DIR) / "mols.tar"
    if not (Path(BOLTZ_MODEL_DIR) / "mols").exists():
        with tarfile.open(str(tar_mols), "r") as tar:
            tar.extractall(BOLTZ_MODEL_DIR)  # noqa: S202

    BOLTZ_VOLUME.commit()


async def download_file(session, url: str, local_path: Path):
    """Download a file asynchronously using aiohttp."""
    async with session.get(url) as response:
        response.raise_for_status()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            while chunk := await response.content.read(8192):
                f.write(chunk)


@app.function(
    volumes={CHAI_MODEL_DIR: CHAI_VOLUME}, timeout=TIMEOUT, image=download_image
)
async def download_chai_models(force=False):
    """From https://modal.com/docs/examples/chai1."""
    import asyncio

    import aiohttp

    base_url = "https://chaiassets.com/chai1-inference-depencencies/"  # sic
    inference_dependencies = [
        "conformers_v1.apkl",
        "models_v2/trunk.pt",
        "models_v2/token_embedder.pt",
        "models_v2/feature_embedding.pt",
        "models_v2/diffusion_module.pt",
        "models_v2/confidence_head.pt",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    # launch downloads concurrently
    chai_model_dir = Path(CHAI_MODEL_DIR)
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        for dep in inference_dependencies:
            local_path = chai_model_dir / dep
            if force or not local_path.exists():
                url = base_url + dep
                print(f"🧬 downloading {dep} to {local_path}")
                tasks.append(download_file(session, url, local_path))

        # run all of the downloads and await their completion
        await asyncio.gather(*tasks)

    CHAI_VOLUME.commit()  # ensures models are visible on remote filesystem before exiting, otherwise takes a few seconds, racing with inference


@app.function(
    image=runtime_image,
    volumes={OUTPUTS_DIR: OUTPUTS_VOLUME, BOLTZ_MODEL_DIR: BOLTZ_VOLUME},
)
def prepare_abcfold2(yaml_str: str, run_id: str) -> None:
    """Prepare inputs to Boltz and Chai using ABCFold2 config."""
    import tempfile

    from abcfold.cli.prepare import prepare_boltz, prepare_chai, search_msa

    out_dir_full: Path = Path(OUTPUTS_DIR) / run_id[:6] / run_id
    out_dir_full.mkdir(parents=True, exist_ok=True)
    yaml_path = Path(out_dir_full) / f"{run_id}.yaml"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_yaml_path = Path(tmpdir) / f"{run_id}.yaml"
        tmp_yaml_path.write_text(yaml_str)

        # Run MSA and template search
        if not yaml_path.exists():
            search_msa(
                conf_file=tmp_yaml_path,
                out_dir=out_dir_full,
                force=True,
                template_cache_dir=f"{OUTPUTS_DIR}/.cache/rcsb",
            )

    # Generate inputs for Boltz and Chai
    prepare_boltz(conf_file=yaml_path, out_dir=out_dir_full)
    prepare_chai(
        conf_file=yaml_path,
        out_dir=out_dir_full,
        ccd_lib_dir=Path(BOLTZ_MODEL_DIR) / "mols",
    )


@app.function(
    image=runtime_image,
    volumes={
        OUTPUTS_DIR: OUTPUTS_VOLUME,
        CHAI_MODEL_DIR: CHAI_VOLUME,
        BOLTZ_MODEL_DIR: BOLTZ_VOLUME,
    },
)
def run_abcfold2(yaml_str: str, run_id: str) -> None:
    """Run ABCFold2 with the given YAML specification string."""
    import subprocess as sp

    # TODO: replace with actual run commands
    out_dir_full: Path = Path(OUTPUTS_DIR) / run_id[:6] / run_id
    out_dir_full.mkdir(parents=True, exist_ok=True)
    yaml_path = Path(out_dir_full) / "input.yaml"
    yaml_path.write_text(yaml_str)

    cmd = ["uv", "run", "abcfold2", "validate", str(yaml_path)]
    sp.run(cmd, cwd=ABCFOLD_DIR, check=True)


@app.local_entrypoint()
def main(
    input_yaml: str,
    run_name: str | None = None,
    download_models: bool = False,
    force_redownload: bool = False,
) -> None:
    """Run BoltzGen locally with results saved to out_dir.

    Args:
        input_yaml: Path to YAML design specification file
        run_name: Optional run name (defaults to timestamp)
        download_models: Whether to download model weights before running
        force_redownload: Whether to force re-download of model weights
    """
    import hashlib
    from datetime import datetime
    from uuid import uuid4

    if download_models:
        print("🧬 Checking Boltz inference dependencies...")
        download_boltz_models.remote(force=force_redownload)

        print("🧬 Checking Chai inference dependencies...")
        download_chai_models.remote(force=force_redownload)

    today: str = datetime.now(UTC).strftime("%Y%m%d%H%M")
    if run_name is None:
        run_name = hashlib.sha256(uuid4().bytes).hexdigest()[:8]  # short id
    run_id: str = f"{today}-{run_name}"

    print(f"🧬 Starting ABCFold2 run {run_id}...")

    yaml_path = Path(input_yaml).expanduser().resolve()
    yaml_str = yaml_path.read_text()
    prepare_abcfold2.remote(yaml_str=yaml_str, run_id=run_id)
