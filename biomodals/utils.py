"""Utility functions often used in modal apps."""


def package_outputs(output_dir: str) -> bytes:
    """Package output directory into a tar.gz archive and return as bytes."""
    import io
    import tarfile
    from pathlib import Path

    tar_buffer = io.BytesIO()
    out_path = Path(output_dir)
    with tarfile.open(fileobj=tar_buffer, mode="w:gz", compresslevel=6) as tar:
        tar.add(out_path, arcname=out_path.name)

    return tar_buffer.getvalue()
