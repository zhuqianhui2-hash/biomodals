"""Tests for FlowPacker workflow-compatible result contracts."""

# ruff: noqa: D103

from pathlib import Path
from types import SimpleNamespace

from biomodals.app.fold import flowpacker_app
from biomodals.schema import AppRunStatus, ArtifactKind, VolumePath


def test_flowpacker_workflow_result_stores_archive_in_volume(
    tmp_path,
    monkeypatch,
) -> None:
    seen_kwargs = {}

    class FakeRunFlowPacker:
        def get_raw_f(self):
            def fake_run_flowpacker(**kwargs):
                seen_kwargs.update(kwargs)
                return b"tarball"

            return fake_run_flowpacker

    class FakeOutputVolume:
        def __init__(self):
            self.commit_count = 0

        def commit(self):
            self.commit_count += 1

    output_volume = FakeOutputVolume()
    output_volume_name = flowpacker_app.CONF.output_volume_name
    monkeypatch.setattr(flowpacker_app, "run_flowpacker", FakeRunFlowPacker())
    monkeypatch.setattr(
        flowpacker_app,
        "CONF",
        SimpleNamespace(
            output_volume=output_volume,
            output_volume_mountpoint=str(tmp_path),
            output_volume_name=output_volume_name,
        ),
    )

    result = flowpacker_app.run_flowpacker_workflow.get_raw_f()(
        input_files=[("input.pdb", b"ATOM\n")],
        run_name="../packed",
    )

    assert seen_kwargs["run_name"] == "packed"
    assert result.status == AppRunStatus.SUCCEEDED
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.name == "flowpacker_outputs"
    assert output.kind == ArtifactKind.ARCHIVE
    assert output.storage == VolumePath(
        volume_name=output_volume_name,
        path="workflow/packed/packed.tar.zst",
        media_type="application/zstd",
    )
    assert output.metadata == {
        "archive_format": "tar.zst",
        "filename": "packed.tar.zst",
    }
    assert (
        Path(tmp_path) / "workflow" / "packed" / "packed.tar.zst"
    ).read_bytes() == b"tarball"
    assert output_volume.commit_count == 1
