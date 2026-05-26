"""Tests for the ShortMD workflow definition."""

# ruff: noqa: D103

from pathlib import Path

import pytest

from biomodals.app.bioinfo import gromacs_app
from biomodals.schema import (
    AppRunStatus,
    ArtifactKind,
    NodePlacement,
    VolumePath,
    WorkflowArtifact,
)
from biomodals.workflow import shortmd_workflow
from biomodals.workflow.core.nodes import NodeRunContext
from biomodals.workflow.shortmd_workflow import (
    ShortMDGromacsSettings,
    ShortMDPrepNode,
    ShortMDReplicateNode,
    ShortMDSummaryNode,
    build_shortmd_workflow,
    clone_prepared_shortmd_run,
    discover_pdb_inputs,
)


def test_shortmd_uses_gromacs_app_volume_metadata() -> None:
    assert shortmd_workflow.GROMACS_APP_NAME == gromacs_app.CONF.name
    assert (
        shortmd_workflow.GROMACS_OUTPUT_MOUNTPOINT
        == gromacs_app.CONF.output_volume_mountpoint
    )
    assert shortmd_workflow.GROMACS_OUTPUT_VOLUME is gromacs_app.OUTPUTS_VOLUME
    assert (
        shortmd_workflow.GROMACS_OUTPUT_VOLUME_NAME == gromacs_app.OUTPUTS_VOLUME_NAME
    )


def test_discover_pdb_inputs_globs_pdb_files(tmp_path: Path) -> None:
    tmp_path.joinpath("b.pdb").write_text("B\n", encoding="utf-8")
    tmp_path.joinpath("a.pdb").write_text("A\n", encoding="utf-8")
    tmp_path.joinpath("ignore.txt").write_text("x\n", encoding="utf-8")

    discovered = discover_pdb_inputs(tmp_path)

    assert set(discovered) == {("a.pdb", b"A\n"), ("b.pdb", b"B\n")}


def test_discover_pdb_inputs_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No PDB files"):
        discover_pdb_inputs(tmp_path)


def test_build_shortmd_workflow_models_prep_replicate_summary_dependencies() -> None:
    workflow = build_shortmd_workflow(
        input_pdbs=[("alpha.pdb", b"ATOM\n"), ("beta.pdb", b"ATOM\n")],
        replicates=2,
        simulation_time_ns=2,
        cpu_only=True,
        max_parallel=8,
    )

    definition = workflow.validate()

    assert workflow.name == "shortmd"
    assert set(definition.nodes) == {
        "prep-alpha",
        "replicate-alpha-r001",
        "replicate-alpha-r002",
        "prep-beta",
        "replicate-beta-r001",
        "replicate-beta-r002",
        "summary",
    }
    assert definition.dependencies["replicate-alpha-r001"] == {"prep-alpha"}
    assert definition.dependencies["replicate-alpha-r002"] == {"prep-alpha"}
    assert definition.dependencies["summary"] == {
        "replicate-alpha-r001",
        "replicate-alpha-r002",
        "replicate-beta-r001",
        "replicate-beta-r002",
    }

    prep_node = definition.nodes["prep-alpha"].node
    replicate_node = definition.nodes["replicate-alpha-r001"].node
    summary_node = definition.nodes["summary"].node

    assert isinstance(prep_node, ShortMDPrepNode)
    assert prep_node.gromacs_app_name == gromacs_app.CONF.name
    assert prep_node.placement == NodePlacement.REMOTE
    assert prep_node.run_name == "alpha"
    assert prep_node.pdb_content == b"ATOM\n"

    assert isinstance(replicate_node, ShortMDReplicateNode)
    assert replicate_node.placement == NodePlacement.REMOTE
    assert replicate_node.gromacs_app_name == gromacs_app.CONF.name
    assert replicate_node.source_run_name == "alpha"
    assert replicate_node.replicate_run_name == "alpha-r001"
    assert replicate_node.gromacs.simulation_time_ns == 2
    assert replicate_node.gromacs.cpu_only is True

    assert isinstance(summary_node, ShortMDSummaryNode)
    assert summary_node.max_parallel == 8


def test_build_shortmd_workflow_rejects_duplicate_sanitized_stems() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        build_shortmd_workflow(
            input_pdbs=[("../a.pdb", b"A\n"), ("a.pdb", b"B\n")],
            replicates=1,
        )


def test_shortmd_prep_node_runs_gromacs_prepare_and_returns_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepare_kwargs = {}

    class FakePrepareFunction:
        def remote(self, **kwargs):
            prepare_kwargs.update(kwargs)
            return f"{gromacs_app.CONF.output_volume_mountpoint}/prepared/source"

    def fake_from_name(app_name, function_name):
        assert (app_name, function_name) == (
            gromacs_app.CONF.name,
            "prepare_tpr_cpu",
        )
        return FakePrepareFunction()

    monkeypatch.setattr(shortmd_workflow.modal.Function, "from_name", fake_from_name)

    node = ShortMDPrepNode(
        pdb_content=b"ATOM\n",
        run_name="../source",
        gromacs=ShortMDGromacsSettings(
            simulation_time_ns=2,
            run_pdbfixer=True,
            cpu_only=True,
            num_threads=8,
            use_openmp_threads=True,
            ld_seed=11,
            gen_seed=12,
            genion_seed=13,
        ),
    )
    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="prep-source",
            attempt_id="attempt-1",
            cache_dir=tmp_path / "cache",
            inputs={},
        )
    )

    assert prepare_kwargs == {
        "pdb_content": b"ATOM\n",
        "run_name": "source",
        "simulation_time_ns": 2,
        "run_pdbfixer": True,
        "num_threads": 8,
        "use_openmp_threads": True,
        "ld_seed": 11,
        "gen_seed": 12,
        "genion_seed": 13,
    }
    assert result.status == AppRunStatus.SUCCEEDED
    assert result.outputs[0].name == "prepared_gromacs_run"
    assert result.outputs[0].kind == ArtifactKind.DIRECTORY
    assert result.outputs[0].storage == VolumePath(
        volume_name=gromacs_app.OUTPUTS_VOLUME_NAME,
        path="prepared/source",
    )
    assert result.outputs[0].metadata == {"stage": "prep", "run_name": "source"}


def test_shortmd_prep_node_rejects_workdir_outside_gromacs_mount(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakePrepareFunction:
        def remote(self, **kwargs):
            return "/outside-gromacs-output"

    monkeypatch.setattr(
        shortmd_workflow.modal.Function,
        "from_name",
        lambda app_name, function_name: FakePrepareFunction(),
    )

    node = ShortMDPrepNode(pdb_content=b"ATOM\n", run_name="source")
    with pytest.raises(ValueError, match="outside"):
        node.run(
            NodeRunContext(
                run_id="run-1",
                node_id="prep-source",
                attempt_id="attempt-1",
                cache_dir=tmp_path / "cache",
                inputs={},
            )
        )


def test_clone_prepared_shortmd_run_copies_prepared_inputs_into_replicate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeOutputVolume:
        def __init__(self) -> None:
            self.commit_count = 0
            self.reload_count = 0

        def commit(self) -> None:
            self.commit_count += 1

        def reload(self) -> None:
            self.reload_count += 1

    output_volume = FakeOutputVolume()
    source_dir = tmp_path / "prepared" / "source"
    source_dir.mkdir(parents=True)
    source_dir.joinpath("source.pdb").write_text("ATOM\n", encoding="utf-8")
    source_dir.joinpath("production_source.tpr").write_text("tpr\n", encoding="utf-8")
    source_dir.joinpath("production_source.xtc").write_text("stale\n", encoding="utf-8")
    source_dir.joinpath("npt_source.gro").write_text("npt\n", encoding="utf-8")

    monkeypatch.setattr(shortmd_workflow, "GROMACS_OUTPUT_MOUNTPOINT", str(tmp_path))
    monkeypatch.setattr(shortmd_workflow, "GROMACS_OUTPUT_VOLUME", output_volume)

    result = clone_prepared_shortmd_run.get_raw_f()(
        source_storage_path="prepared/source",
        source_run_name="source",
        replicate_run_name="source-r001",
    )

    replicate_dir = tmp_path / "source-r001"
    assert result == str(replicate_dir)
    assert replicate_dir.joinpath("source-r001.pdb").read_text(encoding="utf-8") == (
        "ATOM\n"
    )
    assert (
        replicate_dir.joinpath("production_source-r001.tpr").read_text(encoding="utf-8")
        == "tpr\n"
    )
    assert not replicate_dir.joinpath("production_source.xtc").exists()
    assert output_volume.reload_count == 1
    assert output_volume.commit_count == 1


def test_shortmd_replicate_node_clones_then_runs_gromacs_production(
    tmp_path: Path,
    monkeypatch,
) -> None:
    clone_kwargs = {}
    production_kwargs = {}
    stats_kwargs = {}
    function_names = []

    class FakeCloneFunction:
        def remote(self, **kwargs):
            clone_kwargs.update(kwargs)
            return f"{gromacs_app.CONF.output_volume_mountpoint}/source-r001"

    class FakeProductionFunction:
        def remote(self, **kwargs):
            production_kwargs.update(kwargs)
            return f"{gromacs_app.CONF.output_volume_mountpoint}/source-r001"

    class FakeStatsFunction:
        def remote(self, traj_prefix, **kwargs):
            stats_kwargs["traj_prefix"] = traj_prefix
            stats_kwargs.update(kwargs)
            return f"{gromacs_app.CONF.output_volume_mountpoint}/production/source-r001"

    def fake_from_name(app_name, function_name):
        function_names.append((app_name, function_name))
        if function_name == "production_run_gpu":
            return FakeProductionFunction()
        if function_name == "collect_traj_stats":
            return FakeStatsFunction()
        raise AssertionError(f"Unexpected function: {function_name}")

    monkeypatch.setattr(
        shortmd_workflow,
        "clone_prepared_shortmd_run",
        FakeCloneFunction(),
    )
    monkeypatch.setattr(shortmd_workflow.modal.Function, "from_name", fake_from_name)

    node = ShortMDReplicateNode(
        source_run_name="source",
        replicate_run_name="source-r001",
        overwrite_clone=True,
    )
    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="replicate-source-r001",
            attempt_id="attempt-1",
            cache_dir=tmp_path / "cache",
            inputs={
                "prepared": [
                    WorkflowArtifact(
                        artifact_id="source",
                        producing_node_id="prep-source",
                        kind=ArtifactKind.DIRECTORY,
                        storage=VolumePath(
                            volume_name=gromacs_app.OUTPUTS_VOLUME_NAME,
                            path="prepared/source",
                        ),
                        metadata={"stage": "prep", "run_name": "source"},
                    )
                ]
            },
        )
    )

    assert clone_kwargs == {
        "source_storage_path": "prepared/source",
        "source_run_name": "source",
        "replicate_run_name": "source-r001",
        "overwrite": True,
    }
    assert function_names == [
        (gromacs_app.CONF.name, "production_run_gpu"),
        (gromacs_app.CONF.name, "collect_traj_stats"),
    ]
    assert production_kwargs == {
        "run_name": "source-r001",
        "simulation_time_ns": 2,
        "num_threads": 16,
        "use_openmp_threads": False,
    }
    assert stats_kwargs == {
        "traj_prefix": "production_",
        "run_name": "source-r001",
        "save_processed_traj": True,
        "make_figures": True,
    }
    assert result.status == AppRunStatus.SUCCEEDED
    assert result.outputs[0].name == "gromacs_production"
    assert result.outputs[0].kind == ArtifactKind.DIRECTORY
    assert result.outputs[0].storage == VolumePath(
        volume_name=gromacs_app.OUTPUTS_VOLUME_NAME,
        path="production/source-r001",
    )
    assert result.outputs[0].metadata["run_name"] == "source-r001"
    assert result.outputs[0].metadata["source_run_name"] == "source"


def test_shortmd_summary_node_emits_markdown_manifest(tmp_path: Path) -> None:
    node = ShortMDSummaryNode(replicates=2, max_parallel=4)
    context = NodeRunContext(
        run_id="run-1",
        node_id="summary",
        attempt_id="attempt-1",
        cache_dir=tmp_path / "cache",
        inputs={
            "alpha-r001": [
                WorkflowArtifact(
                    artifact_id="alpha-r001",
                    producing_node_id="replicate-alpha-r001",
                    kind=ArtifactKind.DIRECTORY,
                    storage=VolumePath(
                        volume_name=gromacs_app.OUTPUTS_VOLUME_NAME,
                        path="alpha-r001",
                    ),
                    metadata={
                        "source_run_name": "alpha",
                        "run_name": "alpha-r001",
                    },
                )
            ],
            "alpha-r002": [
                WorkflowArtifact(
                    artifact_id="alpha-r002",
                    producing_node_id="replicate-alpha-r002",
                    kind=ArtifactKind.DIRECTORY,
                    storage=VolumePath(
                        volume_name=gromacs_app.OUTPUTS_VOLUME_NAME,
                        path="alpha-r002",
                    ),
                    metadata={
                        "source_run_name": "alpha",
                        "run_name": "alpha-r002",
                    },
                )
            ],
        },
    )

    result = node.run(context)

    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.name == "shortmd_summary"
    assert output.kind == ArtifactKind.REPORT
    assert output.storage.filename == "shortmd-summary.md"
    report = output.storage.data.decode("utf-8")
    assert "# ShortMD Workflow Summary" in report
    assert (
        f"| alpha | alpha-r001 | {gromacs_app.OUTPUTS_VOLUME_NAME} | alpha-r001 |"
        in report
    )
    assert (
        f"| alpha | alpha-r002 | {gromacs_app.OUTPUTS_VOLUME_NAME} | alpha-r002 |"
        in report
    )
