"""Pure workflow orchestrator boundary helpers."""

from __future__ import annotations

from pathlib import Path

from biomodals.schema import AppRunResult
from biomodals.workflow.core.builder import Workflow
from biomodals.workflow.core.runtime import WorkflowRuntime, WorkflowVolume


def run_workflow_definition(
    *,
    workflow_name: str,
    run_id: str,
    workflow_definition: Workflow | dict[str, object],
    volume_root: Path,
    workflow_volume_name: str,
    force: bool = False,
    workflow_volume: WorkflowVolume | None = None,
) -> AppRunResult:
    """Run one workflow definition through the workflow runtime."""
    runtime = WorkflowRuntime.from_definition(
        workflow_name=workflow_name,
        workflow_definition=workflow_definition,
        volume_root=volume_root,
        workflow_volume_name=workflow_volume_name,
        workflow_volume=workflow_volume,
    )
    return runtime.run(run_id=run_id, force=force)
