"""Base node contracts for Biomodals workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from biomodals.schema import (
    AppRunResult,
    NodeExecutionPolicy,
    NodePlacement,
    WorkflowArtifact,
)


@dataclass(frozen=True)
class NodeRunContext:
    """Runtime context passed to workflow node implementations."""

    run_id: str
    node_id: str
    attempt_id: str
    cache_dir: Path
    inputs: dict[str, list[WorkflowArtifact]]


class RemoteFunctionCall(Protocol):
    """Minimal Modal FunctionCall boundary used by remote workflow nodes."""

    object_id: str

    def get(self, timeout: float | int | None = None) -> object:
        """Return the remote function result or raise TimeoutError."""


@dataclass(frozen=True)
class RemoteNodeSubmission:
    """A direct remote call submission for one workflow node attempt."""

    function_call: RemoteFunctionCall
    function_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class WorkflowNode(Protocol):
    """Protocol for one semantic workflow DAG vertex."""

    execution_policy: NodeExecutionPolicy
    placement: NodePlacement

    def submit_remote(self, context: NodeRunContext) -> RemoteNodeSubmission:
        """Submit one remote workflow node attempt."""

    def process_remote_result(
        self, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        """Convert a raw remote result into an app-compatible result."""

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute node implementation and return app-compatible outputs."""


class WorkflowNativeNode:
    """Base class for workflow nodes implemented directly in workflow code."""

    execution_policy = NodeExecutionPolicy.RERUN
    placement = NodePlacement.ORCHESTRATOR

    def submit_remote(self, context: NodeRunContext) -> RemoteNodeSubmission:
        """Submit one remote workflow node attempt."""
        raise NotImplementedError(
            "REMOTE workflow nodes must implement submit_remote(context)"
        )

    def process_remote_result(
        self, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        """Convert a raw remote result into an app-compatible result."""
        return AppRunResult.model_validate(result)

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute workflow-native logic."""
        if self.placement == NodePlacement.REMOTE:
            submission = self.submit_remote(context)
            return self.process_remote_result(
                submission.function_call.get(), submission.metadata
            )
        raise NotImplementedError


class AppBackedNode:
    """Base class for workflow nodes implemented by calling app functions."""

    execution_policy = NodeExecutionPolicy.RERUN
    placement = NodePlacement.REMOTE

    def submit_remote(self, context: NodeRunContext) -> RemoteNodeSubmission:
        """Submit the app-backed remote function call."""
        raise NotImplementedError(
            "App-backed workflow nodes must implement submit_remote(context)"
        )

    def process_remote_result(
        self, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        """Convert a raw remote result into an app-compatible result."""
        return AppRunResult.model_validate(result)

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute the app-backed node implementation."""
        submission = self.submit_remote(context)
        return self.process_remote_result(
            submission.function_call.get(), submission.metadata
        )
