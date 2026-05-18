# Biomodals Workflow Development

Use this guide when creating or changing files under
`src/biomodals/workflow/` or shared workflow contracts under
`src/biomodals/schema/`.

## Vocabulary

- **App**: a deployed Modal app that owns tool runtime and app functions.
- **App Function**: a callable remote Modal function exposed by an app.
- **Local Entrypoint**: a CLI-only `@app.local_entrypoint`.
- **Workflow-Compatible App Function**: a remote app function returning
  `AppRunResult`.
- **Workflow Node**: one semantic DAG vertex.
- **App-Backed Node**: a workflow node that calls app functions.
- **Workflow-Native Node**: a workflow node implemented in workflow code.
- **Workflow Runtime**: validates and schedules workflow nodes.
- **Workflow Orchestrator**: the remote Modal function hosting the runtime.
- **Workflow Artifact**: durable data passed between workflow nodes.
- **Artifact Selector**: a named reference to upstream artifacts.
- **Worker Pool**: fixed-size remote workers for one node's fan-out tasks.

Avoid the terms `app node`, `runner node`, `engine`, and `workflow
entrypoint`; they are ambiguous in this codebase.

## Schema Boundaries

Shared contracts live in `biomodals.schema`.

Schema modules must not import `modal`, `biomodals.app`, or
`biomodals.workflow`. They should contain Pydantic models and primitive fields
only. App-specific config models stay with the app until they become stable
cross-module contracts.

Workflow-compatible app functions return `AppRunResult`. The workflow runtime
materializes each `AppOutput` into one or more `WorkflowArtifact` manifests.
Inline byte outputs must be written into the workflow run volume before they
cross a node boundary.

The first workflow runtime is Python-first. Pass a `Workflow` object across the
orchestrator boundary; serialized workflow dictionaries are intentionally
deferred until the node and app-function contracts stabilize.

## Node Execution Policy

Every workflow node checks durable run state before execution and skips work
when completed artifact manifests already exist.

Incomplete nodes use one of two policies:

- `RERUN`: discard incomplete attempt state and recompute.
- `RESUME`: use a durable node cache to resume or skip completed subwork.

Long-running nodes must be idempotent against deterministic run, node, input,
and attempt identifiers. Store resumable state in volumes, not container-local
scratch paths.

## Node Placement

Use `ORCHESTRATOR` placement for lightweight workflow-native logic such as
filtering, ranking, reporting, and small manifest transforms.

Use `REMOTE` placement for long-running work, app-backed work, and work that
benefits from failure isolation.

## Ledger Layout

The first durable run layout is:

```text
<workflow-volume>/<workflow-name>/<run-id>/
  run.json
  inputs/
  nodes/
    <node-id>/
      status.json
      attempts/
        <attempt-id>/
          started.json
          app_result.json
          logs/
          raw_outputs/
          materialized_outputs/
      cache/
  artifacts/
    <artifact-id>.json
  final/
```

Write JSON through ledger helpers so state files are replaced atomically where
practical. After volume writes inside Modal containers, call `commit()`. Before
reading data written by another container, call `reload()`.

## Modal Preemption

All Modal functions are subject to preemption. Treat remote functions as
restartable with the same inputs.

Remote workflow code should:

- split long work into smaller retryable tasks;
- record attempt status before and after work;
- write cache checkpoints for `RESUME` nodes;
- use deterministic output paths from run and node identifiers;
- leave enough artifacts and logs to reconcile after restart.

## Fan-Out And Worker Pools

The first workflow runtime supports static DAGs with dynamic task fan-out.
The DAG shape is fixed, but a node may derive a runtime task list from upstream
artifacts.

Use barriered fan-out first: a node starts only after all declared upstream
dependencies are complete. Streaming between nodes is deferred.

A fan-out node may spawn a fixed-size worker pool. Workers process that node's
task queue until empty, then the node writes a single completion status.

Independent ready nodes may run in parallel when all dependencies for each node
are satisfied.

## App Interfaces

Local entrypoints stay CLI-only. They parse local paths, submit remote work,
download or report outputs, print user messages, and return `None`.

Workflow reuse happens through workflow-compatible remote app functions. These
functions may reuse behavior from local entrypoints or existing remote
functions, but they return `AppRunResult` and avoid local filesystem UX.

App-backed workflow nodes either define `app_name` and `function_name` so the
runtime can lazily import `modal.Function.from_name(...)`, or override
`load_app_function()`. They should override `build_app_function_kwargs()` to
translate `NodeRunContext.inputs` into the app function's primitive or Pydantic
arguments.

When adding a workflow-compatible app function, keep existing local entrypoint
behavior unchanged and add a focused pytest contract test that does not call
Modal live APIs.

## Testing

Keep tests under top-level `tests/`.

Use pytest for non-Modal tests. Tests must not call `.remote()`, `.spawn()`,
`modal.Function.from_name(...)`, real `modal.Queue`, real `modal.Volume`, or
deployed Modal apps. Mock Modal boundaries with fake objects and deterministic
`AppRunResult` or `WorkflowArtifact` payloads.
