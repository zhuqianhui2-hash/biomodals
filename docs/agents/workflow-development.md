# Biomodals Workflow Development

Detailed workflow-development instructions for `src/biomodals/workflow/` and
shared workflow schemas under `src/biomodals/schema/` live in the repo-local
skill:

- `.agents/skills/biomodals-workflow-development/SKILL.md`
- `.agents/skills/biomodals-workflow-development/references/workflow-development.md`

## How Agents Should Use It

- Invoke or read the `biomodals-workflow-development` skill before creating,
  editing, or reviewing Biomodals workflow code.
- Treat `src/biomodals/workflow/shortmd_workflow.py` as the primary end-to-end
  reference workflow.
- Treat `src/biomodals/workflow/rfd_ligandmpnn_workflow.py` as the reference
  for workflows that read selected files from one app's volume-backed output and
  fan them out into another app's workflow-compatible function.
- Ignore `src/biomodals/workflow/ppiflow_workflow.py` as a reference pattern
  for now because it is expected to be refactored.
- When adding workflow-compatible app functions under `src/biomodals/app/`, also
  follow `docs/agents/app-development.md` and use RFdiffusion or LigandMPNN as
  the app-side reference pattern.
- Keep shared schemas pure Pydantic. For inline bytes, use Pydantic
  `ser_json_bytes` / `val_json_bytes` config for JSON byte handling and enforce
  text-vs-zstd archive policy in workflow runtime materialization code.
- When declaring dependency apps, mirror `AppConfig.depends_on_apps` into a
  Modal-valid `CONF.tags["depends_on"]` value such as
  `",".join(DEPENDENCY_APPS)`, then call `include_dependency_apps(...)`.

## Maintenance

- Update the workflow skill when workflow standards change.
- Keep this document as a pointer and coordination note, not a duplicate copy of
  the skill.
