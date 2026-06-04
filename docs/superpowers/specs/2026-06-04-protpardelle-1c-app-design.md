# Protpardelle-1c Modal App Design

## Purpose

Add `src/biomodals/app/design/protpardelle_1c_app.py`, a persistent,
parallel, and resumable Modal wrapper for backbone-only Protpardelle-1c
sampling.

The first release supports pharmaceutical protein-design use cases while
remaining close to the upstream sampling interface. It generates backbones
only and deliberately excludes ProteinMPNN, ESMFold self-consistency
evaluation, Foldseek clustering, all-atom checkpoints, unconditional
generation, structure prediction, and FastRelax.

## Supported Designs

The app automatically identifies one of four supported design types from the
sampling YAML:

| Design type | Identification | Allowed checkpoint |
| --- | --- | --- |
| Backbone partial diffusion | `partial_diffusion.enabled: true` and `motif_contigs: [partial_diffusion]` | `cc58_epoch416` or `cc95_epoch3490` |
| Motif scaffolding | Supported motif-conditioned YAML not classified below | `cc58_epoch416` or `cc95_epoch3490` |
| Binder generation | Unique model is `cc83_epoch2616` and `hotspots` is non-null | `cc83_epoch2616` |
| Multichain/homodimer generation | Unique model is `cc78_epoch1431` | `cc78_epoch1431` |

Checkpoint-based classification takes precedence where necessary. In
particular, a `cc78_epoch1431` YAML is multichain/homodimer generation even
when it contains hotspots.

The app rejects unsupported or ambiguous combinations rather than guessing.

## YAML Constraints

Each YAML describes one model and one design target:

- `search_space.models` must contain exactly one supported backbone checkpoint.
- `motifs`, `motif_contigs`, `total_lengths`, `hotspots`, and `ssadj` must each
  contain exactly one corresponding entry.
- `ssadj` must be `[null]`.
- The YAML may contain multiple values for other `search_space` parameters.
  Protpardelle-1c retains its upstream Cartesian-product behavior for these
  parameters.
- The app always invokes upstream sampling with `num_mpnn_seqs=0`; this value
  is not user-configurable.

The app rejects YAMLs for unconditional generation, unsupported checkpoints,
all-atom sampling configurations, multiple models, or multiple design targets.

## Input Handling

The sampling YAML is a positional CLI argument, matching the upstream README.
The user passes `--motif-dir` as the local root directory for motif files
referenced by the YAML.

The app parses the single `motifs` entry and uploads only the referenced
PDB/CIF file, preserving its relative path beneath the input root. A motif
entry with no extension follows the upstream rule and resolves to `.pdb`.
Unreferenced files in `--motif-dir` are not uploaded.

The app rejects motif references that:

- do not exist;
- are absolute paths;
- escape `--motif-dir`;
- use unsafe relative paths; or
- do not resolve according to the upstream PDB/CIF naming behavior.

There is no Caliby-specific input mode and no `--motif-pdb` option. A future
`caliby_app.py` will consume the backbone outputs from this app.

## CLI

The minimal sampling command stays close to the upstream interface:

```bash
uv run biomodals app run protpardelle-1c \
  config.yaml \
  --motif-dir ./inputs \
  --num-samples 8 \
  --run-name experiment-001
```

`--run-name` is the only required Modal-specific sampling argument. It provides
a stable persistent-output and resume identity.

Optional sampling arguments include:

- `--batch-size`, defaulting to the upstream value of `32`;
- `--seed`, defaulting to `None`;
- `--samples-per-shard`, unset by default;
- `--max-num-gpus`, defaulting to `1`;
- `--resume`, defaulting to false; and
- `--out-dir`, for downloading a packaged result.

Model download is an explicit, separate operation:

```bash
uv run biomodals app run protpardelle-1c --download-models
```

`--force-redownload` may be used together with `--download-models`.

## Model Download and Validation

The explicit model-download function downloads the complete upstream
Protpardelle-1c archive, validates the complete archive before extraction, and
keeps all extracted model files. It does not delete checkpoints that are
outside the first-release inference allowlist.

Inference permits only these backbone checkpoints and corresponding
configurations:

- `cc58_epoch416`;
- `cc95_epoch3490`;
- `cc83_epoch2616`; and
- `cc78_epoch1431`.

The app does not download ProteinMPNN, ESMFold, LigandMPNN, or Foldseek
resources.

## Architecture

The app contains four main remote responsibilities and one local entrypoint:

### Model download

`download_protpardelle_1c_models` downloads, validates, and extracts the model
archive into the persistent model volume.

### Run preparation

`prepare_protpardelle_1c_run` validates the YAML, referenced motif, selected
checkpoint, CLI values, and existing run state. It determines the design type,
counts search-space combinations, creates the shard plan, and writes the
immutable run manifest and staged inputs.

### Shard execution

`run_protpardelle_1c_shard` occupies one GPU, loads the YAML's unique model,
runs one independent sample shard, and atomically writes its output and
completion marker.

### Finalization

`finalize_protpardelle_1c_run` verifies every shard, creates the top-level
backbone manifest and completion marker, and optionally packages the complete
run as `.tar.zst`.

### Local orchestration

`submit_protpardelle_1c_task` is the local CLI entrypoint. It prepares the run,
submits pending shards with bounded concurrency, finalizes the result, and
optionally downloads the archive.

## Sharding and GPU Scheduling

By default, `--samples-per-shard` is unset. The app creates one shard and
preserves upstream sampling behavior.

When sharding is enabled, only the `num_samples` dimension is divided. The
final shard is not padded:

```text
num_samples=20, samples_per_shard=8
=> shard-000=8, shard-001=8, shard-002=4
```

`num_samples` retains its upstream meaning: it is the number generated for
each Cartesian-product search-space combination. For four parameter
combinations, the example above generates:

```text
shard-000: 4 x 8 = 32 backbones
shard-001: 4 x 8 = 32 backbones
shard-002: 4 x 4 = 16 backbones
total: 80 backbones
```

Each shard uses one independent Modal GPU function and reloads the model. The
maximum simultaneous GPU count is:

```text
min(max_num_gpus, pending_shard_count)
```

If fewer GPUs are allowed than pending shards, remaining shards wait and run
as capacity becomes available. Without sharding, there is only one task and
only one GPU is used regardless of `--max-num-gpus`.

## Seed Semantics

`--seed` is optional and defaults to `None`, including when sharding is
enabled.

- With `seed=None`, each shard uses upstream random behavior. Resume preserves
  completed shards but a rerun of a failed shard may produce different
  backbones.
- With an explicit base seed, shard index `i` uses `base_seed + i`, allowing
  deterministic shard reruns.

Each shard manifest records its effective seed, including `null`.

## Persistent Output

All run state and outputs live in the persistent output volume:

```text
Protpardelle1C-outputs/<run-name>/
├── run_manifest.json
├── inputs/
├── shards/
│   ├── shard-000/
│   │   ├── outputs/
│   │   └── completed.json
│   └── shard-001/
│       ├── outputs/
│       └── completed.json
└── completed.json
```

Shard outputs remain in their shard directories to avoid duplicate storage and
file-renaming collisions. The top-level `completed.json` is written only after
all shards pass validation and lists every generated backbone by relative
path. Downstream apps, including a future Caliby app, consume this manifest
instead of relying on directory-name conventions.

## Resume and Failure Handling

An existing `run-name` is rejected unless the user explicitly passes
`--resume`.

The immutable run manifest records digests of the original YAML, uploaded
motif, and critical execution arguments. Resume rejects a run when these
values differ from the existing manifest.

On resume, the app:

1. returns the existing result immediately if the top-level run is complete
   and valid;
2. validates each shard completion marker and its listed backbone files;
3. submits only missing or incomplete shards; and
4. finalizes the run after every shard is valid.

A shard rerun writes into a new temporary directory and replaces the shard
output only after successful completion. Existing failed output and logs are
retained for diagnosis until replacement succeeds. Completion markers are
written last.

## Validation and Errors

Preparation fails before GPU submission when:

- the YAML violates the single-model or single-target constraints;
- the checkpoint or inferred design type is unsupported or inconsistent;
- the referenced motif is absent or unsafe;
- required model files are missing or invalid;
- `num_samples`, `batch_size`, `samples_per_shard`, or `max_num_gpus` is
  invalid; or
- a resume request does not match the original immutable run configuration.

The local entrypoint reports the inferred design type, selected checkpoint,
number of search-space combinations, sample count per combination, shard
plan, expected total backbone count, and maximum GPU concurrency before
submitting work.

## Verification

Unit tests cover:

- YAML validation and rejection cases;
- design-type identification and checkpoint compatibility;
- safe motif-path resolution;
- search-space combination counting;
- shard partitioning, including an unpadded final shard;
- seed derivation;
- expected total backbone counting; and
- resume planning from fabricated completion markers and missing outputs.

Repository verification includes:

- `prek run --files` for changed files;
- `uv run biomodals app list`;
- `uv run biomodals app help protpardelle-1c`; and
- the relevant test suite.

Modal smoke tests cover a default unsharded run, a small two-shard concurrent
run, and explicit resume after making one shard incomplete.
