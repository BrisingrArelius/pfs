"""
llama_spec.py — Llama-3.1-70B geometry, and the shard plan a PyTorch DCP save
would produce for it under a given tensor/pipeline-parallel configuration.

This module is pure arithmetic: it names every tensor in the model, decides
which rank owns which slice of it, and hands back the exact list of
(rank, fqn, chunk_offsets, chunk_sizes) entries that `torch.distributed.checkpoint`
would record in `.metadata`. Nothing here allocates a model.

Why this exists
---------------
`populate_checkpoint.py` needs to write a *format-exact* DCP checkpoint for a
70B model without ever holding 141 GB of parameters in memory. DCP's on-disk
contract is small and stable (see the module docstring there), so the shard plan
can be computed up front and the bytes streamed one shard at a time.

Sources for the choices made here are marked [CITED] (taken from a paper, a
model card, or upstream library code) or [ASSUMPTION] (an engineering decision
that is defensible but ours). METHODS.md carries the long form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Literal, Optional

# TP rule for a weight stored in HuggingFace's [out_features, in_features] order.
#   "col" — column-parallel in Megatron's sense: split the *output* dim = dim 0.
#   "row" — row-parallel: split the *input* dim = dim 1.
#   "rep" — replicated on every tensor-parallel rank.
TPRule = Literal["col", "row", "rep"]

DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "float8_e4m3fn": 1}


@dataclass(frozen=True)
class ModelSpec:
    """Transformer geometry, in the field names HuggingFace `config.json` uses."""

    name: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    torch_dtype: str = "bfloat16"
    tie_word_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_dim(self) -> int:
        """Output width of k_proj / v_proj under grouped-query attention."""
        return self.num_key_value_heads * self.head_dim

    @property
    def itemsize(self) -> int:
        return DTYPE_BYTES[self.torch_dtype]

    def num_parameters(self) -> int:
        return sum(math.prod(t.shape) for t in self.tensors())

    # -- tensor inventory ---------------------------------------------------

    def tensors(self) -> list["TensorSpec"]:
        """Every tensor in the model, in HuggingFace state-dict order.

        [CITED] Names and shapes follow `transformers`' `LlamaForCausalLM`
        state dict, which is what `from_pretrained` and vLLM's Llama loader
        (`vllm/model_executor/models/llama.py`) key off. Geometry is
        meta-llama/Llama-3.1-70B's config.json.

        [CITED] TP rules follow Megatron-LM's Llama parallelisation, which vLLM
        reproduces: q/k/v and the MLP gate/up projections are column-parallel,
        the attention output and MLP down projections are row-parallel, RMSNorm
        weights are replicated, and the embedding and LM head are
        vocabulary-parallel (a column split of dim 0).
        """
        h, i, v = self.hidden_size, self.intermediate_size, self.vocab_size
        out: list[TensorSpec] = []
        out.append(TensorSpec("model.embed_tokens.weight", (v, h), "col", stage_hint="first"))
        for n in range(self.num_hidden_layers):
            p = f"model.layers.{n}."
            out += [
                TensorSpec(p + "self_attn.q_proj.weight", (h, h), "col", layer=n),
                TensorSpec(p + "self_attn.k_proj.weight", (self.kv_dim, h), "col", layer=n),
                TensorSpec(p + "self_attn.v_proj.weight", (self.kv_dim, h), "col", layer=n),
                TensorSpec(p + "self_attn.o_proj.weight", (h, h), "row", layer=n),
                TensorSpec(p + "mlp.gate_proj.weight", (i, h), "col", layer=n),
                TensorSpec(p + "mlp.up_proj.weight", (i, h), "col", layer=n),
                TensorSpec(p + "mlp.down_proj.weight", (h, i), "row", layer=n),
                TensorSpec(p + "input_layernorm.weight", (h,), "rep", layer=n),
                TensorSpec(p + "post_attention_layernorm.weight", (h,), "rep", layer=n),
            ]
        out.append(TensorSpec("model.norm.weight", (h,), "rep", stage_hint="last"))
        if not self.tie_word_embeddings:
            out.append(TensorSpec("lm_head.weight", (v, h), "col", stage_hint="last"))
        return out


@dataclass(frozen=True)
class TensorSpec:
    fqn: str
    shape: tuple[int, ...]
    tp_rule: TPRule
    layer: Optional[int] = None
    stage_hint: Optional[str] = None  # "first" | "last" for non-layer tensors

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    def shard_axis(self) -> Optional[int]:
        return {"col": 0, "row": 1, "rep": None}[self.tp_rule]


@dataclass(frozen=True)
class ParallelConfig:
    """A tensor/pipeline/data-parallel layout. world_size = tp * pp * dp."""

    tp: int = 1
    pp: int = 1
    dp: int = 1

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.dp

    def __str__(self) -> str:
        return f"TP{self.tp}/PP{self.pp}/DP{self.dp}"

    @staticmethod
    def parse(text: str) -> "ParallelConfig":
        """Accept 'TP8/PP4/DP1' in any order, missing parts default to 1."""
        vals = {"tp": 1, "pp": 1, "dp": 1}
        for part in text.replace(",", "/").split("/"):
            part = part.strip().lower()
            if not part:
                continue
            for key in vals:
                if part.startswith(key):
                    vals[key] = int(part[len(key):])
                    break
            else:
                raise SystemExit(f"cannot parse parallelism component {part!r}")
        return ParallelConfig(**vals)

    def global_rank(self, pp_stage: int, tp_rank: int, dp_rank: int = 0) -> int:
        """Rank order TP-fastest, then PP, then DP — Megatron-LM's default."""
        return (dp_rank * self.pp + pp_stage) * self.tp + tp_rank


# ---------------------------------------------------------------------------
# shard planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardEntry:
    """One chunk of one tensor, owned by one global rank.

    Mirrors a DCP `WriteItem` plus the `ChunkStorageMetadata` it produces.
    """

    fqn: str
    rank: int
    offsets: tuple[int, ...]
    sizes: tuple[int, ...]
    chunk_index: int  # position within the tensor's chunk list in .metadata
    global_shape: tuple[int, ...]
    tp_rule: TPRule
    dtype: str = "bfloat16"
    kind: str = "model"          # "model" | "optim"
    seq: int = 0                 # position in the flattened global state dict

    @property
    def numel(self) -> int:
        return math.prod(self.sizes)

    def nbytes(self) -> int:
        return self.numel * DTYPE_BYTES[self.dtype]


def stage_of(spec: TensorSpec, model: ModelSpec, pp: int) -> int:
    """Which pipeline stage owns a tensor.

    [ASSUMPTION] Layers are split into `pp` contiguous, equal blocks, and the
    embedding rides on the first stage while the final norm and LM head ride on
    the last. This is Megatron-LM's default (`--num-layers-per-virtual-pipeline-stage`
    unset); real 70B runs often rebalance the end stages to offset the LM head's
    cost, which would move a handful of tensors between files but change no
    total.
    """
    if pp == 1:
        return 0
    if spec.layer is not None:
        per = model.num_hidden_layers // pp
        return min(spec.layer // per, pp - 1)
    return 0 if spec.stage_hint == "first" else pp - 1


def _tp_chunks(spec: TensorSpec, tp: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """(offsets, sizes) for each tensor-parallel rank, in rank order.

    Replicated tensors return a single whole-tensor chunk: DCP stores one copy
    and records one chunk, so the replication is invisible in `.metadata`.
    """
    axis = spec.shard_axis()
    if axis is None:
        return [(tuple(0 for _ in spec.shape), spec.shape)]
    extent = spec.shape[axis]
    if extent % tp:
        raise SystemExit(
            f"{spec.fqn} has {extent} along dim {axis}, not divisible by TP={tp}. "
            "Pick a TP that divides num_key_value_heads and the FFN width."
        )
    step = extent // tp
    chunks = []
    for r in range(tp):
        offsets = tuple(r * step if d == axis else 0 for d in range(len(spec.shape)))
        sizes = tuple(step if d == axis else spec.shape[d] for d in range(len(spec.shape)))
        chunks.append((offsets, sizes))
    return chunks


def plan_shards(model: ModelSpec, cfg: ParallelConfig) -> list[ShardEntry]:
    """Every shard a DCP save of `model` under `cfg` writes, in write order.

    Data-parallel replicas hold identical weights, so DCP's deduplication keeps
    exactly one copy; only the dp_rank=0 group writes. Tensor-parallel
    *replicated* tensors (the RMSNorm weights) are likewise duplicated across
    the TP group, and DCP assigns each to whichever rank has the least planned
    bytes so far.

    [CITED] That tie-break is `torch.distributed.checkpoint._dedup_save_plans.
    dedup_save_plans`, which picks `min(plan_indices, key=plan_to_size)` while
    walking write items in plan order. We reproduce it exactly so our
    `.metadata` matches a real save byte for byte.
    """
    specs = model.tensors()
    by_rank: dict[int, list[TensorSpec]] = {}
    chunk_lists: dict[str, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {}

    for spec in specs:
        chunk_lists[spec.fqn] = _tp_chunks(spec, cfg.tp)
        stage = stage_of(spec, model, cfg.pp)
        for tp_rank in range(cfg.tp):
            by_rank.setdefault(cfg.global_rank(stage, tp_rank), []).append(spec)

    # Walk items the way dedup_save_plans sees them: rank order, and within a
    # rank the local state-dict order. First appearance fixes the ordering.
    load = {r: 0 for r in by_rank}
    owner: dict[tuple[str, int], int] = {}      # (fqn, chunk_index) -> rank
    seen: set[tuple[str, int]] = set()
    order: list[tuple[str, int]] = []

    for rank in sorted(by_rank):
        tp_rank = rank % cfg.tp
        for spec in by_rank[rank]:
            chunk_index = 0 if spec.tp_rule == "rep" else tp_rank
            key = (spec.fqn, chunk_index)
            if key in seen:
                continue
            seen.add(key)
            order.append(key)
            if spec.tp_rule == "rep":
                stage = stage_of(spec, model, cfg.pp)
                candidates = [cfg.global_rank(stage, t) for t in range(cfg.tp)]
            else:
                candidates = [rank]
            pick = min(candidates, key=lambda r: (load[r], r))
            owner[key] = pick
            load[pick] += math.prod(chunk_lists[spec.fqn][chunk_index][1]) * model.itemsize

    spec_by_fqn = {s.fqn: s for s in specs}
    seq_of = {s.fqn: i for i, s in enumerate(specs)}
    entries = []
    for fqn, chunk_index in order:
        spec = spec_by_fqn[fqn]
        offsets, sizes = chunk_lists[fqn][chunk_index]
        entries.append(
            ShardEntry(
                fqn=fqn,
                rank=owner[(fqn, chunk_index)],
                offsets=offsets,
                sizes=sizes,
                chunk_index=chunk_index,
                global_shape=spec.shape,
                tp_rule=spec.tp_rule,
                dtype=model.torch_dtype,
                kind="model",
                seq=seq_of[fqn],
            )
        )
    return entries


# ---------------------------------------------------------------------------
# optimizer state
# ---------------------------------------------------------------------------

OPTIMIZER_KINDS = {
    # (suffix, dtype) per parameter. Bytes per parameter in the comment.
    "none": (),
    "adam": (("exp_avg", "float32"), ("exp_avg_sq", "float32")),                   # 8 B
    "adam_master": (("exp_avg", "float32"), ("exp_avg_sq", "float32"),
                    ("fp32_master", "float32")),                                   # 12 B
}


def add_optimizer_state(entries: list[ShardEntry], kind: str) -> list[ShardEntry]:
    """Append Adam optimizer shards mirroring each model shard.

    [ASSUMPTION] Each rank holds optimizer state for exactly the parameter
    shards it owns, in fp32, with the same sharding as the parameter. That is
    true for Megatron-LM's distributed optimizer without ZeRO-1 sharding; under
    ZeRO-1 the moments are re-sharded flat across the data-parallel group
    instead, which changes their *shapes* but not their total bytes or the fact
    that an eval job never asks for them. `--optimizer none` omits them.

    [CITED] Naming follows the flattened key a real save produces from
    `dcp.save({"model": ..., "optim": ...})`, where `torch.distributed.
    checkpoint._nested_dict.flatten_state_dict` joins nested keys with ".".
    Ordering also follows that flattening: every model tensor first, then the
    optimizer block.
    """
    moments = OPTIMIZER_KINDS[kind]
    if not moments:
        return list(entries)
    base = max(e.seq for e in entries) + 1
    out = list(entries)
    for e in entries:
        for k, (suffix, dt) in enumerate(moments):
            out.append(
                ShardEntry(
                    fqn=f"optim.state.{e.fqn}.{suffix}",
                    rank=e.rank,
                    offsets=e.offsets,
                    sizes=e.sizes,
                    chunk_index=e.chunk_index,
                    global_shape=e.global_shape,
                    tp_rule=e.tp_rule,
                    dtype=dt,
                    kind="optim",
                    seq=base + e.seq * len(moments) + k,
                )
            )
    return out


# ---------------------------------------------------------------------------
# file assignment
# ---------------------------------------------------------------------------


def assign_files(
    entries: list[ShardEntry], layout: str, thread_count: int = 1
) -> list[tuple[str, list[ShardEntry]]]:
    """Group shards into `.distcp` files exactly as DCP's writer would.

    [CITED] `FileSystemWriter.write_data` names files `__{rank}_{n}.distcp` and
    chooses their contents three ways, which produce three very different
    directories:

      per-rank      single_file_per_rank=True, thread_count=1 (the default).
                    One file per rank holding everything that rank owns, in
                    local state-dict order.

      per-rank-threads
                    single_file_per_rank=True, thread_count=N. Items are split
                    across N files per rank by `_split_by_size_and_type`, which
                    despite its name buckets *tensors by size only*: sort
                    descending, then greedy least-full bucket. Model and
                    optimizer shards are therefore interleaved arbitrarily.

      per-shard     single_file_per_rank=False. One file per write item -- the
                    file-per-shard layout Gossman et al. (SCA/HPC Asia 2026)
                    measure, where a 3B model on 4 GPUs becomes 132 files.
    """
    by_rank: dict[int, list[ShardEntry]] = {}
    for e in entries:
        by_rank.setdefault(e.rank, []).append(e)
    for r in by_rank:
        by_rank[r].sort(key=lambda e: e.seq)

    files: list[tuple[str, list[ShardEntry]]] = []
    for rank in sorted(by_rank):
        items = by_rank[rank]
        if layout == "per-shard":
            for i, e in enumerate(items):
                files.append((f"__{rank}_{i}.distcp", [e]))
        elif layout == "per-rank" or thread_count == 1:
            files.append((f"__{rank}_0.distcp", items))
        elif layout == "per-rank-threads":
            buckets: list[list[ShardEntry]] = [[] for _ in range(thread_count)]
            sizes = [0] * thread_count
            for e in sorted(items, key=lambda e: e.nbytes(), reverse=True):
                i = min(range(thread_count), key=lambda b: (sizes[b], b))
                buckets[i].append(e)
                sizes[i] += e.nbytes()
            for i, b in enumerate(buckets):
                files.append((f"__{rank}_{i}.distcp", b))
        else:
            raise SystemExit(f"unknown layout {layout!r}")
    return files


PRESETS = {
    # meta-llama/Llama-3.1-70B config.json
    "llama31_70b": ModelSpec("llama31_70b", 80, 8192, 28672, 128256, 64, 8),
    # meta-llama/Llama-3.1-8B — same family, useful for a cheaper full-scale run
    "llama31_8b": ModelSpec("llama31_8b", 32, 4096, 14336, 128256, 32, 8),
    # Llama-shaped but tiny: same tensor inventory, same TP rules, ~43M params.
    # For byte-real correctness runs where the real geometry will not fit.
    "llama_tiny": ModelSpec("llama_tiny", 4, 512, 1376, 32000, 8, 8),
}


def preset(name: str, num_hidden_layers: Optional[int] = None) -> ModelSpec:
    if name not in PRESETS:
        raise SystemExit(f"unknown model {name!r}; have {', '.join(PRESETS)}")
    m = PRESETS[name]
    if num_hidden_layers is not None:
        m = ModelSpec(**{**m.__dict__, "num_hidden_layers": num_hidden_layers})
    return m
