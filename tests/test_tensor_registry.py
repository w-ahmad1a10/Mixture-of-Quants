#!/usr/bin/env python3
"""
test_tensor_registry.py
Unit tests for the unified TensorRegistry and grouping logic.
"""
import json
import os
import tempfile
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moq.moq_core import (
    TensorRegistry,
    _group_key_for,
    build_group_param_counts,
    build_group_tensors,
    _parse_tensor_map,
    _get_ce_key,
    _get_layer_idx,
    MoQFinalQuantizer,
)


# ─── fixtures ────────────────────────────────────────────────────────────────

TENSOR_MAP_LINES = """\
model.embed_tokens = token_embd.weight
model.layers.0.conv.in_proj = blk.0.shortconv.in_proj.weight
model.layers.0.conv.out_proj = blk.0.shortconv.out_proj.weight
model.layers.0.feed_forward.w1 = blk.0.ffn_gate.weight
model.layers.0.feed_forward.w2 = blk.0.ffn_down.weight
model.layers.0.feed_forward.w3 = blk.0.ffn_up.weight
model.layers.0.self_attn.q_proj = blk.0.attn_q.weight
model.layers.0.self_attn.k_proj = blk.0.attn_k.weight
model.layers.0.self_attn.v_proj = blk.0.attn_v.weight
model.layers.0.self_attn.out_proj = blk.0.attn_output.weight
model.layers.1.conv.in_proj = blk.1.shortconv.in_proj.weight
model.layers.1.conv.out_proj = blk.1.shortconv.out_proj.weight
model.layers.1.feed_forward.w1 = blk.1.ffn_gate.weight
model.layers.1.feed_forward.w2 = blk.1.ffn_down.weight
model.layers.1.feed_forward.w3 = blk.1.ffn_up.weight
model.layers.2.feed_forward.w1 = blk.2.ffn_gate.weight
model.layers.2.feed_forward.w2 = blk.2.ffn_down.weight
model.layers.2.feed_forward.w3 = blk.2.ffn_up.weight
model.layers.2.self_attn.q_proj = blk.2.attn_q.weight
model.layers.2.self_attn.k_proj = blk.2.attn_k.weight
model.layers.2.self_attn.v_proj = blk.2.attn_v.weight
model.layers.2.self_attn.out_proj = blk.2.attn_output.weight
"""

PARAM_COUNTS = {
    "token_embd": 67_108_864,
    "shortconv.in_proj": 25_165_824,
    "shortconv.out_proj": 8_388_608,
    "ffn_gate": 36_700_160,
    "ffn_down": 36_700_160,
    "ffn_up": 36_700_160,
    "attn_q": 6_291_456,
    "attn_k": 3_145_728,
    "attn_v": 3_145_728,
    "attn_output": 6_291_456,
}

N_BLOCKS = 16  # pretend 16 layers total


@pytest.fixture
def tensor_map(tmp_path):
    p = tmp_path / "tensor_map.txt"
    p.write_text(TENSOR_MAP_LINES)
    return str(p)


@pytest.fixture
def registry(tensor_map):
    return TensorRegistry(_parse_tensor_map(tensor_map), PARAM_COUNTS, N_BLOCKS)


# ─── helper tests ────────────────────────────────────────────────────────────

class TestHelpers:
    def test_parse_tensor_map(self, tensor_map):
        m = _parse_tensor_map(tensor_map)
        assert "model.embed_tokens" in m
        assert m["model.embed_tokens"] == "token_embd.weight"
        assert m["model.layers.0.conv.in_proj"] == "blk.0.shortconv.in_proj.weight"

    def test_get_ce_key_layer(self):
        assert _get_ce_key("model.layers.0.conv.in_proj") == "conv_in_proj"
        assert _get_ce_key("model.layers.10.self_attn.q_proj") == "self_attn_q_proj"

    def test_get_ce_key_non_layer(self):
        assert _get_ce_key("model.embed_tokens") == "embed_tokens"

    def test_get_layer_idx(self):
        assert _get_layer_idx("blk.0.shortconv.in_proj.weight") == 0
        assert _get_layer_idx("blk.10.attn_q.weight") == 10
        assert _get_layer_idx("token_embd.weight") is None


# ─── registry tests ──────────────────────────────────────────────────────────

class TestTensorRegistry:
    def test_entry_count(self, registry):
        assert len(registry.registry) == 22

    def test_embedding_params(self, registry):
        entry = registry.get_entry("model.embed_tokens")
        assert entry is not None
        assert entry["gguf_name"] == "token_embd.weight"
        assert entry["ce_key"] == "embed_tokens"
        assert entry["layer_idx"] is None
        assert entry["param_count"] == 67_108_864

    def test_layer_tensor_params_divided_by_n_blocks(self, registry):
        entry = registry.get_entry("model.layers.0.conv.in_proj")
        assert entry is not None
        assert entry["gguf_name"] == "blk.0.shortconv.in_proj.weight"
        assert entry["ce_key"] == "conv_in_proj"
        assert entry["layer_idx"] == 0
        # shortconv.in_proj total = 25_165_824, / 16 layers = 1_572_864
        assert entry["param_count"] == 25_165_824 // N_BLOCKS

    def test_by_gguf_name_lookup(self, registry):
        entry = registry.get_by_gguf_name("blk.0.ffn_gate.weight")
        assert entry is not None
        assert entry["pt_name"] == "model.layers.0.feed_forward.w1"

    def test_by_ce_key(self, registry):
        entries = registry.entries_by_ce_key("conv_in_proj")
        assert len(entries) == 2  # layers 0 and 1

    def test_by_layer_range(self, registry):
        entries = registry.entries_by_layer_range(0, 1)
        # layer 0: 9 tensors, layer 1: 5 tensors = 14 total
        assert len(entries) == 14


# ─── grouping tests ──────────────────────────────────────────────────────────

class TestGrouping:
    def test_by_tensor_type_keys(self, registry):
        gk = _group_key_for("model.layers.0.conv.in_proj", registry, "by-tensor-type")
        assert gk == "conv_in_proj"
        gk = _group_key_for("model.layers.0.self_attn.q_proj", registry, "by-tensor-type")
        assert gk == "self_attn_q_proj"
        gk = _group_key_for("model.embed_tokens", registry, "by-tensor-type")
        assert gk == "embed_tokens"

    def test_by_layer_keys(self, registry):
        # layer 0 -> block 0-3
        gk = _group_key_for("model.layers.0.conv.in_proj", registry, "by-layer")
        assert gk == "layer_0-3"
        # layer 1 -> block 0-3
        gk = _group_key_for("model.layers.1.feed_forward.w1", registry, "by-layer")
        assert gk == "layer_0-3"
        # layer 10 -> block 8-11
        # (need a layer 10 entry; we only have up to layer 2 in fixture)
        # test embed stays as ce_key
        gk = _group_key_for("model.embed_tokens", registry, "by-layer")
        assert gk == "embed_tokens"

    def test_by_layer_tensor_keys(self, registry):
        gk = _group_key_for("model.layers.0.conv.in_proj", registry, "by-layer-tensor")
        assert gk == "conv_in_proj_layer_0-3"
        gk = _group_key_for("model.layers.1.feed_forward.w1", registry, "by-layer-tensor")
        assert gk == "feed_forward_w1_layer_0-3"

    def test_individual_keys(self, registry):
        gk = _group_key_for("model.layers.0.conv.in_proj", registry, "individual")
        assert gk == "model.layers.0.conv.in_proj"
        gk = _group_key_for("model.embed_tokens", registry, "individual")
        assert gk == "model.embed_tokens"

    def test_build_group_param_counts_by_tensor_type(self, registry):
        pc = build_group_param_counts(registry, "by-tensor-type")
        # conv_in_proj: 2 layers * (25_165_824 / 16) = 3_145_728
        assert pc["conv_in_proj"] == 2 * (25_165_824 // N_BLOCKS)
        # embed_tokens: 1 * 67_108_864
        assert pc["embed_tokens"] == 67_108_864

    def test_build_group_param_counts_by_layer(self, registry):
        pc = build_group_param_counts(registry, "by-layer")
        # layer_0-3 has layers 0,1,2,3 — but our fixture only has 0,1,2
        # layer 0: 10 tensors, layer 1: 5 tensors, layer 2: 7 tensors
        # All in block 0-3
        layer_0_3_count = sum(
            registry.get_entry(pt)["param_count"]
            for pt in registry.registry
            if registry.get_entry(pt)["layer_idx"] in (0, 1, 2)
        )
        assert pc["layer_0-3"] == layer_0_3_count
        # embed_tokens stays separate
        assert pc["embed_tokens"] == 67_108_864

    def test_build_group_tensors_by_tensor_type(self, registry):
        gt = build_group_tensors(registry, "by-tensor-type")
        assert "blk.0.shortconv.in_proj.weight" in gt["conv_in_proj"]
        assert "blk.1.shortconv.in_proj.weight" in gt["conv_in_proj"]
        assert len(gt["conv_in_proj"]) == 2

    def test_build_group_tensors_by_layer(self, registry):
        gt = build_group_tensors(registry, "by-layer")
        assert "blk.0.shortconv.in_proj.weight" in gt["layer_0-3"]
        assert "blk.0.ffn_gate.weight" in gt["layer_0-3"]
        assert "blk.1.shortconv.in_proj.weight" in gt["layer_0-3"]

    def test_build_group_tensors_by_layer_tensor(self, registry):
        gt = build_group_tensors(registry, "by-layer-tensor")
        assert "blk.0.shortconv.in_proj.weight" in gt["conv_in_proj_layer_0-3"]
        assert "blk.1.shortconv.in_proj.weight" in gt["conv_in_proj_layer_0-3"]
        assert "blk.2.shortconv.in_proj.weight" not in gt["conv_in_proj_layer_0-3"]
        assert "blk.1.shortconv.in_proj.weight" in gt["conv_in_proj_layer_0-3"]

    def test_build_group_tensors_individual(self, registry):
        gt = build_group_tensors(registry, "individual")
        assert gt["model.layers.0.conv.in_proj"] == ["blk.0.shortconv.in_proj.weight"]
        assert gt["model.embed_tokens"] == ["token_embd.weight"]

    def test_all_tensors_accounted_for(self, registry):
        for mode in ["by-tensor-type", "by-layer", "by-layer-tensor", "individual"]:
            gt = build_group_tensors(registry, mode)
            all_tensors = set()
            for tensors in gt.values():
                all_tensors.update(tensors)
            expected = {entry["gguf_name"] for entry in registry.registry.values()}
            assert all_tensors == expected, f"Mode {mode} missing tensors"


# ─── end-to-end config generation test ──────────────────────────────────────

class TestConfigGeneration:
    def test_all_four_modes_generate_non_bf16(self, tensor_map, tmp_path):
        """For each mode, the solver should assign non-bf16 to at least some tensors."""
        registry = TensorRegistry(_parse_tensor_map(tensor_map), PARAM_COUNTS, N_BLOCKS)

        # Build group keys for each mode and create mock CE data for ALL groups
        all_groups = {}
        for mode in ["by-tensor-type", "by-layer", "individual", "by-layer-tensor"]:
            pc = build_group_param_counts(registry, mode)
            for gk in pc:
                all_groups[gk] = [
                    {"quant": "Q4_0", "bpw": 4.5, "ce": 0.1},
                    {"quant": "bf16", "bpw": 16.0, "ce": 0.0},
                ]

        for mode in ["by-tensor-type", "by-layer", "individual", "by-layer-tensor"]:
            pc = build_group_param_counts(registry, mode)
            gt = build_group_tensors(registry, mode)

            # Write param_counts to temp file
            param_file = tmp_path / f"param_counts_{mode}.json"
            param_file.write_text(json.dumps(pc))

            # Write weights file (tensor names only)
            weights_file = tmp_path / f"weights_{mode}.txt"
            all_names = sorted({name for names in gt.values() for name in names})
            weights_file.write_text('\n'.join(f"{n}=bf16" for n in all_names) + '\n')

            # Write CE results file
            ce_file = tmp_path / f"ce_{mode}.txt"
            with open(ce_file, 'w') as f:
                f.write("CE ELASTICITY ANALYSIS REPORT\n")
                for gk, opts in all_groups.items():
                    if gk not in pc:
                        continue
                    f.write(f"\n{'='*50}\n  TENSOR TYPE : {gk.upper()}\n{'='*50}\n\n")
                    f.write(f"--- FULL CE REPORT ({gk.upper()}) ---\n\n")
                    for opt in opts:
                        f.write(f"MoQ-{gk}-{opt['quant']}.gguf {opt['bpw']} 0.0 0.0 0.0 {opt['ce']}\n")

            # Run a tiny MCKP (budget = 5.0)
            import argparse
            args = argparse.Namespace(
                ce_results=str(ce_file),
                output_dir=str(tmp_path / f"configs_{mode}"),
                base_gguf=None,
                imatrix=None,
                llama_bin_dir="/fake",
                bits_list=[5.0],
                model_name="test",
                tensor_files_dir=str(weights_file),
                dry_run=True,
                layer_block_size=4,
                mapping=tensor_map,
                param_counts=str(param_file),
                quant_mode=mode,
            )
            q = MoQFinalQuantizer(args)
            q.group_tensors = gt

            configs = q.generate_moq_tensor_files(q.parse_ce_report(str(ce_file)))
            assert len(configs) == 1
            content = open(configs[0]["path"]).read()
            non_bf16 = [line for line in content.strip().split('\n') if not line.endswith('=bf16')]
            assert len(non_bf16) > 0, f"Mode {mode} has all bf16 assignments"
