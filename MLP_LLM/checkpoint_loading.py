"""Strict loading for the historical concat checkpoint layout."""
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from Tracking_Encoder.trajectory_branch import TrajectoryBackbone, TrajectoryConcatHead


def remap_tracking_state(state):
    """Dropout added in current source shifted Linear indices, not tensors.

    Only the three known old Linear locations are renamed. CLS tensors are
    reported separately because the selected concat forward never uses CLS.
    """
    state = dict(state)
    renames = {}
    for old, new in (
        ("trajectory_backbone.numeric_mlp.2.", "trajectory_backbone.numeric_mlp.3."),
        ("trajectory_backbone.object_mlp.2.", "trajectory_backbone.object_mlp.3."),
        ("trajectory_token_projector.proj.1.", "trajectory_token_projector.proj.2."),
    ):
        for suffix in ("weight", "bias"):
            src, dst = old + suffix, new + suffix
            if src in state:
                if dst in state:
                    raise ValueError(f"Conflicting legacy/current keys: {src}, {dst}")
                state[dst] = state.pop(src)
                renames[src] = dst
    return state, renames


def strict_tracking_load(backbone, projector, checkpoint):
    state, renames = remap_tracking_state(load_file(str(Path(checkpoint) / "trajectory_branch.safetensors"), device="cpu"))
    skipped = sorted(key for key in state if key.startswith("trajectory_cls_head."))
    expected_prefixes = ("trajectory_backbone.", "trajectory_token_projector.", "trajectory_cls_head.")
    unknown = [key for key in state if not key.startswith(expected_prefixes)]
    if unknown:
        raise ValueError(f"Unknown tracking checkpoint modules: {unknown}")
    checked = 0
    for name, module in (("trajectory_backbone", backbone), ("trajectory_token_projector", projector)):
        selected = {key[len(name) + 1:]: value for key, value in state.items() if key.startswith(name + ".")}
        if not all(torch.isfinite(value).all() for value in selected.values()):
            raise ValueError(f"Non-finite checkpoint weights in {name}")
        module.load_state_dict(selected, strict=True)
        checked += len(selected)
    return {"active_tensors_loaded": checked, "key_remaps": renames, "inactive_cls_tensors": skipped}


def strict_bridge_load(model, checkpoint):
    state = load_file(str(Path(checkpoint) / "qformer_bridge.safetensors"), device="cpu")
    expected = {f"{name}.{key}" for name in ("qformer_input_proj", "qformer_to_mlp1_proj") for key in getattr(model, name).state_dict()}
    if set(state) != expected:
        raise ValueError(f"Bridge keys differ: missing={sorted(expected-set(state))}, extra={sorted(set(state)-expected)}")
    for name in ("qformer_input_proj", "qformer_to_mlp1_proj"):
        selected = {key[len(name) + 1:]: tensor for key, tensor in state.items() if key.startswith(name + ".")}
        if not all(torch.isfinite(value).all() for value in selected.values()):
            raise ValueError("Non-finite bridge weights")
        getattr(model, name).load_state_dict(selected, strict=True)
    return {"active_tensors_loaded": len(state)}


def make_tracking(architecture, num_heads, dropout=0.0):
    keys = ("vocab_size", "direction_vocab_size", "d_cat", "d_dir", "d_numeric_hidden", "d_traj", "num_layers", "ffn_dim", "num_objects")
    backbone = TrajectoryBackbone(**{key: architecture[key] for key in keys}, num_heads=num_heads, dropout=dropout)
    projector = TrajectoryConcatHead(architecture["d_traj"], architecture["llm_hidden_size"], dropout=dropout)
    return backbone, projector


def verify_checkpoint_tensors(checkpoint, architecture):
    """Load all active tracking/bridge weights on CPU and compare every tensor.

    A single attention head is a SHAPE-ONLY construction choice here: state
    shapes do not encode head count. No forward pass is run or claimed.
    """
    backbone, projector = make_tracking(architecture, num_heads=1)
    tracking_report = strict_tracking_load(backbone, projector, checkpoint)
    holder = nn.Module()
    holder.qformer_input_proj = nn.Sequential(nn.LayerNorm(architecture["visual_width"]), nn.Linear(architecture["visual_width"], architecture["qformer_memory_width"]))
    holder.qformer_to_mlp1_proj = nn.Sequential(nn.LayerNorm(architecture["qformer_hidden_size"]), nn.Linear(architecture["qformer_hidden_size"], architecture["visual_width"]))
    bridge_report = strict_bridge_load(holder, checkpoint)
    tracking_state, _ = remap_tracking_state(load_file(str(Path(checkpoint) / "trajectory_branch.safetensors")))
    bridge_state = load_file(str(Path(checkpoint) / "qformer_bridge.safetensors"))
    for name, module in (("trajectory_backbone", backbone), ("trajectory_token_projector", projector), ("qformer_input_proj", holder.qformer_input_proj), ("qformer_to_mlp1_proj", holder.qformer_to_mlp1_proj)):
        source = tracking_state if name.startswith("trajectory_") else bridge_state
        for key, tensor in module.state_dict().items():
            if not torch.equal(tensor, source[f"{name}.{key}"]):
                raise ValueError(f"Weight verification failed: {name}.{key}")
    return {"tracking": tracking_report, "bridge": bridge_report, "all_active_tensors_equal": True, "scope": "tracking/bridge CPU load only; no forward, base model, or LoRA application"}
