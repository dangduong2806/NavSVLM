"""Inference-only STC-Cacher for local and trust_remote_code InternViT.

Adapted from the selective-key algorithm in third_party/STC/stc/cacher/
reference_forward.py: compute current keys for every token, recompute Q/V,
attention and MLP for dynamic tokens, reuse reference V/attention/MLP elsewhere.
InternViT additionally needs fused QKV, Q/K RMSNorm and LayerScale support.

Usage::

    cache = enable_stc_cacher(model.vision_model, update_token_ratio=0.25)
    with torch.inference_mode():
        for pixels in video:  # ONE frame per call; batch entries are its tiles
            features = model.vision_model(pixel_values=pixels)
    cache.reset()  # required at a video/scene boundary or after changing weights

Tile identities/order must remain stable between calls. Reset explicitly when
the crop layout changes even if the tensor shape stays the same. This is a
stateful, single-stream adapter; concurrent streams need separate model instances.
Cache tensors are runtime state and are never added to the model state_dict.
"""

from dataclasses import dataclass
import math
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class STCCacherConfig:
    update_token_ratio: float = 0.25
    cache_interval: int = 4

    def __post_init__(self):
        if not math.isfinite(self.update_token_ratio) or not 0 < self.update_token_ratio <= 1:
            raise ValueError("update_token_ratio must be in (0, 1]")
        if isinstance(self.cache_interval, bool) or not isinstance(self.cache_interval, int) or self.cache_interval < 1:
            raise ValueError("cache_interval must be a positive integer")


@dataclass
class _Reference:
    key: torch.Tensor
    value: torch.Tensor
    attention: torch.Tensor
    mlp: torch.Tensor


def _gather(tokens, indices):
    return tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


def _merge(reference, selected, indices):
    # Never mutate the reference or an output retained by output_hidden_states.
    result = reference.clone()
    result.scatter_(1, indices.unsqueeze(-1).expand_as(selected), selected)
    return result


def _project(attn, tokens, part):
    # InternViT packs [all Q channels, all K channels, all V channels].
    width = attn.embed_dim
    start, end = part * width, (part + 1) * width
    bias = attn.qkv.bias
    return F.linear(tokens, attn.qkv.weight[start:end], None if bias is None else bias[start:end])


def _attention(attn, query, key, value):
    batch, count, width = query.shape

    def heads(tensor):
        return tensor.reshape(batch, -1, attn.num_heads, width // attn.num_heads).transpose(1, 2)

    query, key, value = heads(query), heads(key), heads(value)
    if attn.use_flash_attn:
        # Packed self-attention cannot represent selected Q against full K/V.
        context = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False, scale=attn.scale,
        )
    else:
        weights = ((query * attn.scale) @ key.transpose(-2, -1)).softmax(dim=-1)
        context = attn.attn_drop(weights) @ value
    context = context.transpose(1, 2).reshape(batch, count, width)
    return attn.proj_drop(attn.proj(context))


class InternViTSTCCacher:
    """Runtime controller returned by enable_stc_cacher().

    References refresh on calls 1, 1 + cache_interval, ... . CLS is always
    recomputed in addition to the requested fraction of spatial patch tokens.
    Training or grad-enabled calls use the original forward and clear the cache.
    A ratio of 1 still executes selective recomputation between references,
    updating every token. Use remove() for an unmodified baseline forward.
    """

    def __init__(self, vision_model, config):
        self.config = config
        self.vision_model = vision_model
        self.encoder = vision_model.encoder
        self.layers = tuple(self.encoder.layers)
        if not self.layers:
            raise ValueError("InternViT encoder must contain at least one layer")
        for layer in self.layers:
            required = ("norm1", "norm2", "attn", "mlp", "ls1", "ls2", "drop_path1", "drop_path2")
            if not all(hasattr(layer, name) for name in required):
                raise TypeError("STC-Cacher requires InternVisionEncoderLayer blocks")
            attn = layer.attn
            if type(attn.qkv) is not nn.Linear or attn.qkv.out_features != 3 * attn.embed_dim:
                raise TypeError("STC-Cacher requires an unwrapped InternViT fused QKV Linear")

        self._original_encoder_forward = self.encoder.forward
        self._original_layer_forwards = tuple(layer.forward for layer in self.layers)
        # Preserve pre-existing instance overrides on removal.
        self._overrides = [(module, module.__dict__.get("forward")) for module in (self.encoder, *self.layers)]
        self._removed = False
        self.reset()
        controller = self

        def encoder_forward(encoder, inputs_embeds, *args, **kwargs):
            return controller._forward(inputs_embeds, *args, **kwargs)

        self.encoder.forward = MethodType(encoder_forward, self.encoder)
        for index, layer in enumerate(self.layers):
            def layer_forward(block, hidden_states, _index=index):
                return controller._layer_forward(_index, hidden_states)
            layer.forward = MethodType(layer_forward, layer)

    def reset(self):
        """Release all references before a new video, scene, crop layout or weights."""
        self._references = {}
        self._signature = None
        self._active = False
        self._refresh = True
        self.frame_index = 0
        self.last_mode = None

    def remove(self):
        """Restore original methods and release cached tensors; safe to repeat."""
        if self._removed:
            return
        for module, override in self._overrides:
            if override is None:
                delattr(module, "forward")
            else:
                module.forward = override
        self.reset()
        delattr(self.vision_model, "_stc_cacher")
        self._removed = True

    def _forward(self, inputs_embeds, *args, **kwargs):
        if self._active:
            raise RuntimeError("STC-Cacher cannot share one model across concurrent streams")
        if self.vision_model.training or self.encoder.training or torch.is_grad_enabled() or any(layer.training for layer in self.layers):
            self.reset()
            return self._original_encoder_forward(inputs_embeds, *args, **kwargs)
        signature = (tuple(inputs_embeds.shape), inputs_embeds.device, inputs_embeds.dtype)
        if signature != self._signature:
            self.reset()
            self._signature = signature
        self._refresh = self.frame_index % self.config.cache_interval == 0
        self.last_mode = "reference" if self._refresh else "selective"
        self._active = True
        try:
            result = self._original_encoder_forward(inputs_embeds, *args, **kwargs)
        except Exception:
            # Do not reuse a partially refreshed reference after a failed frame.
            self.reset()
            raise
        finally:
            self._active = False
        self.frame_index += 1
        return result

    def _reference_forward(self, index, hidden_states):
        layer = self.layers[index]
        captured = {}

        def capture_qkv(module, args, output):
            width = layer.attn.embed_dim
            key = output[..., width:2 * width]
            if layer.attn.qk_normalization:
                key = layer.attn.k_norm(key)
            captured["key"] = key.detach().clone()
            captured["value"] = output[..., 2 * width:].detach().clone()

        def capture_attention(module, args, output):
            captured["attention"] = output.detach().clone()

        def capture_mlp(module, args, output):
            captured["mlp"] = output.detach().clone()

        # Capture the native full forward: preserve its kernel and dtype casts.
        handles = [
            layer.attn.qkv.register_forward_hook(capture_qkv),
            layer.attn.register_forward_hook(capture_attention),
            layer.mlp.register_forward_hook(capture_mlp),
        ]
        try:
            result = self._original_layer_forwards[index](hidden_states)
        finally:
            for handle in handles:
                handle.remove()
        self._references[index] = _Reference(**captured)
        return result

    def _layer_forward(self, index, hidden_states):
        if not self._active:
            return self._original_layer_forwards[index](hidden_states)
        reference = self._references.get(index)
        if self._refresh or reference is None:
            return self._reference_forward(index, hidden_states)
        layer = self.layers[index]
        attn = layer.attn
        normalized = layer.norm1(hidden_states).to(hidden_states.dtype)
        key = _project(attn, normalized, 1)
        projection_dtype = key.dtype
        if attn.qk_normalization:
            key = attn.k_norm(key)
        if (key.dtype != reference.key.dtype or key.device != reference.key.device
                or projection_dtype != reference.value.dtype):
            # Also handles autocast changes when input dtype itself is unchanged.
            # RMSNorm can promote K back to fp32 while V remains bf16/fp16.
            self._refresh = True
            self.last_mode = "reference"
            return self._reference_forward(index, hidden_states)

        if self.config.update_token_ratio == 1:
            # Exercise the SAME selective Q/V, attention, MLP and merge path.
            # No similarity ranking is needed when every token is updated.
            indices = torch.arange(hidden_states.shape[1], device=hidden_states.device)
            indices = indices.unsqueeze(0).expand(hidden_states.shape[0], -1)
        else:
            patches = hidden_states.shape[1] - 1
            count = max(1, int(patches * self.config.update_token_ratio)) if patches else 0
            # Score each tile against its own reference; never mix tile identities.
            scores = F.cosine_similarity(key[:, 1:].float(), reference.key[:, 1:].float(), dim=-1)
            dynamic = scores.topk(count, dim=1, largest=False).indices + 1
            cls = torch.zeros((hidden_states.shape[0], 1), dtype=torch.long, device=hidden_states.device)
            indices = torch.cat((cls, dynamic), dim=1)
        selected = _gather(normalized, indices)
        query = _project(attn, selected, 0)
        if attn.qk_normalization:
            query = attn.q_norm(query)
        value = _merge(reference.value, _project(attn, selected, 2), indices)
        attention = _merge(reference.attention, _attention(attn, query, key, value), indices)
        hidden_states = hidden_states + layer.drop_path1(attention * layer.ls1)

        selected = _gather(hidden_states, indices)
        mlp_selected = layer.mlp(layer.norm2(selected).to(hidden_states.dtype))
        mlp = _merge(reference.mlp, mlp_selected, indices)
        return hidden_states + layer.drop_path2(mlp * layer.ls2)


def enable_stc_cacher(vision_model, update_token_ratio=0.25, cache_interval=4):
    """Attach to InternVisionModel, including one loaded with trust_remote_code.

    Returns the existing controller for identical settings; different settings
    replace it and discard the old stream. No checkpoint weights are changed.
    """
    config = STCCacherConfig(update_token_ratio, cache_interval)
    existing = getattr(vision_model, "_stc_cacher", None)
    if existing is not None:
        if existing.config == config:
            return existing
        existing.remove()
    controller = InternViTSTCCacher(vision_model, config)
    vision_model._stc_cacher = controller
    return controller
