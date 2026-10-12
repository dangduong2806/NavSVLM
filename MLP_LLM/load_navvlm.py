"""Load the full InternVL base, source Q-Former, saved projections and LoRA."""
from pathlib import Path

from MLP_LLM.checkpoint_assets import read_json, tensor_header
from MLP_LLM.validate_assets import validate


def load_navvlm(config):
    audit = validate(config)
    if not audit["ready_for_inference"]:
        raise ValueError("Cannot load semantic NavVLM:\n- " + "\n- ".join(audit["errors"]))

    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    from Concat_Q_Former.qformer_bridge import attach_qformer_bridge, align_qformer_bridge_runtime
    from MLP_LLM.checkpoint_loading import strict_bridge_load, strict_tracking_load

    runtime, arch = config["runtime"], audit["checkpoint"]["architecture"]
    device, dtype = torch.device(runtime["device"]), getattr(torch, runtime["dtype"])
    checkpoint = Path(config["checkpoint"])
    options = dict(torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
                   revision=config.get("base_revision"), local_files_only=runtime.get("local_files_only", False))
    quantized = runtime.get("load_in_4bit", False)
    if quantized:
        options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                                                           bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        options["device_map"] = {"": str(device)}
    # The local model class does not implement InternLM. Use the full base's class.
    model = AutoModel.from_pretrained(config["base_model"], **options)
    if not quantized:
        model.to(device)
    if not all(hasattr(model, key) for key in ("vision_model", "language_model", "mlp1", "conv_template")):
        raise ValueError("base_model must be a complete InternVL checkpoint, not a standalone InternLM LLM")
    if model.config.llm_config.hidden_size != arch["llm_hidden_size"]:
        raise ValueError("Base LLM hidden size differs from tracking projector")
    visual_width = model.config.vision_config.hidden_size * int(1 / model.config.downsample_ratio) ** 2
    if visual_width != arch["visual_width"]:
        raise ValueError("Base vision width differs from saved Q-Former bridge")
    if getattr(model.config.llm_config, "model_type", None) != "internlm2":
        raise ValueError("This adapter requires an InternLM2-based InternVL model")
    image_size = getattr(model.config, "force_image_size", None) or model.config.vision_config.image_size
    if image_size != config["preprocess"]["image_size"]:
        raise ValueError("Preprocess image size differs from base vision config")

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True, use_fast=False, local_files_only=True)
    adapter_header = tensor_header(checkpoint / "adapter_model.safetensors")
    embedding_shapes = [v["shape"] for k, v in adapter_header.items() if k.endswith(("tok_embeddings.weight", "output.weight"))]
    if not embedding_shapes or any(shape != [len(tokenizer), arch["llm_hidden_size"]] for shape in embedding_shapes):
        raise ValueError("Checkpoint tokenizer size differs from saved LLM embeddings/output")
    if model.language_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.language_model.resize_token_embeddings(len(tokenizer))
    context_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    if context_id is None or context_id == tokenizer.unk_token_id:
        raise ValueError("Checkpoint tokenizer has no IMG_CONTEXT token")
    model.img_context_token_id = context_id
    vocab = read_json(config["vocabulary_file"])
    trajectory = {key: arch[key] for key in ("d_cat", "d_dir", "d_numeric_hidden", "d_traj", "num_layers", "ffn_dim", "num_objects", "numeric_dim")}
    trajectory.update(enabled=True, fusion_mode="concat", num_heads=config["tracking_num_heads"],
                      dropout=config.get("tracking_dropout", 0.0), source_file=config["tracking_file"], vocabulary=vocab)
    branch_config = {
        "qformer": {"enabled": True, "source_model": config["qformer_source"], "revision": config.get("qformer_revision"),
                    "num_query_tokens": arch["num_query_tokens"], "freeze_qformer": True, "freeze_mlp1": True,
                    "local_files_only": runtime.get("local_files_only", False)},
        "trajectory": trajectory,
    }
    attach_qformer_bridge(model, branch_config)
    bridge_report = strict_bridge_load(model, checkpoint)
    tracking_report = strict_tracking_load(model.trajectory_backbone, model.trajectory_token_projector, checkpoint)
    model.language_model = PeftModel.from_pretrained(model.language_model, str(checkpoint), is_trainable=False)
    # PEFT warns rather than raises for some missing keys. Verify saved adapter schema.
    restored = get_peft_model_state_dict(model.language_model, save_embedding_layers=True)
    if set(restored) != set(adapter_header):
        raise ValueError(f"Restored adapter key mismatch: missing={sorted(set(adapter_header)-set(restored))}, extra={sorted(set(restored)-set(adapter_header))}")
    for key, tensor in restored.items():
        if list(tensor.shape) != adapter_header[key]["shape"]:
            raise ValueError(f"Adapter tensor shape mismatch: {key}")
    del restored
    align_qformer_bridge_runtime(model)
    model.eval()
    model.system_message = "You are a navigation assistant for visually impaired users."
    report = {"bridge": bridge_report, "tracking": tracking_report,
              "adapter_tensor_count": len(adapter_header), "model_class": f"{type(model).__module__}.{type(model).__name__}",
              "base_model": config["base_model"], "base_revision": getattr(model.config, "_commit_hash", None) or config.get("base_revision"),
              "qformer_source": config["qformer_source"], "qformer_revision": config.get("qformer_revision"),
              "architecture": arch, "device": str(device), "dtype": str(dtype)}
    return model, tokenizer, report
