"""Convert the Qwen3.8-27B DFlash 2 drafter into its sidecar artifact.

Canonical invocation::

    python3 -m tools.convert.qwen3_8_27b.convert_dflash2 \
      --model /path/to/Qwen3.8-27B-DFlash2 \
      --out out/qwen3_8_27b_dflash2.ninfer

The drafter container holds only drafter weights. It carries no frontend
resources and no output head: both come from the target artifact the Engine
loads alongside it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.build_id import stamp_build_id
from tools.artifact.container import ArtifactIdentity, ArtifactObject, ArtifactWriter
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import recipe as family_recipe

from . import dflash2_inventory as inventory
from . import dflash2_recipe as recipe


RECIPE_ID = "qwen3_8_27b_dflash2-v1"

ObjectPlan = family_conversion.ObjectPlan


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    config_summary: dict[str, object]
    source: family_recipe.SourcePreflight
    object_plan: ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def open_source(model_dir: str | Path) -> ShardReader:
    """Open the drafter checkpoint, sharded or not.

    The released drafter is a single `model.safetensors` with no index; a
    sharded re-export would carry one.
    """
    model = Path(model_dir)
    index = model / "model.safetensors.index.json"
    if index.is_file():
        return ShardReader(model)
    single = model / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(
            f"{model} contains neither model.safetensors.index.json nor model.safetensors"
        )
    return ShardReader.from_file(single)


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    """Reject a checkpoint whose architecture the object contract cannot express."""

    dflash = config.get("dflash_config")
    if not isinstance(dflash, Mapping):
        raise ValueError("drafter config.json has no dflash_config block")

    expected: tuple[tuple[str, object, object], ...] = (
        ("hidden_size", config.get("hidden_size"), inventory.HIDDEN),
        ("intermediate_size", config.get("intermediate_size"), inventory.INTERMEDIATE),
        ("num_hidden_layers", config.get("num_hidden_layers"), len(inventory.LAYERS)),
        ("num_attention_heads", config.get("num_attention_heads"), inventory.QUERY_HEADS),
        ("num_key_value_heads", config.get("num_key_value_heads"), inventory.KV_HEADS),
        ("head_dim", config.get("head_dim"), inventory.HEAD_DIM),
        ("vocab_size", config.get("vocab_size"), inventory.VOCAB_ROWS),
        ("sliding_window", config.get("sliding_window"), inventory.SLIDING_WINDOW),
        ("dflash_config.block_size", dflash.get("block_size"), inventory.BLOCK_SIZE),
        (
            "dflash_config.conv_kernel_size",
            dflash.get("conv_kernel_size"),
            inventory.CONV_KERNEL_SIZE,
        ),
        (
            "dflash_config.conv_group_size",
            dflash.get("conv_group_size"),
            inventory.CONV_GROUP_SIZE,
        ),
        ("dflash_config.selector_rank", dflash.get("selector_rank"), inventory.SELECTOR_RANK),
        ("dflash_config.selector_top_k", dflash.get("selector_top_k"), inventory.SELECTOR_TOP_K),
        ("dflash_config.mask_token_id", dflash.get("mask_token_id"), inventory.MASK_TOKEN_ID),
        (
            "dflash_config.target_layer_ids",
            tuple(dflash.get("target_layer_ids") or ()),
            inventory.TARGET_FEATURE_LAYERS,
        ),
    )
    for field, actual, required in expected:
        if actual != required:
            raise ValueError(
                f"drafter config {field}={actual!r} does not match the registered "
                f"contract value {required!r}"
            )

    # The drafter attends bidirectionally inside a block; a causal checkpoint
    # would silently draft a different distribution.
    if config.get("is_causal") is not False:
        raise ValueError("drafter config must declare is_causal=false")

    return {
        "architectures": list(config.get("architectures") or ()),
        "hidden_size": inventory.HIDDEN,
        "num_hidden_layers": len(inventory.LAYERS),
        "block_size": inventory.BLOCK_SIZE,
        "conv_kernel_size": inventory.CONV_KERNEL_SIZE,
        "conv_group_size": inventory.CONV_GROUP_SIZE,
        "selector_rank": inventory.SELECTOR_RANK,
        "selector_top_k": inventory.SELECTOR_TOP_K,
        "mask_token_id": inventory.MASK_TOKEN_ID,
        "target_layer_ids": list(inventory.TARGET_FEATURE_LAYERS),
        "sliding_window": inventory.SLIDING_WINDOW,
        "rope_theta": (config.get("rope_parameters") or {}).get("rope_theta"),
        "rms_norm_eps": config.get("rms_norm_eps"),
    }


def preflight_inventory() -> None:
    if len(inventory.TENSOR_SPECS) != 66 or len(inventory.OBJECT_SPECS) != 66:
        raise ValueError("registered DFlash 2 drafter inventory is incomplete")
    recipe.validate_recipe_coverage()


def build_object_plan() -> ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, {})


def preflight_conversion(model_dir: str | Path) -> ConversionPreflight:
    model = Path(model_dir)
    config = family_conversion.load_json(model / "config.json")
    config_summary = validate_config(config)
    preflight_inventory()
    with open_source(model) as reader:
        source = family_recipe.preflight_source_reader(reader, recipe.RECIPE_SPECS)
    return ConversionPreflight(
        model_dir=model,
        config_summary=config_summary,
        source=source,
        object_plan=build_object_plan(),
    )


def materialize_tensor(spec: inventory.TensorSpec, reader: ShardReader) -> torch.Tensor:
    tensor = family_recipe.materialize_recipe(recipe.RECIPES_BY_NAME[spec.name], reader, None)
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return tensor


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{preflight.source.source_tensor_count} source tensors, device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID)
    with open_source(model) as reader:
        with ArtifactWriter(output, identity, preflight.object_plan.specs) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError("writer object plan differs from completed preflight")
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                tensor = materialize_tensor(spec, reader)
                payload = family_conversion.encode_tensor_payload(
                    tensor, spec, resolved_device
                )
                del tensor
                writer.write(spec.name, payload)
                del payload
                print(f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}", flush=True)

    # The payload is final only once the writer has closed, and the digest covers the
    # payload alone, so stamping is a separate pass rather than part of the write.
    build_id = stamp_build_id(output)
    print(f"build_id {build_id}", flush=True)

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    report = family_conversion.build_conversion_report(
        identity=identity,
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=model,
        out_path=output,
        arguments={
            "model": str(model_dir),
            "out": str(out_path),
            "device": requested_device,
        },
        config_summary=preflight.config_summary,
        source_preflight=preflight.source,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}", flush=True)
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device)


if __name__ == "__main__":
    main()
