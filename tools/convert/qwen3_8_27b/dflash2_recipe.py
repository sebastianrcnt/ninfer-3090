"""Source-checkpoint transforms for the Qwen3.8-27B DFlash 2 drafter artifact.

Maps `z-lab/Qwen3.8-27B-DFlash2` (equivalently its `incoai` mirror) onto the
object contract in `dflash2_inventory`. The checkpoint carries neither an
embedding table nor an output head: the drafter consumes target hidden states
through `fc` and emits logits through the target's own output head, so the
sidecar contains only the drafter backbone, its convolutions, and the selector.
"""

from __future__ import annotations

from tools.convert.qwen3_6.common.recipe import Concat, TensorRecipe, source
from tools.convert.qwen3_6.common.recipe import (
    validate_recipe_coverage as _common_validate_recipe_coverage,
)

from . import dflash2_inventory as inventory


HIDDEN = inventory.HIDDEN
INTERMEDIATE = inventory.INTERMEDIATE
QUERY_SIZE = inventory.QUERY_SIZE
KV_SIZE = inventory.KV_SIZE
HEAD_DIM = inventory.HEAD_DIM
FEATURE_ROWS = inventory.FEATURE_ROWS
CONV_TAPS = inventory.CONV_KERNEL_SIZE
CONV_PROJECTION_ROWS = inventory.CONV_PROJECTION_ROWS
SELECTOR_RANK = inventory.SELECTOR_RANK
VOCAB_ROWS = inventory.VOCAB_ROWS


def _build_layer_recipes(layer: int) -> tuple[TensorRecipe, ...]:
    source_prefix = f"layers.{layer}."
    object_prefix = f"dflash2/layers/{layer}/"
    recipes: list[TensorRecipe] = [
        TensorRecipe(
            object_prefix + "input_norm",
            source(source_prefix + "input_layernorm.weight", (HIDDEN,)),
        ),
        TensorRecipe(
            object_prefix + "attention/query_key_value",
            Concat(
                (
                    source(source_prefix + "self_attn.q_proj.weight", (QUERY_SIZE, HIDDEN)),
                    source(source_prefix + "self_attn.k_proj.weight", (KV_SIZE, HIDDEN)),
                    source(source_prefix + "self_attn.v_proj.weight", (KV_SIZE, HIDDEN)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            object_prefix + "attention/query_norm",
            source(source_prefix + "self_attn.q_norm.weight", (HEAD_DIM,)),
        ),
        TensorRecipe(
            object_prefix + "attention/key_norm",
            source(source_prefix + "self_attn.k_norm.weight", (HEAD_DIM,)),
        ),
        TensorRecipe(
            object_prefix + "attention/output",
            source(source_prefix + "self_attn.o_proj.weight", (HIDDEN, QUERY_SIZE)),
        ),
        TensorRecipe(
            object_prefix + "post_attention_norm",
            source(source_prefix + "post_attention_layernorm.weight", (HIDDEN,)),
        ),
        TensorRecipe(
            object_prefix + "mlp/gate_up",
            Concat(
                (
                    source(source_prefix + "mlp.gate_proj.weight", (INTERMEDIATE, HIDDEN)),
                    source(source_prefix + "mlp.up_proj.weight", (INTERMEDIATE, HIDDEN)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            object_prefix + "mlp/down",
            source(source_prefix + "mlp.down_proj.weight", (HIDDEN, INTERMEDIATE)),
        ),
    ]
    # One convolution wraps attention and one wraps the MLP. Each keeps its
    # identity-initialised base kernel and the projection that produces the
    # per-group deltas for both the input and the output side.
    for role in ("attention_conv", "mlp_conv"):
        recipes.extend(
            (
                TensorRecipe(
                    object_prefix + f"{role}/base_kernel",
                    source(source_prefix + f"{role}.base_kernel", (2, CONV_TAPS, HIDDEN)),
                ),
                TensorRecipe(
                    object_prefix + f"{role}/kernel_projection",
                    source(
                        source_prefix + f"{role}.kernel_projection.weight",
                        (CONV_PROJECTION_ROWS, HIDDEN),
                    ),
                ),
            )
        )
    return tuple(recipes)


def _build_dflash2_recipes() -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = [
        TensorRecipe(
            "dflash2/feature_projection",
            source("fc.weight", (HIDDEN, FEATURE_ROWS)),
        ),
        TensorRecipe(
            "dflash2/context_norm",
            source("hidden_norm.weight", (HIDDEN,)),
        ),
    ]
    for layer in inventory.LAYERS:
        recipes.extend(_build_layer_recipes(layer))
    recipes.extend(
        (
            TensorRecipe("dflash2/final_norm", source("norm.weight", (HIDDEN,))),
            TensorRecipe(
                "dflash2/selector/hidden_projection",
                source(
                    "candidate_selector.hidden_projection.weight",
                    (SELECTOR_RANK, HIDDEN),
                ),
            ),
            TensorRecipe(
                "dflash2/selector/predecessor_codebook",
                source(
                    "candidate_selector.predecessor_codebook", (VOCAB_ROWS, SELECTOR_RANK)
                ),
            ),
            TensorRecipe(
                "dflash2/selector/successor_codebook",
                source(
                    "candidate_selector.successor_codebook", (VOCAB_ROWS, SELECTOR_RANK)
                ),
            ),
        )
    )
    return tuple(recipes)


RECIPE_SPECS = _build_dflash2_recipes()
RECIPES_BY_NAME = {item.object_name: item for item in RECIPE_SPECS}


def validate_recipe_coverage() -> None:
    """Validate exact output pairing against the drafter object contract."""

    _common_validate_recipe_coverage(RECIPE_SPECS, inventory.TENSOR_SPECS)
    expected = len(inventory.TENSOR_SPECS)
    if len(RECIPE_SPECS) != expected or len(RECIPES_BY_NAME) != expected:
        raise ValueError(
            f"DFlash 2 recipe does not contain exactly {expected} tensor transforms"
        )
