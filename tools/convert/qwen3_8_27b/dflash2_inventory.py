"""Persistent-object contract for the Qwen3.8-27B DFlash 2 drafter artifact.

The drafter ships as its own `.ninfer` container rather than as additional
objects inside `qwen3_8_27b.ninfer`. Two properties force that split:

* The drafter is trained and released independently of the target, so binding it
  into the target artifact would make every drafter revision a full 16.67 GiB
  reconversion of weights that did not change.
* Reconversion needs the BF16 Qwen3.8-27B checkpoint. Deployments that already
  hold a converted target artifact do not necessarily still hold its source.

Numeric formats follow the registered DFlash 1 drafter: every streamed matrix is
W8, and norms plus the convolution base kernels stay BF16.

The two selector codebooks are the exception. They are gathered a few rows at a
time -- at most block_size x top_k rows per verification -- rather than streamed,
so quantising them buys no bandwidth while adding error directly to the path
scores that choose between candidates. They stay BF16, at a cost of 127 MiB.
"""

from __future__ import annotations

from tools.convert.qwen3_6_27b import inventory as qwen3_6_inventory


MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "dflash2-groupwise-int"
TARGET_KEY = "qwen3_8_27b_dflash2"
ROLE = "draft"

BF16 = qwen3_6_inventory.BF16
W8 = qwen3_6_inventory.W8

FORMAT_NAMES = qwen3_6_inventory.FORMAT_NAMES
LAYOUT_NAMES = qwen3_6_inventory.LAYOUT_NAMES
ResourceSpec = qwen3_6_inventory.ResourceSpec
StoredObjectSpec = qwen3_6_inventory.StoredObjectSpec
TensorSpec = qwen3_6_inventory.TensorSpec
tensor_spec = qwen3_6_inventory.tensor_spec

# Architecture, from the drafter checkpoint's config.json.
LAYERS = tuple(range(5))
HIDDEN = 5120
INTERMEDIATE = 17408
QUERY_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
QUERY_SIZE = QUERY_HEADS * HEAD_DIM          # 4096
KV_SIZE = KV_HEADS * HEAD_DIM                # 1024
BLOCK_SIZE = 8
SLIDING_WINDOW = 2048
MASK_TOKEN_ID = 248070
VOCAB_ROWS = 248320

CONV_KERNEL_SIZE = 2
CONV_GROUP_SIZE = 16
CONV_GROUPS = HIDDEN // CONV_GROUP_SIZE      # 320
# Both convolution sides, both taps, one coefficient per group.
CONV_PROJECTION_ROWS = 2 * CONV_KERNEL_SIZE * CONV_GROUPS   # 1280

SELECTOR_RANK = 256
SELECTOR_TOP_K = 16

# Target decoder layers whose hidden states feed the drafter.
TARGET_FEATURE_LAYERS = (5, 19, 33, 47, 61)
FEATURE_ROWS = len(TARGET_FEATURE_LAYERS) * HIDDEN           # 25600


def _build_dflash2_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        tensor_spec("dflash2/feature_projection", (HIDDEN, FEATURE_ROWS), W8),
        tensor_spec("dflash2/context_norm", (HIDDEN,), BF16),
    ]
    for layer in LAYERS:
        prefix = f"dflash2/layers/{layer}/"
        specs.extend(
            (
                tensor_spec(prefix + "input_norm", (HIDDEN,), BF16),
                tensor_spec(
                    prefix + "attention/query_key_value",
                    (QUERY_SIZE + 2 * KV_SIZE, HIDDEN),
                    W8,
                ),
                tensor_spec(prefix + "attention/query_norm", (HEAD_DIM,), BF16),
                tensor_spec(prefix + "attention/key_norm", (HEAD_DIM,), BF16),
                tensor_spec(prefix + "attention/output", (HIDDEN, QUERY_SIZE), W8),
                tensor_spec(prefix + "post_attention_norm", (HIDDEN,), BF16),
                tensor_spec(prefix + "mlp/gate_up", (2 * INTERMEDIATE, HIDDEN), W8),
                tensor_spec(prefix + "mlp/down", (HIDDEN, INTERMEDIATE), W8),
                tensor_spec(
                    prefix + "attention_conv/base_kernel",
                    (2, CONV_KERNEL_SIZE, HIDDEN),
                    BF16,
                ),
                tensor_spec(
                    prefix + "attention_conv/kernel_projection",
                    (CONV_PROJECTION_ROWS, HIDDEN),
                    W8,
                ),
                tensor_spec(
                    prefix + "mlp_conv/base_kernel",
                    (2, CONV_KERNEL_SIZE, HIDDEN),
                    BF16,
                ),
                tensor_spec(
                    prefix + "mlp_conv/kernel_projection",
                    (CONV_PROJECTION_ROWS, HIDDEN),
                    W8,
                ),
            )
        )
    specs.extend(
        (
            tensor_spec("dflash2/final_norm", (HIDDEN,), BF16),
            tensor_spec("dflash2/selector/hidden_projection", (SELECTOR_RANK, HIDDEN), W8),
            tensor_spec(
                "dflash2/selector/predecessor_codebook", (VOCAB_ROWS, SELECTOR_RANK), BF16
            ),
            tensor_spec(
                "dflash2/selector/successor_codebook", (VOCAB_ROWS, SELECTOR_RANK), BF16
            ),
        )
    )
    return tuple(specs)


TENSOR_SPECS = _build_dflash2_specs()

# The drafter reuses the target artifact's tokenizer and output head, so the
# sidecar carries no frontend resources of its own.
RESOURCE_SPECS: tuple[ResourceSpec, ...] = ()
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}
