#pragma once

#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/model_view.h>
#include <ninfer/targets/qwen3_6/startup_features.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

namespace ninfer::targets::qwen3_6_27b::detail {

inline constexpr std::size_t kTextLayers          = 64;
inline constexpr std::size_t kFullAttentionLayers = 16;
inline constexpr std::size_t kGdnLayers           = 48;
inline constexpr std::size_t kDFlashLayers        = 5;

struct WeightPlan {
    artifact::ObjectHandle object;
    artifact::NumericFormat format          = artifact::NumericFormat::BF16;
    std::uint32_t weight_scale_divisor_bits = 0;
    std::uint32_t input_scale_divisor_bits  = 0;
};

struct MlpPlan {
    WeightPlan gate_up;
    WeightPlan down;
};

struct SplitAttentionProjectionPlan {
    WeightPlan query_key;
    WeightPlan gate_value;
};

struct FusedAttentionProjectionPlan {
    WeightPlan query_key_gate_value;
};

struct FullAttentionPlan {
    std::variant<SplitAttentionProjectionPlan, FusedAttentionProjectionPlan> projection;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan output;
};

struct SplitGdnInputProjectionPlan {
    WeightPlan query_key;
    WeightPlan value_z;
};

struct FusedGdnInputProjectionPlan {
    WeightPlan query_key_value_z;
};

struct GdnPlan {
    artifact::ObjectHandle a_log;
    artifact::ObjectHandle dt_bias;
    artifact::ObjectHandle convolution;
    artifact::ObjectHandle a_projection;
    artifact::ObjectHandle b_projection;
    std::variant<SplitGdnInputProjectionPlan, FusedGdnInputProjectionPlan> input_projection;
    artifact::ObjectHandle norm;
    WeightPlan output;
};

struct TextLayerPlan {
    artifact::ObjectHandle input_norm;
    FullAttentionPlan attention{};
    GdnPlan gdn{};
    bool is_full_attention = false;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
};

struct MtpPlan {
    artifact::ObjectHandle input_projection;
    artifact::ObjectHandle embedding_norm;
    artifact::ObjectHandle hidden_norm;
    artifact::ObjectHandle input_norm;
    artifact::ObjectHandle query_key_gate_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle output;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
    artifact::ObjectHandle final_norm;
};

struct DFlashConvPlan {
    artifact::ObjectHandle base_kernel;
    WeightPlan kernel_projection;
};

struct DFlashLayerPlan {
    artifact::ObjectHandle input_norm;
    WeightPlan query_key_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan attention_output;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
    DFlashConvPlan attention_conv;
    DFlashConvPlan mlp_conv;
};

struct DFlashSelectorPlan {
    WeightPlan hidden_projection;
    artifact::ObjectHandle predecessor_codebook;
    artifact::ObjectHandle successor_codebook;
};

struct DFlashPlan {
    WeightPlan feature_projection;
    artifact::ObjectHandle context_norm;
    std::array<DFlashLayerPlan, kDFlashLayers> layers;
    artifact::ObjectHandle final_norm;
    DFlashSelectorPlan selector;
};

struct DraftBindingPlan {
    DFlashPlan dflash;
};

struct DraftArtifactLoadPlan {
    DraftBindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

struct BindingPlan {
    qwen3_6::FrontendResourcePlan frontend;
    qwen3_6::StartupFeatures features;

    WeightPlan token_embedding;
    std::array<TextLayerPlan, kTextLayers> text_layers;
    artifact::ObjectHandle final_norm;
    WeightPlan output_head;
    artifact::ObjectHandle draft_head;
    artifact::ObjectHandle draft_head_token_ids;
    MtpPlan mtp;

    qwen3_6::VisionBackbonePlan vision_backbone;
    qwen3_6::VisionMergerInputPlan vision_merger_input;
    artifact::ObjectHandle vision_merger_fc2;
    artifact::ObjectHandle vision_merger_fc2_bias;
    qwen3_6::VisionMergerNormPlan vision_merger_norm;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features);

// Bind the DFlash 2 drafter from its own artifact. Every object is resident: a
// drafter container holds nothing else, so there is no placement choice to make.
DraftArtifactLoadPlan bind_draft_artifact(artifact::Binder& binder);

struct DensePostMixerPayload {
    Weight gate_up;
    Weight down;
};

struct SplitAttentionProjectionPayload {
    Weight query_key;
    Weight gate_value;
};

struct FusedAttentionProjectionPayload {
    Weight query_key_gate_value;
};

using FullAttentionProjectionPayload =
    std::variant<SplitAttentionProjectionPayload, FusedAttentionProjectionPayload>;

struct SplitGdnInputProjectionPayload {
    Weight query_key;
    Weight value_z;
};

struct FusedGdnInputProjectionPayload {
    Weight query_key_value_z;
};

using GdnInputProjectionPayload =
    std::variant<SplitGdnInputProjectionPayload, FusedGdnInputProjectionPayload>;

struct GdnProjectionPayload {
    Tensor a_log;
    Tensor dt_bias;
    Weight a_projection;
    Weight b_projection;
    GdnInputProjectionPayload input_projection;
};

struct MtpAttentionPayload {
    Weight packed;
    Weight query;
    Weight key;
    Weight output_gate;
    Weight value;
};

using RuntimeModelView =
    qwen3_6::ModelView<FullAttentionProjectionPayload, GdnProjectionPayload, DensePostMixerPayload,
                       MtpAttentionPayload, DensePostMixerPayload, qwen3_6::DFlashWeights<kDFlashLayers>,
                       kFullAttentionLayers, kGdnLayers>;
using FullAttentionWeights = RuntimeModelView::FullLayer;
using GdnWeights           = RuntimeModelView::GdnLayer;
using MtpWeights           = RuntimeModelView::MtpLayer;

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized,
                    const DraftBindingPlan* draft_plan,
                    std::optional<artifact::MaterializedArtifact> draft_materialized);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    std::optional<artifact::MaterializedArtifact> draft_backing;
    qwen3_6::FrontendResources frontend;
    RuntimeModelView runtime;
};

class LoadedModel::Impl {
public:
    Impl(WeightsProfile weights_profile_in, BindingPlan plan,
         artifact::MaterializedArtifact materialized,
         std::optional<DraftBindingPlan> draft_plan,
         std::optional<artifact::MaterializedArtifact> draft_materialized)
        : weights_profile(weights_profile_in), draft_bindings(std::move(draft_plan)),
          data(std::move(plan), std::move(materialized),
               draft_bindings.has_value() ? &*draft_bindings : nullptr,
               std::move(draft_materialized)) {}

    WeightsProfile weights_profile;
    // Declared before `data`: the drafter plan outlives the constructor call that
    // reads it through a pointer.
    std::optional<DraftBindingPlan> draft_bindings;
    LoadedModelData data;
};

} // namespace ninfer::targets::qwen3_6_27b::detail
