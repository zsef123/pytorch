#include <ATen/core/jit_type.h>
#include <ATen/native/xnnpack/OpContext.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/prepack_folding.h>
#include <torch/csrc/jit/passes/quantization.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/xnnpack_rewrite.h>

namespace torch {
namespace jit {

#ifdef USE_XNNPACK

namespace {

void insertPrePackedLinearOp(std::shared_ptr<Graph>& graph) {
  std::string linear_before_inline = R"(
    graph(%linear, %input, %weight, %bias):
        %r = prim::CallFunction(%linear, %input, %weight, %bias)
        return (%r))";
  std::string prepacked_ops_pattern_before_inline = R"(
    graph(%linear, %input, %weight, %bias):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %output_min_max, %output_min_max)
        %res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        return (%res))";
  std::string linear_pattern = R"(
    graph(%input, %weight, %bias):
        %r = aten::linear(%input, %weight, %bias)
        return (%r))";
  std::string prepacked_ops_pattern = R"(
    graph(%input, %weight, %bias):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %output_min_max, %output_min_max)
        %res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        return (%res))";

  auto filter = [](const Match& match,
                   const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto linear_value = match_vmap.at(vmap.at("linear"));
    auto func_name = graph_rewrite_helper::getFuncName(linear_value);
    if (func_name == "linear") {
      return true;
    }
    return false;
  };

  SubgraphRewriter linear_call_fn_rewriter;
  linear_call_fn_rewriter.RegisterRewritePattern(
      linear_before_inline, prepacked_ops_pattern_before_inline);
  linear_call_fn_rewriter.runOnGraph(graph, filter);

  SubgraphRewriter linear_rewriter;
  linear_rewriter.RegisterRewritePattern(linear_pattern, prepacked_ops_pattern);
  linear_rewriter.runOnGraph(graph);
}

void insertPrePackedConv2dOp(std::shared_ptr<Graph>& graph) {
  // Replace _convolution with conv2d
  graph_rewrite_helper::replaceConvolutionWithConv2d(graph);

  std::string conv_2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = aten::conv2d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string prepacked_ops_conv2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min_max, %output_min_max)
        %r = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        return (%r) )";

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(
      conv_2d_pattern, prepacked_ops_conv2d_pattern);
  rewriter.runOnGraph(graph);
}

bool isClampFusable(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  const auto& match_vmap = match.values_map;
  TORCH_CHECK(
      vmap.find("dummy_min_max") != vmap.end(),
      "Expected to find dummy_min_max Value in the subgraph to be replaced.");
  auto dummy_min_max =
      graph_rewrite_helper::getIValue("dummy_min_max", match_vmap, vmap);

  auto is_fusable = !dummy_min_max || dummy_min_max.value().isNone();

  // Also check if the output_min and output_max values are actually constant.
  // If hardtanh's min/max Value's are not actually constants, we will end up
  // rerouting those values to prepack op. And if they are not constants
  // we will not be able to remove prepacking ops.
  if (vmap.find("output_min") != vmap.end()) {
    // aten::relu pattern does not have output_min/output_max.
    // aten::hardtanh/_ does.
    TORCH_CHECK(
        vmap.find("output_max") != vmap.end(),
        "Expected to find output_max as well given "
        "output_min exist in pattern graph.");
    // If output_min/max are not constant, we get c10::nullopt.
    auto output_min =
        graph_rewrite_helper::getIValue("output_min", match_vmap, vmap);
    auto output_max =
        graph_rewrite_helper::getIValue("output_max", match_vmap, vmap);
    is_fusable = is_fusable && (output_min.has_value() && output_max.has_value());
  }

  return is_fusable;
}

void fuseHardtanhWithPackedOps(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;

  std::string linear_prepack_run_hardtanh_fused = R"(
    graph(%input, %weight, %bias, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias : __torch__.torch.classes.xnnpack.LinearOpContext = prepacked::linear_clamp_prepack(
            %weight, %bias, %output_min, %output_max)
        %res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        return (%res))";

  std::string conv2d_prepack_run_hardtanh_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias : __torch__.torch.classes.xnnpack.Conv2dOpContext = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min, %output_max)
        %r = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        return (%r) )";

  std::string linear_prepack_run_hardtanh = R"(
    graph(%input, %weight, %bias, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        %res = aten::hardtanh(%linear_res, %output_min, %output_max)
        return (%res))";

  rewriter.RegisterRewritePattern(
      linear_prepack_run_hardtanh, linear_prepack_run_hardtanh_fused);

  std::string conv2d_prepack_run_hardtanh = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        %r = aten::hardtanh(%conv2d_res, %output_min, %output_max)
        return (%r) )";

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_hardtanh, conv2d_prepack_run_hardtanh_fused);

  std::string linear_prepack_run_hardtanh_inplace = R"(
    graph(%input, %weight, %bias, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        %res = aten::hardtanh_(%linear_res, %output_min, %output_max)
        return (%res))";

  std::string conv2d_prepack_run_hardtanh_inplace = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        %r = aten::hardtanh_(%conv2d_res, %output_min, %output_max)
        return (%r) )";

  rewriter.RegisterRewritePattern(
      linear_prepack_run_hardtanh_inplace, linear_prepack_run_hardtanh_fused);
  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_hardtanh_inplace, conv2d_prepack_run_hardtanh_fused);

  rewriter.runOnGraph(graph, isClampFusable);
}

void fuseReluWithPackedOps(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;

  std::string linear_prepack_run_relu_fused = R"(
    graph(%input, %weight, %bias, %dummy_min_max):
        %output_min: float = prim::Constant[value=0.0]()
        %output_max: None = prim::Constant()
        %packed_weight_bias : __torch__.torch.classes.xnnpack.LinearOpContext = prepacked::linear_clamp_prepack(
            %weight, %bias, %output_min, %output_max)
        %res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        return (%res))";

  std::string conv2d_prepack_run_relu_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %output_min: float = prim::Constant[value=0.0]()
        %output_max: None = prim::Constant()
        %packed_weight_bias : __torch__.torch.classes.xnnpack.Conv2dOpContext = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min, %output_max)
        %r = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        return (%r) )";

  std::string linear_prepack_run_relu = R"(
    graph(%input, %weight, %bias, %dummy_min_max):
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        %res = aten::relu(%linear_res)
        return (%res))";

  rewriter.RegisterRewritePattern(
      linear_prepack_run_relu, linear_prepack_run_relu_fused);

  std::string conv2d_prepack_run_relu = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        %r = aten::relu(%conv2d_res)
        return (%r) )";

  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_relu, conv2d_prepack_run_relu_fused);

  std::string linear_prepack_run_relu_inplace = R"(
    graph(%input, %weight, %bias, %dummy_min_max):
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        %res = aten::relu_(%linear_res)
        return (%res))";

  std::string conv2d_prepack_run_relu_inplace = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        %r = aten::relu_(%conv2d_res)
        return (%r) )";

  rewriter.RegisterRewritePattern(
      linear_prepack_run_relu_inplace, linear_prepack_run_relu_fused);
  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_relu_inplace, conv2d_prepack_run_relu_fused);
  rewriter.runOnGraph(graph, isClampFusable);
}

} // namespace

void insertPrePackedOps(std::shared_ptr<Graph>& graph) {
  insertPrePackedLinearOp(graph);
  insertPrePackedConv2dOp(graph);
}

void insertPrePackedOps(script::Module& module) {
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    insertPrePackedOps(graph);
  }
  for (script::Module m : module.children()) {
    insertPrePackedOps(m);
  }
}

void fusePrePackedLinearConvWithClamp(script::Module& module) {
  auto graph = module.get_method("forward").graph();
  fuseReluWithPackedOps(graph);
  fuseHardtanhWithPackedOps(graph);
}

void FoldPrePackingOps(script::Module& m) {
  PrePackingOpsFilterFn filter_fn = [](const Node* n) -> bool {
    return (
        (n->kind() ==
         Symbol::fromQualString("prepacked::linear_clamp_prepack")) ||
        n->kind() == Symbol::fromQualString("prepacked::conv2d_clamp_prepack"));
  };
  PrePackingOpsFolder(m, filter_fn, "prepack_folding");
}

void optimizeForMobile(script::Module& m) {
  m = FoldConvBatchNorm2d(m);
  m = freeze_module(m);
  insertPrePackedOps(m);
  FoldPrePackingOps(m);
}

#else

void insertPrePackedOps(std::shared_ptr<Graph>& graph) {
  TORCH_INTERNAL_ASSERT(
      "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

void insertPrePackedOps(script::Module& module) {
  TORCH_INTERNAL_ASSERT(
      "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

void fusePrePackedLinearConvWithClamp(script::Module& module) {
  TORCH_INTERNAL_ASSERT(
      "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

void FoldPrePackingOps(script::Module& m) {
  TORCH_INTERNAL_ASSERT(
      "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

void optimizeForMobile(script::Module& m) {
  TORCH_INTERNAL_ASSERT(
      "Mobile optimizaiton only available with XNNPACK at the moment. "
      "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

#endif
} // namespace jit
} // namespace torch
