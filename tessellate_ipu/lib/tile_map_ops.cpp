// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "tile_map_ops.hpp"

#include <iostream>
namespace ipu {

std::vector<poplar::Tensor> TileMapEquation::allocateInputTensors(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs) const {
  FMT_ASSERT(inputs.size() <= inputs_info.size(),
             "Inconsistent input vector size.");

  std::vector<poplar::Tensor> inputs_all;
  int input_idx = 0;
  for (const auto& input_info : inputs_info) {
    if (input_info.isConstantInput()) {
      // Create a replicated constant tensor.
      // TODO: support sharded constant as well.
      const std::string raw_values = input_info.constant_data.decode();
      const auto raw_values_ref =
          poplar::ArrayRef<char>(raw_values.data(), raw_values.size());
      auto t = createReplicatedConstantTensor(graph, input_info.aval.dtype,
                                              input_info.aval.shape,
                                              raw_values_ref, this->tiles);
      inputs_all.push_back(t);
    } else {
      // Keep existing input tensor.
      inputs_all.push_back(inputs[input_idx]);
      input_idx++;
    }
  }
  return inputs_all;
}

std::vector<poplar::Tensor> TileMapEquation::allocateOutputTensors(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs) const {
  FMT_ASSERT(inputs.size() == inputs_info.size(),
             "Inconsistent input vector size.");

  std::vector<poplar::Tensor> outputs;
  for (const auto& outinfo : outputs_info) {
    if (outinfo.iotype == VertexIOType::InOut) {
      // Find the input tensor used as output.
      const auto it = std::find_if(inputs_info.begin(), inputs_info.end(),
                                   [&outinfo](const VertexIOInfo& ininfo) {
                                     return ininfo.name == outinfo.name;
                                   });
      const auto idx = std::distance(inputs_info.begin(), it);
      outputs.push_back(inputs.at(idx));
    } else if (outinfo.iotype == VertexIOType::Out) {
      // Allocate an output tensor with proper shape.
      outputs.push_back(createShardedVariable(graph,
                                              toPoplar(outinfo.aval.dtype),
                                              outinfo.aval.shape, this->tiles));
    } else {
      throw std::runtime_error("Unknown IO type for vertex output tensor.");
    }
  }
  return outputs;
}

std::optional<poplar::Tensor> TileMapEquation::allocateTmpSpaceTensor(
    poplar::Graph& graph) const {
  if (!useTmpSpace()) {
    return std::nullopt;
  }
  return createShardedVariable(graph, toPoplar(tmp_space_aval.dtype),
                               {tmp_space_aval.size()}, this->tiles);
}

void TileMapEquation::add(poplar::Graph& graph, poplar::program::Sequence& prog,
                          const std::vector<poplar::Tensor>& inputs,
                          const std::vector<poplar::Tensor>& outputs,
                          const poplar::DebugContext& debug_prefix) const {
  FMT_ASSERT(inputs.size() == inputs_info.size(),
             "Inconsistent inputs vector size.");
  FMT_ASSERT(outputs.size() == outputs_info.size(),
             "Inconsistent outputs vector size.");
  poplar::DebugContext debug_context(debug_prefix, this->pname);

  // Tensor used for vertex temp. scratch space.
  auto tmp_space_tensor_opt = allocateTmpSpaceTensor(graph);

  poplar::ComputeSet cs = graph.addComputeSet(debug_context);
  for (size_t tidx = 0; tidx < tiles.size(); ++tidx) {
    const auto tile = tiles[tidx];
    // Add vertex on the tile.
    auto v = graph.addVertex(cs, this->vname);
    graph.setTileMapping(v, tile);
    if (perf_estimate > 0) {
      graph.setPerfEstimate(v, perf_estimate);
    }
    // Map/connect vertex input tensors.
    for (size_t k = 0; k < inputs.size(); ++k) {
      const auto& info = inputs_info[k];
      const auto tensor = info.connectReshape(inputs[k][tidx]);
      graph.connect(v[info.name], tensor);
    }
    // Map/connect vertex output tensors.
    for (size_t k = 0; k < outputs.size(); ++k) {
      // InOut tensors already mapped. Just need to connect pure output.
      if (outputs_info[k].iotype == VertexIOType::Out) {
        const auto& info = outputs_info[k];
        graph.connect(v[info.name], info.connectReshape(outputs[k][tidx]));
      }
    }
    // Connect tmp scratch space.
    if (tmp_space_tensor_opt.has_value()) {
      auto tmp_space_tensor = tmp_space_tensor_opt.value();
      graph.connect(v[tmp_space_name], tmp_space_tensor[tidx]);
    }
    // Map vertex attributes.
    for (const auto& attr : attributes_i32) {
      graph.setInitialValue(v[attr.name], attr.value);
    }
    for (const auto& attr : attributes_f32) {
      graph.setInitialValue(v[attr.name], attr.value);
    }
  }
  prog.add(poplar::program::Execute(cs, debug_context));
}

std::vector<poplar::Tensor> TileMapEquation::add(
    poplar::Graph& graph, poplar::program::Sequence& prog,
    const std::vector<poplar::Tensor>& inputs,
    const poplar::DebugContext& debug_prefix) const {
  // All input tensors: i.e. add constant tensors.
  const auto inputs_all = this->allocateInputTensors(graph, inputs);

  // No vertex => assume identity function.
  // Forwarding inputs, with just potential change of shape and dtype.
  if (this->vname.empty()) {
    // Check inputs/outputs consistent.
    if (this->numInputs() != this->numOutputs()) {
      throw std::logic_error(
          "Inconsistent number of inputs/outputs for an identity function.");
    }
    // Generate output tensors (potential reshaping + change of dtype).
    std::vector<poplar::Tensor> outputs_all;
    outputs_all.reserve(inputs_all.size());
    for (size_t idx = 0; idx < inputs_all.size(); ++idx) {
      const auto& in = inputs_all[idx];
      const auto& outinfo = outputs_info[idx];
      const auto outshape = shapePrependAxis(tiles.size(), outinfo.aval.shape);
      const auto outdtype = toPoplar(outinfo.aval.dtype);
      auto out = in.reshape(outshape).reinterpret(outdtype);
      outputs_all.push_back(out);
    }
    return outputs_all;
  }
  // Usual path => map a vertex.
  const auto outputs = this->allocateOutputTensors(graph, inputs);
  this->add(graph, prog, inputs_all, outputs, debug_prefix);
  return outputs;
}

std::size_t TileMapEquation::numInOuts() const {
  std::size_t num_inouts0{0}, num_inouts1{0};
  // Check consistency between the in/out collections.
  for (const auto& ininfo : inputs_info) {
    num_inouts0 += (ininfo.iotype == VertexIOType::InOut);
  }
  for (const auto& outinfo : outputs_info) {
    num_inouts1 += (outinfo.iotype == VertexIOType::InOut);
  }
  if (num_inouts0 != num_inouts1) {
    throw std::logic_error(
        "Inconsistent number of in/outs in the IPU tile map equation.");
  }
  // TODO: add checking on tensor size (not necessarily shape).
  return num_inouts0;
}

poplar::program::Program lowerTileMapCallToPoplar(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const TileMapEquation& tile_map_eqn,
    const poplar::DebugContext& debug_context) {
  auto prog = poplar::program::Sequence();
  // IPU tiles synchronization before compute set.
  if (tile_map_eqn.sync) {
    const auto sync_type = poplar::SyncType::INTERNAL;
    prog.add(poplar::program::Sync(sync_type, debug_context));
  }
  outputs = tile_map_eqn.add(graph, prog, inputs, debug_context);
  return prog;
}

}  // namespace ipu
