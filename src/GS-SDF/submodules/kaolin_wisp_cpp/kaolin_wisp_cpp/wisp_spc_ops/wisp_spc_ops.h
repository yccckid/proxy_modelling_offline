#pragma once
#include <torch/torch.h>

namespace wisp_spc_ops {
torch::Tensor dilate_points(torch::Tensor points, int level);

std::pair<torch::Tensor, torch::Tensor>
pointcloud_to_octree(const torch::Tensor &pointcloud, int level,
                     const torch::Tensor &attributes = torch::Tensor(),
                     int dilate = 0);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
octree_to_spc(torch::Tensor octree);

torch::Tensor sample_from_depth_intervals(torch::Tensor depth_intervals,
                                          int num_samples);

torch::Tensor expand_pack_boundary(torch::Tensor pack_boundary,
                                   int num_samples);
} // namespace wisp_spc_ops